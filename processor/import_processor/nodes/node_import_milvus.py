# processor/import_processor/nodes/node_import_milvus.py
import json
import logging
from typing import Dict, Any, List

from pymilvus import DataType

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeImportMilvus(BaseNode):
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：Milvus切片数据入库主流程
        执行流程（串行执行，一步一校验，保证数据一致性）：
            1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
            2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
            3. 幂等清理：按 file_title 删除本文件旧切片，避免重复存储，也不误删同商品名的其他文件
            4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
            5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用

        异常处理：
            任一步骤失败抛出ValueError，终止节点执行，保证数据不脏写

        必要参数：task_id、chunks
        更新参数：chunks字段回填chunk_id

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据有效性校验
        chunks_json_data, vector_dimension = self._step_1_check_input(state)

        # 步骤2：Milvus客户端连接+集合准备（自动建表）
        client = self._step_2_prepare_collection(vector_dimension)

        # 步骤3：幂等性处理 - 优先按 file_title 清理本文件旧数据
        self._step_3_clean_old_data(client, chunks_json_data)

        # 步骤4：批量插入数据+主键chunk_id回填
        updated_chunks = self._step_4_insert_data(client, chunks_json_data)

        # 步骤5：更新全局状态，将回填后的切片回传下游
        state["chunks"] = updated_chunks


        return state

    def _step_1_check_input(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """
        步骤1：输入数据有效性校验
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含dense_vector核心字段
            3. 提取向量维度，为集合创建/索引构建提供依据
        参数：
            state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
        返回：
            tuple - (校验通过的切片列表, 稠密向量维度)
        异常：
            任一校验项不通过，抛出ValueError终止入库流程，避免脏数据处理

        """

        # 校验1：chunks非空
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("核心参数chunks为空或非列表类型")

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise ValueError("错误: 数据中缺失dense_vector字段")

        # 校验3：切片包含 sparse_vector 字段
        if 'sparse_vector' not in first_chunk:
            raise ValueError("错误: 数据中缺失sparse_vector字段")

        # 提取向量维度和商品名称，用于后续集合创建/日志展示
        vector_dimension = len(first_chunk['dense_vector'])
        item_name = first_chunk.get('item_name', '未知商品名')
        self.logger.info(
            f"Milvus入库校验通过，待入库切片数：{len(chunks)} | 向量维度：{vector_dimension} | 商品名称：{item_name}")

        return chunks, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension: int):
        """
        步骤2：Milvus客户端连接+集合准备
        核心逻辑：
            1. 获取Milvus单例客户端，验证连接有效性
            2. 集合不存在则自动创建（Schema+索引），存在则直接复用
        参数：
            vector_dimension: int - 稠密向量维度（步骤1提取）
        返回：
            MilvusClient - 已连接、集合准备完成的客户端实例
        异常：
            客户端获取失败/集合名称未配置，抛出ValueError终止流程
        """

        # 1、从环境变量读取 Milvus 核心配置，与 MilvusConfig 配置类保持一致
        collection_name = milvus_config.chunks_collection
        # 从配置文件读取切片集合名称，与配置解耦，便于环境切换

        # 2、配置缺失校验：配置为空则跳过 Milvus 存储，记录警告
        if not collection_name:
            self.logger.error("Milvus集合名称未配置：CHUNKS_COLLECTION_NAME为空")
            raise ValueError("未配置CHUNKS_COLLECTION集合名称")

        # 3、获取 Milvus 单例客户端，连接失败则直接返回
        client = get_milvus_client()
        if not client:
            self.logger.error("Milvus客户端获取失败：get_milvus_client()返回空，连接可能异常")
            raise ValueError("Milvus 连接失败：get_milvus_client() 返回空")

        # 4. 集合不存在则自动创建
        if not client.has_collection(collection_name=collection_name):

            self.logger.info(f"Milvus集合{collection_name}不存在，开始自动创建Schema和索引")
            self._create_collection(client, collection_name, vector_dimension)
        else:
            self.logger.info(f"Milvus集合{collection_name}已存在，直接复用")

        return client

    def _create_collection(self, client, collection_name: str, vector_dimension: int):
        """
        辅助函数：Milvus集合+索引自动创建
        核心逻辑：
            1. 定义集合Schema：包含业务字段+双向量字段，自增主键chunk_id
            2. 构建向量索引：稠密向量用AUTOINDEX（Milvus自动选最优索引），稀疏向量用专用索引
        参数：
            client - MilvusClient实例（已连接）
            collection_name: str - 要创建的集合名称
            vector_dimension: int - 稠密向量维度（与向量化模型保持一致）
        """
        # 1. 创建Schema：自增主键+支持动态字段，适配灵活的业务扩展
        schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)

        # 2. 新增字段：业务字段+主键+双向量字段，字段类型/长度适配业务场景
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)  # 切片标题
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)  # 父标题
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)  # 源文件标题
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)  # 商品名称（幂等性依据）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

        # 3. 构建索引参数：为向量字段创建索引，提升检索性能
        index_params = client.prepare_index_params()
        # 稠密向量索引：AUTOINDEX自动选最优索引类型+余弦相似度（语义检索常用）
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+内积（IP），适配稀疏向量检索
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        # 4. 创建集合：Schema+索引参数结合，一次性完成初始化
        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        self.logger.info(f"Milvus集合创建成功：{collection_name}，向量维度：{vector_dimension}")

    def _step_3_clean_old_data(self, client, chunks_json_data: List[Dict[str, Any]]):
        """
        步骤3：幂等性处理 - 基于 file_title 清理旧数据
        为什么不能用 item_name 做清理键（上轮体检挖出的孤儿根因）：
            切片表 kb_chunks 旧版按 item_name 删除、而注册表 kb_item_names 按 file_title 删除，
            两个清理键不一致：同一个文件重新入库时如果商品名写法变了（旧版无命名约束很常见），
            新名底下没东西可删，旧名那批切片就再也没人引用 → 变成孤儿（实际遗留过7条）。
            商品名归一化后，一个产品多本手册会故意共用同一个 item_name，此时按 item_name
            删除会把这个产品的其他手册切片连带删掉，所以必须回到「文件维度」做幂等。
        兼容：切片里没带 file_title（存量脏数据/跳过切分节点）时，退回按 item_name 清理
        参数：
            client - MilvusClient实例
            chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
        """
        # 提取并去重 file_title：一个文件的所有切片只清一次
        file_titles = sorted({
            str(x.get("file_title", "")).strip()
            for x in chunks_json_data or []
            if str(x.get("file_title", "")).strip()
        })
    
        if file_titles:
            if len(file_titles) > 1:
                self.logger.warning(f"Milvus幂等性清理：本次检测到多个file_title，将逐个清理：{file_titles}")
            for title in file_titles:
                self._clear_chunks_by_file_title(client, title)
            return
    
        # 兜底分支：切片无 file_title，只能按商品名清理
        item_names = sorted({
            str(x.get("item_name", "")).strip()
            for x in chunks_json_data or []
            if str(x.get("item_name", "")).strip()
        })
        if not item_names:
            self.logger.warning("Milvus幂等性清理跳过：切片中既无 file_title 也无有效 item_name")
            return
    
        self.logger.warning("Milvus幂等性清理：切片缺少 file_title，退回按 item_name 清理（可能误删同商品名的其他文件）")
        for i_name in item_names:
            self._clear_chunks_by_item_name(client, i_name)
    
    def _clear_chunks_by_file_title(self, client, file_title: str):
        """
        内部核心函数：根据 file_title 删除本文件上一次入库的全部切片
        为什么这个键是安全的：file_title 来自文件名，同一文件多次入库值恒定，
        删除作用域严格限定在“这个文件自己产生的切片”，不会触碰其他文件的数据
        参数：
            client - MilvusClient实例
            file_title: str - 源文件标题
        异常：
            清理失败抛出ValueError，终止整个入库流程（保证幂等性）
        """
        try:
            # 1. 文件标题安全转义，避免filter表达式报错
            safe_file_title = escape_milvus_string(file_title)
            filter_expr = f'file_title == "{safe_file_title}"'
    
            # 2. 执行删除操作
            client.delete(collection_name=milvus_config.chunks_collection, filter=filter_expr)
            self.logger.info(f"Milvus幂等性清理完成：成功删除file_title={file_title}的旧切片")
    
        except Exception as e:
            self.logger.error(f"Milvus幂等性清理失败：file_title={file_title} | 错误：{str(e)}", exc_info=True)
            raise ValueError(f"幂等清理失败（file_title={file_title}）: {e}")
    
    def _clear_chunks_by_item_name(self, client, item_name: str):
        """
        内部核心函数：根据item_name删除Milvus中的旧切片数据
        参数：
            client - MilvusClient实例
            item_name: str - 要清理的商品名称
        异常：
            清理失败抛出ValueError，终止整个入库流程（保证幂等性）
        """

        try:
            # 1. 商品名称安全转义，避免filter表达式报错
            safe_item_name = escape_milvus_string(item_name)
            filter_expr = f'item_name == "{safe_item_name}"'

            # 2. 执行删除操作
            client.delete(collection_name=milvus_config.chunks_collection, filter=filter_expr)
            self.logger.info(f"Milvus幂等性清理完成：成功删除item_name={item_name}的旧数据")

        except Exception as e:
            self.logger.error(f"Milvus幂等性清理失败：item_name={item_name} | 错误：{str(e)}", exc_info=True)
            raise ValueError(f"幂等清理失败（item_name={item_name}）: {e}")

    def _step_4_insert_data(self, client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        步骤4：批量插入切片数据到Milvus+主键回填
        核心逻辑：
            1. 批量插入数据：提升入库效率，减少Milvus连接次数
            2. 回填chunk_id：将Milvus生成的自增主键回填到切片，供下游业务使用
        参数：
            client - MilvusClient实例
            chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
        返回：
            List[Dict[str, Any]] - 回填了chunk_id的切片列表
        """
        # 1. 预处理数据：移除手动chunk_id，避免与Milvus自增主键冲突
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()

            # 补充 part 字段
            if "part" not in item_copy:
                item_copy["part"] = 0

            # 添加到待插入列表
            data_to_insert.append(item_copy)

        self.logger.info(f"Milvus数据插入：准备{len(data_to_insert)}条切片数据，开始批量插入")

        # 2. 执行批量插入
        insert_result = client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
        insert_count = insert_result.get('insert_count', 0)
        self.logger.info(f"Milvus数据插入完成：成功插入{insert_count}条数据，插入结果：{insert_result}")

        # 3. 主键回填：将Milvus生成的chunk_id回填到原始切片
        inserted_ids = insert_result.get('ids', [])
        if inserted_ids:
            self.logger.info(f"Milvus主键回填：开始将{len(inserted_ids)}个自增chunk_id回填到切片")
            for idx, item in enumerate(chunks_json_data):
                item['chunk_id'] = str(inserted_ids[idx])
            self.logger.info("Milvus主键回填完成：所有切片已绑定chunk_id")

        return chunks_json_data
if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\youxi\wangke\output\hak180产品安全手册\state_vector.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)
    init_state = {
        "chunks": state.get("chunks"),
        # "file_title": "H3C LA2608室内无线网关 用户手册-6W100-整本手册"
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))