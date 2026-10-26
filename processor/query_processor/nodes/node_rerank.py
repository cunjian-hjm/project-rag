## processor/query_processor/nodes/node_rerank.py
from typing import Dict, Any, List

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.reranker_http_utils import rerank_documents

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 3 #总数最少条数

# 断崖阈值（绝对，判断高分文档）
RERANK_GAP_ABS: float = 0.5
# 断崖阈值（相对，判断低分文档）
RERANK_GAP_RATIO: float = 0.25

# -----------------------------
# 联网通道（web）降权配置
# -----------------------------
# 背景：本地手册切片与联网搜索结果合并后竟争同一个 Top-K，联网结果只要
# 关键词命中（如问“化学品”撞上“危化品库房安全规范”）就能挤进高位，
# 导致 context_precision 偏低，且会让大模型把外部厂商的说法张冠李戴。
# 产品手册类问答应以本地权威资料为主，联网只作补充。

# 分数惩罚系数：web 文档的 rerank 得分乘于此系数，同等相关度下本地优先
WEB_SCORE_PENALTY: float = 0.8
# 名额上限：最终保留的 web 文档条数上限（超出部分从低到高丢弃）
WEB_MAX_DOCS: int = 2
# 准入最短长度：snippet 低于此字符数视为垃圾（如搜索结果只有“82”）
WEB_MIN_SNIPPET_LEN: int = 50
# 通用最短长度：任何源的空/碎片文档直接丢弃
DOC_MIN_LEN: int = 20

class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行重排序
        流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断 → 联网名额限流
        :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
        :return: 更新后的 state，包含 reranked_docs
        """

        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)

        # 2. Rerank 精排(精排打分，web 分数乘以惩罚系数)
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3. 动态 Top_K 截取(断崖检测)
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 联网通道名额限流（最多 WEB_MAX_DOCS 条 web，名额优先留给本地手册）
        final_docs = self._step_4_apply_web_quota(cutoff_docs)

        # 5. 更新state
        state['reranked_docs'] = final_docs

        # 6. 返回state
        return state

    def _step_1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式（带垃圾文档过滤）"""

        final_docs = []

        # 1. 获取本地 RRF 的文档
        for rrf_doc in state.get('rrf_chunks'):
            content = (rrf_doc.get('content') or "").strip()
            # 空内容或碎片（如只有页码）不进入精排，避免占用 Top-K 名额
            if len(content) < DOC_MIN_LEN:
                logger.debug(f"本地切片过短被丢弃: len={len(content)} content={content[:20]!r}")
                continue
            format_rrf_doc = {
                "content": content,
                "title": rrf_doc.get('item_name'),
                "chunk_id": rrf_doc.get('chunk_id'),
                "url": None,
                "source": "local"
            }
            final_docs.append(format_rrf_doc)

        # 2. 获取 web 远程的文档（联网结果普遍较短且噪声高，准入用更严的 WEB_MIN_SNIPPET_LEN）
        for web_doc in state.get('web_search_docs'):
            content = (web_doc.get('snippet') or "").strip()
            if len(content) < WEB_MIN_SNIPPET_LEN:
                logger.debug(f"联网 snippet 过短被丢弃: len={len(content)} title={web_doc.get('title')!r}")
                continue
            format_web_doc = {
                "content": content,
                "title": web_doc.get('title'),
                "chunk_id": None,
                "url": web_doc.get('url'),
                "source": "web"
            }
            final_docs.append(format_web_doc)

        local_count = sum(1 for d in final_docs if d["source"] == "local")
        web_count = len(final_docs) - local_count
        logger.info(f"合并多源文档: 本地={local_count} 联网={web_count}（已过滤过短文档）")
        return final_docs

    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""

        try:
            user_query = state.get('rewritten_query')
            # 获取文档列表的conten字段组成列表
            contents = [doc.get("content") for doc in merged_multi_docs]
            # 调用Rerank模型：交叉编码器（精排阶段）
            # Query 和 Document 联合编码，精度更高
            rerank_scores = rerank_documents(user_query, contents)

            scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]

            # 联网通道降权：同等相关度下让本地手册排在前面
            # 注：惩罚在排序前施加，因此后续断崖检测用的也是降权后的分数
            for doc in scored_docs:
                if doc.get("source") == "web" and doc.get("score") is not None:
                    doc["score"] = float(doc["score"]) * WEB_SCORE_PENALTY

            sorted_score_docs = sorted(
                scored_docs,
                key=lambda x: x["score"],
                reverse=True
            )

            return sorted_score_docs

        except Exception as e:
            logger.error(f"Rerank 重排序失败: {str(e)}")
            return [{**merged_multi_docs, "score": None}]

    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not ranked_docs:
            return []

        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)

        # 默认值：取满硬上限（最多10条）
        cutoff_pos = upper_bound

        # 遍历范围：从min_topk-1到max_topk-2（索引从0开始），检测相邻两个文档的分数差
        # 例：min_topk=3，max_topk=10 → 遍历i=2,3,4,5,6,7,8（对应第3~9条文档，检测与下一条的差距）
        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            abs_gap = current_score - next_score
            # 计算相对差距：绝对差距 / 当前文档分数（+1e-6避免除数为0/极小值，防止程序报错）
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）。
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                cutoff_pos = idx + 1
                logger.debug(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break

        return ranked_docs[:cutoff_pos]

    def _step_4_apply_web_quota(self, cutoff_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        联网名额限流：经过降权后仍进入 Top-K 的 web 文档，最多保留 WEB_MAX_DOCS 条。

        为什么在断崖截断之后做而不是之前：先让 rerank+降权决定“谁更相关”，
        本步只负责控制“外部资料占比”，不干预相关性排序本身。
        丢弃方向：从分数低往高分丢（保留 web 中最相关的几条，联网结论仍可作补充）。
        输出顺序：保持原有降序位置不变，仅移除超额条目。
        """
        web_docs = [d for d in cutoff_docs if d.get("source") == "web"]
        if len(web_docs) <= WEB_MAX_DOCS:
            return cutoff_docs

        # cutoff_docs 已按分数降序，超出名额的 web 文档就是靠后的那些
        keep_web = {id(d) for d in web_docs[:WEB_MAX_DOCS]}
        dropped = len(web_docs) - WEB_MAX_DOCS
        result = [
            d for d in cutoff_docs
            if d.get("source") != "web" or id(d) in keep_web
        ]
        logger.info(f"联网名额限流: web {len(web_docs)}→{WEB_MAX_DOCS}（丢弃低分 web 文档 {dropped} 条）")
        return result



if __name__ == "__main__":

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "item_name": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "item_name": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "item_name": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "item_name": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(serialize_json(result, indent=4))