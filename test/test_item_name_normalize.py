# test/test_item_name_normalize.py
"""
商品名归一化 + 同品对齐的离线回归测试
为什么要这个文件：归一化逻辑一旦改坏，代价是知识库里同一商品重新裂成多个名字（上次真实
遗留过7条孤儿切片），而这种回归靠肉眼看日志发现不了，必须留一组可重复跑的断言。

不依赖任何外部服务：embedding 与 MilvusClient 全部打桩，直接
    uv run python test/test_item_name_normalize.py
"""
import os
import sys
import types

# 直接 python test/xxx.py 启动时，sys.path[0] 是 test/ 本身，
# 而 test/ 下有一个练习用的 fastapi/ 目录，会把真 fastapi 顶掉（实测报
# ImportError: cannot import name 'Request' from 'fastapi'），所以先把脚本目录从路径里摘掉
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if (os.path.abspath(p) if p else os.getcwd()) != _HERE]
sys.path.insert(0, _ROOT)

# 桩1：屏蔽 BGE-M3 模型加载（本测试不关心向量化）
_fake_embedding = types.ModuleType("utils.embedding_utils")
_fake_embedding.generate_embeddings = lambda texts: {
    "dense": [[0.0] * 1024 for _ in texts],
    "sparse": [{1: 0.1} for _ in texts],
}
sys.modules["utils.embedding_utils"] = _fake_embedding

from utils.item_name_utils import (  # noqa: E402
    extract_model,
    normalize_item_name,
    pick_existing_name,
    same_product,
    suspect_pinyin_brand,
)
from processor.import_processor.nodes import node_item_name_recognition as nr  # noqa: E402


def check_model_extraction():
    """型号抽取：数字段为锚 + 左侧3字母窗口（SCHNHAK180 必须归到 HAK180）"""
    cases = {
        "SCHNHAK180烫金机": "HAK180",
        "兄弟HAK180烫金机": "HAK180",
        "RS-12数字万用表": "RS12",
        "新华三LA2608室内无线网关": "LA2608",
        "5000W电磁炉": "5000",
        "": "",
    }
    for raw, expect in cases.items():
        got = extract_model(raw)
        assert got == expect, f"型号抽取错误: {raw!r} 期望 {expect!r} 实际 {got!r}"


def check_normalize():
    """结构化清洗：文档类型词/公司名/全角/品牌中英别名"""
    cases = {
        "兄弟HAK180烫金机使用说明书": "兄弟HAK180烫金机",
        "Brother HAK180 烫金机": "兄弟HAK180烫金机",
        "ＨＰ LaserJet Pro MFP 1188打印机产品安全手册": "惠普LaserJetProMFP1188打印机",
        "H3CLA2608室内无线网关 用户手册": "新华三LA2608室内无线网关",
        "深圳市某某科技有限公司": "深圳市某某",
    }
    for raw, expect in cases.items():
        got = normalize_item_name(raw)
        assert got == expect, f"清洗错误: {raw!r} 期望 {expect!r} 实际 {got!r}"


def check_same_product():
    """同品判断：该合的合、不该合的绝不能合（误合比分裂更危险）"""
    must_merge = [
        ("兄弟HAK180烫金机", "SCHNHAK180烫金机"),
        ("兄弟HAK180烫金机", "BrotherHAK180烫金机"),
        ("万用表RS-12的使用", "RS-12数字万用表"),
        ("H3CLA2608室内无线网关", "新华三LA2608无线网关"),
    ]
    must_split = [
        ("兄弟HAK180烫金机", "兄弟HAK200烫金机"),      # 同品牌不同型号
        ("RS-12数字万用表", "RS-12打印机"),            # 同型号不同品类
        ("苏伯尓5000W大功率电磁炉", "海尔5000W洗衣机"),  # 数字撞车
        ("兄弟HAK180烫金机", "新华三LA2608室内无线网关"),
        ("兄弟HAK180烫金机", ""),
    ]
    for a, b in must_merge:
        assert same_product(a, b), f"应当判定同品却未合并: {a!r} vs {b!r}"
        assert same_product(b, a), f"同品判断不对称: {a!r} vs {b!r}"
    for a, b in must_split:
        assert not same_product(a, b), f"不应合并却被合并了: {a!r} vs {b!r}"
        assert not same_product(b, a), f"不应合并却被合并了(反向): {b!r} vs {a!r}"


def check_pick_existing():
    """复用库里写法：孤儿名 SCHNHAK180 必须命中已有的 BrotherHAK180烫金机"""
    existing = ["BrotherHAK180烫金机", "新华三LA2608室内无线网关", "RS12万用表"]
    assert pick_existing_name("SCHNHAK180烫金机", existing) == "BrotherHAK180烫金机"
    assert pick_existing_name("万用表RS-12", existing) == "RS12万用表"
    assert pick_existing_name("未知产品X9", existing) is None


def check_suspect_brand():
    """疑似拼音缩写品牌只用于告警"""
    assert suspect_pinyin_brand("SCHNHAK180烫金机") is True
    assert suspect_pinyin_brand("HAK180烫金机") is False


class _FakeClient:
    """假 Milvus 客户端：记录调用、返回预置行，异常分支也能模拟"""

    def __init__(self, rows=None, boom=False):
        self.rows = rows or []
        self.boom = boom
        self.filters = []

    def has_collection(self, collection_name=None):
        return True

    def load_collection(self, collection_name=None):
        return None

    def query(self, collection_name=None, filter=None, output_fields=None, limit=None):
        self.filters.append(filter)
        if self.boom:
            raise RuntimeError("连接中断")
        if filter and filter.startswith("pk > 0"):
            return self.rows
        # item_name=="X" 形式的存在性查询：按 rows 手工匹配
        if filter and filter.startswith('item_name=='):
            target = filter[len('item_name=="'):-1]
            return [r for r in self.rows if r.get("item_name") == target][: (limit or 1)]
        return []


def check_node_methods():
    """节点上的两个新方法：读已有商品名、同品对齐、存在性判断"""
    node = nr.NodeItemNameRecognition()

    # 1. _list_existing_item_names 应去空去重（直接改模块属性打桩，避开真实连接）
    original_cls = nr.MilvusClient
    nr.MilvusClient = lambda uri: _FakeClient(rows=[
        {"item_name": "BrotherHAK180烫金机"},
        {"item_name": "BrotherHAK180烫金机"},
        {"item_name": ""},
        {"item_name": "RS12万用表"},
    ])
    try:
        names = node._list_existing_item_names()
        assert names == ["BrotherHAK180烫金机", "RS12万用表"], f"读取已有商品名异常: {names}"

        # 2. _step_3_5_normalize：孤儿名要对齐到库内写法
        aligned = node._step_3_5_normalize("SCHNHAK180烫金机", "hak180产品安全手册")
        assert aligned == "BrotherHAK180烫金机", f"同品对齐失败: {aligned}"

        # 3. 清洗噪声
        cleaned = node._step_3_5_normalize("兄弟HAK180烫金机 使用说明书", "hak180使用说明书")
        assert cleaned == "BrotherHAK180烫金机", f"清洗+对齐失败: {cleaned}"

        # 4. 全空时用文件标题兜底，绝不返回空商品名
        fallback = node._step_3_5_normalize("说明书", "万用表RS-12的使用")
        assert fallback, "商品名兜底失败，返回了空值"

        # 5. _item_name_exists
        client = _FakeClient(rows=[{"item_name": "BrotherHAK180烫金机"}])
        assert node._item_name_exists(client, "kb_item_names", "BrotherHAK180烫金机") is True
        assert node._item_name_exists(client, "kb_item_names", "全新产品Z9") is False
        assert node._item_name_exists(_FakeClient(boom=True), "kb_item_names", "X") is False
    finally:
        nr.MilvusClient = original_cls


def check_registry_failure_not_fatal():
    """Milvus 挂了也必须放行导入（同品对齐是增强项，不是前置条件）"""
    node = nr.NodeItemNameRecognition()
    original_cls = nr.MilvusClient
    nr.MilvusClient = lambda uri: _FakeClient(boom=True)
    try:
        assert node._list_existing_item_names() == [], "查询失败应当返回空列表而不是抛异常"
        got = node._step_3_5_normalize("兄弟HAK180烫金机", "hak180使用说明书")
        assert got == "兄弟HAK180烫金机", f"库不可用时应保留清洗后的原名: {got}"
    finally:
        nr.MilvusClient = original_cls


if __name__ == "__main__":
    steps = [
        ("型号抽取", check_model_extraction),
        ("结构化清洗", check_normalize),
        ("同品判断", check_same_product),
        ("复用库内写法", check_pick_existing),
        ("疑似缩写品牌告警", check_suspect_brand),
        ("节点方法", check_node_methods),
        ("Milvus异常不阻断", check_registry_failure_not_fatal),
    ]
    for title, fn in steps:
        fn()
        print(f"[PASS] {title}")
    print(f"\n全部通过：{len(steps)} 组断言")
