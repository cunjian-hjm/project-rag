# eval/diag_milvus.py
"""
Bad Case 诊断工具：核查"关键事实到底是没入库，还是入库了没被召回"
================================================================
用法（项目根目录）：
    uv run python -m eval.diag_milvus 化学品 卡纸 火灾
    uv run python -m eval.diag_milvus 卡纸 火灾 --both        # 同时包含两个关键词
    uv run python -m eval.diag_milvus                        # 仅做全库体检

背景：RAGAS 给出 context_recall 低时，无法区分三种成因
    A. 该事实压根没入库（导入/切分阶段丢失）
    B. 入库了，但向量召回没命中（切片过大导致语义稀释、item_name 过滤挡掉）
    C. 入库了也召回了，但被 RRF / rerank 断崖截断过滤掉
本工具用 Milvus 的 like 过滤直接查底层切片，先把 A 排除，再结合
eval_results.csv 的 retrieved_contexts 判断是 B 还是 C。
"""

import json
import sys
from pathlib import Path

from config.milvus_config import milvus_config
from utils.milvus_utils import get_milvus_client

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def count_by_item(rows):
    dist = {}
    for r in rows:
        key = r.get("item_name")
        dist[key] = dist.get(key, 0) + 1
    return dist


def search_keyword(client, coll, kw):
    rows = client.query(
        collection_name=coll,
        filter=f'content like "%{kw}%"',
        output_fields=["chunk_id", "item_name"],
        limit=200,
    )
    print(f'[含 "{kw}" 的切片] 命中 {len(rows)} 条，按商品名分布：{json.dumps(count_by_item(rows), ensure_ascii=False)}')
    return rows


def dump_both(client, coll, kws):
    """把同时包含所有关键词的切片正文导出，供人工核对原文"""
    expr = " and ".join([f'content like "%{k}%"' for k in kws])
    rows = client.query(
        collection_name=coll,
        filter=expr,
        output_fields=["chunk_id", "item_name", "content"],
        limit=20,
    )
    print(f'\n[同时包含 {kws} 的切片] 命中 {len(rows)} 条')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "diag_hits.txt"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("=== chunk_id=%s item=%s\n%s\n\n" % (r.get("chunk_id"), r.get("item_name"), r.get("content")))
    print(f"  正文已导出：{out}")
    return rows


def is_garbled(text: str) -> bool:
    """
    启发式判定 OCR/编码乱码：正常中文手册里不应大量出现
    带圈字符、彝文、易经卦符、印度文种等 Unicode 区块。
    """
    if not text:
        return False
    bad_ranges = [
        (0x2460, 0x24FF),   # 带圈字符 ⑟
        (0x3200, 0x33FF),   # 编码字符（CJK 兼容）
        (0x4DC0, 0x4DFF),   # 易经卦符
        (0xA000, 0xA4CF),   # 彝文
        (0x0B80, 0x0C7F),   # 泰米尔/泰卢固（中文文档中不应出现）
    ]
    hits = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in bad_ranges))
    return hits >= 5


def health_check(client, coll):
    """全库体检：切片总数、超短垃圾切片、item_name 重复命名（同一手册多份导入）"""
    rows = client.query(collection_name=coll, filter="chunk_id > 0", output_fields=["chunk_id", "item_name", "content"], limit=16000)
    print(f"\n===== 全库体检 =====\n切片总数: {len(rows)}")

    dist = count_by_item(rows)
    print(f"item_name 变体数: {len(dist)}")
    for name, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {name}: {cnt} 条切片")

    short = [r for r in rows if len(str(r.get("content") or "").strip()) < 20]
    print(f"长度 < 20 字符的疑似垃圾切片: {len(short)}")
    for r in short[:10]:
        print(f"  chunk_id={r.get('chunk_id')} content={str(r.get('content'))[:40]!r}")

    garbled = [r for r in rows if is_garbled(str(r.get("content") or ""))]
    print(f"疑似含 OCR 乱码串的切片: {len(garbled)}")
    for r in garbled[:10]:
        print(f"  chunk_id={r.get('chunk_id')} item={r.get('item_name')}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    both_mode = "--both" in sys.argv

    client = get_milvus_client()
    coll = milvus_config.chunks_collection
    print("collections:", client.list_collections())

    if not args:
        health_check(client, coll)
        return

    for kw in args:
        search_keyword(client, coll, kw)

    if both_mode and len(args) >= 2:
        dump_both(client, coll, args)


if __name__ == "__main__":
    main()
