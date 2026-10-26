# scripts/reset_kb.py
"""
知识库重置工具：体检 + 清空，为「删掉所有内容重新导入」做准备
================================================================
为什么需要它：项目只有导入接口、没有删除接口，线上想换一批手册只能手搓
Milvus/MinIO/MongoDB 三套客户端。本脚本把三套清理收敛成一条命令，并且
**默认只体检不删除**（避免误操作把库清空）。

用法（项目根目录，PowerShell 先执行 chcp 65001 防中文乱码）：
    uv run python -m scripts.reset_kb                     # 只看现状，什么都不删
    uv run python -m scripts.reset_kb --yes                # 清空 Milvus + MinIO + MongoDB
    uv run python -m scripts.reset_kb --yes --target milvus   # 只清向量库（两张集合）
    uv run python -m scripts.reset_kb --yes --drop         # 连集合一起 drop（改了向量维度/Schema 时用）
    uv run python -m scripts.reset_kb --yes --target local    # 额外清本地 D:\\file 与 temp-files
    uv run python -m scripts.reset_kb --out peek.txt         # 报告另存一份 UTF-8 文件（控制台乱码时用）

清理范围（--target 可选值，逗号分隔）：
    milvus  kb_chunks（切片+向量）与 kb_item_names（商品名注册表）
    minio   图片目录(upload-images) 与 PDF 备份目录(pdf_files)
    mongo   kb001.chat_message（会话历史，含 eval_session_* 评估痕迹）
    local   DATA_BASED_ROOT_DIR(默认 D:\\file) 与 MD_ROOT_DIR(./temp-files) —— 需显式指定，默认不动

注意：Milvus 的 delete 是软删除，脚本会 flush 一次；若你极度在意「物理立刻清空」，
用 --drop，导入流程会在下次插入时按 Schema 自动重建集合。
"""

import argparse
import contextlib
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

from config.milvus_config import milvus_config
from config.minio_config import minio_config
from utils.milvus_utils import get_milvus_client, escape_milvus_string

# 支持清理的目标（同时也是 --yes 时除 local 之外的默认集合）
VALID_TARGETS = ("milvus", "minio", "mongo", "local")
DEFAULT_TARGETS = "milvus,minio,mongo"

# 本地 PDF 在 MinIO 中的对象前缀（与 web/api/import_service.py 的上传规则保持一致）
MINIO_PDF_PREFIX = "pdf_files/"


# ============================================================
# 一、体检
# ============================================================
def peek_milvus(client) -> None:
    """打印两张集合的现状，并交叉校验「商品名注册表」与「切片」是否对得上"""
    print("\n【Milvus】")
    collections = client.list_collections()
    print(f"  当前库中的集合: {collections}")

    chunks_coll = milvus_config.chunks_collection
    names_coll = milvus_config.item_name_collection

    chunk_names: Optional[Counter] = None
    name_pairs: Optional[List[Tuple[str, str]]] = None

    # 1) 切片集合：按 item_name 统计每个商品有多少切片
    if chunks_coll in collections:
        client.load_collection(collection_name=chunks_coll)
        rows = client.query(
            collection_name=chunks_coll,
            filter="chunk_id > 0",
            output_fields=["item_name"],
            limit=16384,
        )
        chunk_names = Counter((r.get("item_name") or "<空>") for r in rows)
        print(f"  {chunks_coll}: 切片总数 = {len(rows)}，商品名 {len(chunk_names)} 个")
        for name, cnt in chunk_names.most_common():
            print(f"      - {name:<28} {cnt} 条")
    else:
        print(f"  {chunks_coll}: 集合不存在")

    # 2) 商品名注册表：同一商品名被注册多次 → 查询时会出现重复候选
    if names_coll in collections:
        client.load_collection(collection_name=names_coll)
        rows = client.query(
            collection_name=names_coll,
            filter="pk > 0",
            output_fields=["item_name", "file_title"],
            limit=1000,
        )
        name_pairs = [(r.get("item_name") or "<空>", r.get("file_title") or "<空>") for r in rows]
        name_counter = Counter(n for n, _ in name_pairs)
        print(f"\n  {names_coll}: 注册条目 = {len(name_pairs)}，不同商品名 = {len(name_counter)}")
        for name, cnt in name_counter.most_common():
            titles = [t for n, t in name_pairs if n == name]
            flag = "  ⚠️ 同名多条" if cnt > 1 else ""
            print(f"      - {name:<28} {cnt} 条{flag}")
            for t in titles:
                print(f"            来源文件: {t}")
    else:
        print(f"\n  {names_coll}: 集合不存在")

    # 3) 交叉校验：孤儿切片（有切片无注册）与空商品（有注册无切片）都会造成检索异常
    if chunk_names is not None and name_pairs is not None:
        registered = {n for n, _ in name_pairs}
        orphans = sorted(set(chunk_names) - registered)
        empties = sorted(registered - set(chunk_names))
        near_dups = _find_near_duplicates(sorted(registered))
        print("\n  交叉校验:")
        print(f"      切片有、注册表无（孤儿，检索时会带出无法定位商品的内容）: {orphans or '无'}")
        print(f"      注册表有、切片无（空商品，命中后必然查不到内容）  : {empties or '无'}")
        print(f"      疑似同一商品的不同写法（会造成召回重复/上下文污染）: {near_dups or '无'}")


def _find_near_duplicates(names: List[str]) -> List[List[str]]:
    """
    粗粒度识别"同一商品的两种写法"：忽略大小写/空格/中英文品牌名后归桶。
    典型场景：同一本手册被识别成 BrotherHAK180烫金机 与 兄弟HAK180烫金机。
    """
    def normalize(s: str) -> str:
        keep = "".join(ch for ch in s.lower() if ch.isalnum())
        for brand in ("brother", "兄弟", "hp", "惠普", "huawei", "华为", "h3c", "newbalance"):
            keep = keep.replace(brand, "")
        return keep

    buckets: Dict[str, List[str]] = {}
    for n in names:
        buckets.setdefault(normalize(n), []).append(n)
    return [sorted(v) for v in buckets.values() if len(v) > 1]


def peek_minio() -> None:
    """打印 MinIO 桶内对象分布（按顶层目录聚合）"""
    print("\n【MinIO】")
    from utils.minio_utils import get_minio_client
    client = get_minio_client()
    if client is None:
        print("  MinIO 客户端不可用，跳过（检查 MINIO_ENDPOINT / 服务是否启动）")
        return
    print(f"  桶: {minio_config.bucket_name}")
    prefixes: Counter = Counter()
    total = 0
    for obj in client.list_objects(minio_config.bucket_name, recursive=True):
        total += 1
        top = obj.object_name.split("/", 1)[0] if "/" in obj.object_name else obj.object_name
        prefixes[top] += 1
    print(f"  对象总数 = {total}")
    for top, cnt in prefixes.most_common():
        print(f"      - {top:<28} {cnt} 个")


def peek_mongo() -> None:
    """打印会话历史规模（评估跑出来的 eval_session_* 也在这里）"""
    print("\n【MongoDB】")
    try:
        from pymongo import MongoClient
    except ImportError:
        print("  未安装 pymongo，跳过")
        return
    mongo_url, db_name = os.getenv("MONGO_URL"), os.getenv("MONGO_DB_NAME")
    try:
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        coll = client[db_name]["chat_message"]
        total = coll.count_documents({})
        sessions = coll.distinct("session_id")
        eval_sessions = [s for s in sessions if str(s).startswith("eval_session")]
        print(f"  库 {db_name}.chat_message: 消息 {total} 条 / 会话 {len(sessions)} 个")
        print(f"      其中评估产生的会话: {len(eval_sessions)} 个")
    except Exception as e:
        print(f"  连接失败，跳过: {e}")


def peek_local() -> None:
    """打印本地导入工作目录（PDF、MinerU 解析产物、临时 MD）"""
    print("\n【本地目录】")
    for env_key, default in (("DATA_BASED_ROOT_DIR", None), ("MD_ROOT_DIR", "./temp-files")):
        root = os.getenv(env_key) or default
        if not root:
            continue
        path = Path(root)
        if not path.exists():
            print(f"  {env_key} = {root}（不存在）")
            continue
        files = list(path.rglob("*"))
        dirs = [p for p in files if p.is_dir()]
        size_mb = sum(p.stat().st_size for p in files if p.is_file()) / 1024 / 1024
        print(f"  {env_key} = {root}：{len(dirs)} 个目录 / {len(files)} 项 / {size_mb:.1f} MB")
        if env_key == "DATA_BASED_ROOT_DIR":
            print("      提示: 这里是「按日期/task_id」分层的上传工作目录，源 PDF 不在这里，删了不影响你的手册原件")


# ============================================================
# 二、清理
# ============================================================
def wipe_milvus(client, drop: bool) -> None:
    """清空 kb_chunks / kb_item_names：默认 delete 行数据，--drop 时直接删集合"""
    print("\n[Milvus] 开始清理")
    specs = [
        (milvus_config.chunks_collection, "chunk_id > 0"),
        (milvus_config.item_name_collection, "pk > 0"),
    ]
    collections = client.list_collections()
    for coll, pk_filter in specs:
        if coll not in collections:
            print(f"  集合 {coll} 不存在，跳过")
            continue
        if drop:
            client.drop_collection(collection_name=coll)
            print(f"  已 drop 集合 {coll}（下次导入会按 Schema 自动重建）")
            continue
        result = client.delete(collection_name=coll, filter=pk_filter)
        print(f"  已删除 {coll} 全部行（filter={pk_filter!r}），结果: {result}")
        # 软删除需要 flush 才落盘，否则 count 可能仍显示旧值
        try:
            client.flush(collection_name=coll)
        except Exception as e:
            print(f"  flush 失败（可忽略，下次导入前会自动生效）: {e}")


def wipe_minio() -> None:
    """删除 MinIO 中的图片目录与 PDF 备份目录（只删本项目写入的前缀，不动别人的数据）"""
    print("\n[MinIO] 开始清理")
    from minio.deleteobjects import DeleteObject
    from utils.minio_utils import get_minio_client
    client = get_minio_client()
    if client is None:
        print("  MinIO 客户端不可用，跳过")
        return

    img_dir = (minio_config.img_dir or "").strip("/")
    prefixes = [p for p in (img_dir, MINIO_PDF_PREFIX.rstrip("/")) if p]
    for prefix in prefixes:
        # 同一目录可能存在带/不带前导斜杠两种写法（历史配置 MINIO_IMG_DIR=/upload-images）
        candidates = {prefix, f"/{prefix}"}
        deleted = 0
        for cand in candidates:
            objects = list(client.list_objects(minio_config.bucket_name, prefix=cand, recursive=True))
            if not objects:
                continue
            errors = client.remove_objects(
                minio_config.bucket_name,
                [DeleteObject(o.object_name) for o in objects],
            )
            for err in errors:
                print(f"  删除失败: {err}")
            deleted += len(objects)
        print(f"  前缀 {prefix}/：删除 {deleted} 个对象")


def wipe_mongo() -> None:
    """清空会话历史，避免旧历史里的 item_names 干扰新库的第一轮提问"""
    print("\n[MongoDB] 开始清理")
    from pymongo import MongoClient
    mongo_url, db_name = os.getenv("MONGO_URL"), os.getenv("MONGO_DB_NAME")
    try:
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        result = client[db_name]["chat_message"].delete_many({})
        print(f"  已删除 chat_message 消息 {result.deleted_count} 条")
    except Exception as e:
        print(f"  清理失败（不影响向量库重置）: {e}")


def wipe_local() -> None:
    """清空本地工作目录内容（保留目录本身），源 PDF 不在此目录"""
    print("\n[本地] 开始清理")
    for env_key, default in (("DATA_BASED_ROOT_DIR", None), ("MD_ROOT_DIR", "./temp-files")):
        root = os.getenv(env_key) or default
        if not root:
            continue
        path = Path(root)
        if not path.exists():
            print(f"  {env_key} = {root}（不存在，跳过）")
            continue
        removed = 0
        for child in path.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed += 1
            except Exception as e:
                print(f"  删除失败 {child}: {e}")
        print(f"  {env_key} = {root}：清理顶层条目 {removed} 个")


# ============================================================
# 三、入口
# ============================================================
def parse_targets(raw: str) -> List[str]:
    """解析 --target，非法值直接报错退出，避免用户拼错导致「以为删了其实没删」"""
    targets = [t.strip() for t in raw.split(",") if t.strip()]
    invalid = [t for t in targets if t not in VALID_TARGETS]
    if invalid:
        raise SystemExit(f"--target 含无效值 {invalid}，可选: {list(VALID_TARGETS)}")
    if not targets:
        raise SystemExit("--target 解析为空，请检查参数")
    return targets


class _Tee:
    """同时写控制台与文件：PowerShell 管道会把 UTF-8 输出按 GBK 解码导致乱码，落盘一份干净的"""

    def __init__(self, stream, file_handle) -> None:
        self._stream = stream
        self._file = file_handle

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库重置工具（默认只体检，加 --yes 才删除）")
    parser.add_argument("--yes", action="store_true", help="确认执行删除；不加此参数只做体检")
    parser.add_argument("--target", default=DEFAULT_TARGETS, help=f"清理范围，逗号分隔，可选 {list(VALID_TARGETS)}")
    parser.add_argument("--drop", action="store_true", help="Milvus 直接 drop 集合而非 delete 行")
    parser.add_argument("--out", default=None, help="报告同时写入的 UTF-8 文件路径")
    args = parser.parse_args()

    targets = parse_targets(args.target)

    if args.out:
        # 自己用 UTF-8 落盘，不走 PowerShell 管道，避开代码页转换导致的乱码
        with open(args.out, "w", encoding="utf-8-sig") as f, contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            _run(targets, args)
        print(f"\n报告已保存: {args.out}")
    else:
        _run(targets, args)


def _run(targets: List[str], args) -> None:
    print("=" * 70)
    print("知识库现状体检")
    print("=" * 70)
    milvus_client = None
    if "milvus" in targets:
        try:
            milvus_client = get_milvus_client()
            peek_milvus(milvus_client)
        except Exception as e:
            print(f"\n【Milvus】连接失败，请确认 Milvus 服务(19530)已启动: {e}")
    if "minio" in targets:
        peek_minio()
    if "mongo" in targets:
        peek_mongo()
    if "local" in targets:
        peek_local()

    if not args.yes:
        print("\n" + "=" * 70)
        print("以上为体检结果，未做任何修改。确认无误后执行：")
        print(f"    uv run python -m scripts.reset_kb --yes --target {args.target}" + (" --drop" if args.drop else ""))
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print(f"执行清理，范围: {targets}" + ("（Milvus drop 集合）" if args.drop else ""))
    print("=" * 70)
    if "milvus" in targets:
        if milvus_client is None:
            print("\n[Milvus] 连接不可用，跳过")
        else:
            wipe_milvus(milvus_client, drop=args.drop)
            # 复核：清理后立即再体检一次，确认计数归零（避免"日志说删了但库里还有"）
            peek_milvus(milvus_client)
    if "minio" in targets:
        wipe_minio()
        peek_minio()
    if "mongo" in targets:
        wipe_mongo()
        peek_mongo()
    if "local" in targets:
        wipe_local()

    print("\n清理完成。重新导入步骤：")
    print("  1) 启动后端（导入接口）: uv run python -m web.api.import_service   # http://127.0.0.1:8000/import.html")
    print("  2) 页面上传源手册 PDF，等待任务状态 completed")
    print("  3) 校验商品名: uv run python -m scripts.reset_kb   # 看「疑似同一商品的不同写法」是否为「无」")
    print("  4) 重新评估: chcp 65001; uv run python -m eval.eval_main --limit 2")


if __name__ == "__main__":
    main()
