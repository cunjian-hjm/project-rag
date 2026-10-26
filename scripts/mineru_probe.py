# scripts/mineru_probe.py
"""
MinerU 云端任务探针：查状态 / 补救下载已解析结果
==================================================
背景：node_pdf_to_md 的轮询超时现在可读配置（MINERU_POLL_TIMEOUT_SECONDS，默认 1800s），
但 MinerU 公共云高峰排队仍可能超过它。超时后 **MinerU 侧的任务并没有被取消**，它还在
继续解析，本脚本凭 batch_id（或 runtime/mineru_pending.json 登记表）查真实状态：
    - done    → --resume 直接落回工作目录并跑完导入后半段（或 --download 只拉 ZIP）
    - pending → 还在排队，--watch 持续复查即可
    - failed  → 看 err_msg，通常是 PDF 损坏/加密/超出页数额度

用法（项目根目录）：
    uv run python -m scripts.mineru_probe <batch_id> [<batch_id> ...]   # 按 batch_id 查
    uv run python -m scripts.mineru_probe --pending                      # 查登记表里所有未收尾任务
    uv run python -m scripts.mineru_probe --pending --watch 1800         # 每 30 秒复查，最多等 30 分钟
    uv run python -m scripts.mineru_probe --pending --resume             # 把已 done 的捞回并续跑导入后半段
    uv run python -m scripts.mineru_probe <batch_id> --download --dest D:\\file  # 只下载解压，不续跑

--resume 是关键：MinerU 解析完成后，结果会被落回原 task 工作目录，然后直接执行
node_md_img → node_document_split → node_item_name_recognition → node_bge_embedding
→ node_import_milvus，等于把那次「白等的解析」用完，不必重新上传重新排队。

注意：--resume 会调多模态模型做图片摘要、调 LLM 识别商品名、跑本地 BGE-M3 向量化，
一个文件可能几分钟，且会真的往 Milvus 写数据（kb_chunks 集合不存在会自动重建）。

后台长时间 --watch 建议重定向到文件，读的时候记得指定 UTF-8：
    uv run python -u -m scripts.mineru_probe --pending --watch 5400 --interval 60 > temp-files\watch.log 2>&1
    Get-Content temp-files\watch.log -Encoding UTF8 -Tail 12
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

# Windows 下输出被重定向到文件时 stdout 默认用 GBK，而本脚本回显里有 ✓ ✗ → 这些符号，
# 会直接 UnicodeEncodeError 把工具弄挂（--resume 的日志全中），因此强制改为 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.mineru_config import mineru_config
from utils.mineru_pending_utils import list_recoverable, load_all, record_batch, update_state, STATE_RESUMED


class BatchGoneError(RuntimeError):
    """MinerU 明确拒绝该 batch（不存在/已过期/token 无权限），重试也不会变"""


def fetch_extract_result(batch_id: str) -> List[Dict[str, Any]]:
    """查询单个 batch 的解析结果列表（一个 batch 可含多个文件）"""
    if not mineru_config.mineru_api_token:
        raise SystemExit("缺少 MINERU_API_TOKEN 配置")
    url = f"{mineru_config.mineru_base_url}/extract-results/batch/{batch_id}"
    headers = {"Authorization": f"Bearer {mineru_config.mineru_api_token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != 0:
        raise BatchGoneError(f"batch {batch_id} 查询失败: {payload.get('msg')}")
    return payload["data"].get("extract_result", [])


def show(batch_id: str, raw: bool = False) -> List[Dict[str, Any]]:
    """打印每个文件的 state / err_msg / zip 地址，返回结果列表供后续处理"""
    results = fetch_extract_result(batch_id)
    print(f"\nbatch_id: {batch_id}")
    for r in results:
        print(f"  文件: {r.get('file_name')}")
        print(f"  状态: {r.get('state')}")
        print(f"  页数: {r.get('total_pages', '未知')}  已完成: {r.get('done_pages', '未知')}")
        if r.get("err_msg"):
            print(f"  错误: {r.get('err_msg')}")
        if r.get("full_zip_url"):
            print(f"  ZIP : {r.get('full_zip_url')}")
        if raw:
            # 原始响应字段：用于判断是否被降级到低优先级队列 / 有无排队位次信息
            print("  原始字段: " + json.dumps(r, ensure_ascii=False))
    return results


def download(batch_id: str, results: List[Dict[str, Any]], dest: Path) -> None:
    """把 done 状态的文件 ZIP 下载到 dest/{file_stem}/ 下并解压（与 node_pdf_to_md 目录结构一致）"""
    import shutil
    import zipfile

    for r in results:
        zip_url = r.get("full_zip_url")
        if not zip_url:
            print(f"  跳过 {r.get('file_name')}：状态不是 done，无 ZIP")
            continue
        stem = Path(r.get("file_name", "demo")).stem
        resp = requests.get(zip_url, timeout=120)
        if resp.status_code != 200:
            print(f"  下载失败 {r.get('file_name')}: HTTP {resp.status_code}")
            continue
        out_dir = dest / stem
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / f"{stem}.zip"
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        # MinerU 结果包里通常是 full.md + images/，这里只汇报不重命名，避免干扰正式导入
        print(f"  已解压 {r.get('file_name')} → {out_dir}")
        for p in sorted(out_dir.iterdir()):
            print(f"      {p.name}")


def resume_one(record: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """
    把单个已 done 的 MinerU 任务接回导入流程
    流程：下载 ZIP → 解压到 file_dir/{name}/ → full.md 改名为 {name}.md → 从 node_md_img 起跑完剩余节点
    目录布局与 node_pdf_to_md._step_3_download_and_extract 完全一致，后续节点无感知
    :param record: 登记表中的任务上下文（task_id / file_dir / file_name）
    :param result: MinerU 返回的解析结果（含 full_zip_url）
    :return: 成功续跑返回 True
    """
    # 延迟导入：只查状态时不需要加载 BGE 模型、MinIO、LangGraph 这些重量级依赖
    from processor.import_processor.state import get_default_state
    from processor.import_processor.nodes.node_md_img import NodeMDImg
    from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
    from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
    from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
    from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
    import shutil
    import zipfile

    batch_id = record["batch_id"]
    file_name = record.get("file_name") or ""
    stem = Path(file_name).stem
    file_dir = Path(record.get("file_dir") or "")
    if not stem or not file_dir.exists():
        print(f"  ✗ 跳过 {batch_id}：缺少 file_name 或工作目录不存在（{file_dir}）")
        return False

    zip_url = result.get("full_zip_url")
    resp = requests.get(zip_url, timeout=300)
    if resp.status_code != 200:
        print(f"  ✗ 下载失败 {file_name}: HTTP {resp.status_code}")
        return False

    # 1. 落盘解压（与正常导入一致的目录布局）
    zip_save_path = file_dir / f"{stem}.zip"
    zip_save_path.write_bytes(resp.content)
    extract_dir = file_dir / stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_save_path, "r") as zf:
        zf.extractall(extract_dir)

    full_md = extract_dir / "full.md"
    if not full_md.exists():
        print(f"  ✗ 解压结果里没有 full.md，检查目录：{extract_dir}")
        return False
    md_path_obj = full_md.with_name(f"{stem}.md")
    full_md.rename(md_path_obj)
    print(f"  ✓ 解析结果已落盘：{md_path_obj}（{md_path_obj.stat().st_size} 字节）")

    # 2. 拼导入状态，从 node_md_img 开始跑后半段（跳过 node_pdf_to_md，不重排队）
    # file_title 必须等于 PDF 名（无后缀）：kb_item_names 的幂等清理就是按它删旧数据
    state = get_default_state()
    state.update({
        "task_id": record.get("task_id") or f"resume_{batch_id[:8]}",
        "file_dir": str(file_dir),
        "import_file_path": record.get("import_file_path") or str(file_dir / file_name),
        "pdf_path": record.get("import_file_path") or str(file_dir / file_name),
        "is_pdf_read_enabled": True,
        "file_title": stem,
        "md_path": str(md_path_obj),
        "md_content": md_path_obj.read_text(encoding="utf-8"),
    })

    steps = [
        ("node_md_img", NodeMDImg()),
        ("node_document_split", NodeDocumentSplit()),
        ("node_item_name_recognition", NodeItemNameRecognition()),
        ("node_bge_embedding", NodeBGEEmbedding()),
        ("node_import_milvus", NodeImportMilvus()),
    ]
    for step_name, node in steps:
        print(f"  → {step_name} ...")
        state = node(state)
    print(f"  ✓ 续跑完成：{file_name}，商品名={state.get('item_name')}，切片数={len(state.get('chunks') or [])}")

    update_state(batch_id, STATE_RESUMED)
    return True


def import_local_md(md_input: Path, file_title: str = "", task_id: str = "") -> bool:
    """
    把一份已经解析好的 Markdown 直接送进导入后半段（完全不碰 MinerU）
    适用：MinerU 网页版下载的结果 ZIP、本地 MinerU 的输出、甚至手写 md（只要 md 同级有 images/ 目录）
    目录约定与 node_pdf_to_md 解压结果一致：<某目录>/<名称>.md + <某目录>/images/
    :param md_input: md 文件路径，或包含 md 的目录
    :param file_title: 入库用的文件标题，默认取 md 文件名（无后缀）
    :param task_id: 仅用于日志，不填则用 local_<标题>
    :return: 入库成功返回 True
    """
    from processor.import_processor.state import get_default_state
    from processor.import_processor.nodes.node_md_img import NodeMDImg
    from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
    from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
    from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
    from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus

    if md_input.is_dir():
        candidates = sorted(md_input.glob("*.md"))
        if not candidates:
            print(f"  ✗ 目录里没有 .md 文件：{md_input}")
            return False
        if len(candidates) > 1:
            print(f"  ✗ 目录里有多份 md，请指定具体文件：{[p.name for p in candidates]}")
            return False
        md_path_obj = candidates[0]
    else:
        md_path_obj = md_input

    if not md_path_obj.exists():
        print(f"  ✗ md 文件不存在：{md_path_obj}")
        return False

    title = file_title or md_path_obj.stem
    images_dir = md_path_obj.parent / "images"
    print(f"  md: {md_path_obj}")
    print(f"  file_title: {title}")
    print(f"  images 目录: {images_dir}（存在={images_dir.exists()}，图片数={len(list(images_dir.glob('*')) if images_dir.exists() else [])}）")

    state = get_default_state()
    state.update({
        "task_id": task_id or f"local_{title}",
        "file_dir": str(md_path_obj.parent),
        "import_file_path": str(md_path_obj),
        "is_pdf_read_enabled": False,
        "is_md_read_enabled": True,
        "file_title": title,
        "md_path": str(md_path_obj),
        "md_content": md_path_obj.read_text(encoding="utf-8"),
    })

    steps = [
        ("node_md_img", NodeMDImg()),
        ("node_document_split", NodeDocumentSplit()),
        ("node_item_name_recognition", NodeItemNameRecognition()),
        ("node_bge_embedding", NodeBGEEmbedding()),
        ("node_import_milvus", NodeImportMilvus()),
    ]
    for step_name, node in steps:
        print(f"  → {step_name} ...")
        state = node(state)
    print(f"  ✓ 入库完成：{title}，商品名={state.get('item_name')}，切片数={len(state.get('chunks') or [])}")
    return True


def run_probe(args: argparse.Namespace) -> None:
    """探针主体（拆出来便于整体包进 --out 的输出重定向）"""
    # --import-md：走本地已解析结果直接入库，不查 MinerU、不建 batch，跑完就结束
    if args.import_md:
        import_local_md(Path(args.import_md), args.file_title or "", args.task_id or "")
        return

    # --submit：先建任务，把新 batch_id 加入本轮观察列表
    if args.submit:
        print(f"\n提交解析任务: {args.submit}（通道 {args.model}）")
        args.batch_ids.append(submit(Path(args.submit), args.model, args.task_dir or ""))

    # 确定本轮要查哪些 batch：命令行优先，--pending 那么从登记表补充
    pending_records = list_recoverable() if args.pending else []
    batch_ids = list(args.batch_ids) + [r["batch_id"] for r in pending_records]
    if not batch_ids:
        raise SystemExit("未指定 batch_id，也没有可恢复的登记任务（用 --pending 或传 batch_id）")

    record_by_batch = {r["batch_id"]: r for r in load_all()}
    dest = Path(args.dest)
    deadline = time.time() + args.watch if args.watch else 0

    while True:
        done_batches: List[Dict[str, Any]] = []
        unfinished = 0
        for batch_id in batch_ids:
            # 单次查询必须兜住异常：--watch 要跑几十分钟，mineru.net 拖挡或瞬时连不上
            # 不能把整个复查循环弄挂（否则日志停在某一时刻，反而看不出任务其实还活着）
            try:
                results = show(batch_id, raw=args.raw)
            except BatchGoneError as exc:
                # MinerU 返回 code!=0（token 失效 / batch 已过期）：重试也不会变，当终结处理
                print(f"  ✗ batch_id={batch_id} 已无法查询，不再重试：{exc}")
                continue
            except Exception as exc:
                print(f"  ⚠ 本轮查询失败（下一轮重试）batch_id={batch_id}，错误：{type(exc).__name__}: {exc}")
                unfinished += 1
                continue

            for r in results:
                state = r.get("state")
                if state == "done" and r.get("full_zip_url"):
                    done_batches.append({"batch_id": batch_id, "result": r})
                else:
                    # pending / running / failed 统统计入未收尾，failed 再多轮一次也只会快速返回
                    unfinished += 1

        if args.download:
            for item in done_batches:
                try:
                    download(item["batch_id"], [item["result"]], dest)
                except Exception as exc:
                    print(f"  ✗ 下载失败 batch_id={item['batch_id']}，错误：{exc}")

        if args.resume:
            for item in done_batches:
                batch_id = item["batch_id"]
                record = record_by_batch.get(batch_id)
                if not record:
                    print(f"\n⚠ batch_id={batch_id} 不在登记表里，不知道落回哪个 task 目录，只能 --download 手动处理")
                    continue
                print(f"\n续跑: {record.get('file_name')} (task={record.get('task_id')})")
                # 一个文件续跑失败不能影响其他文件（已成功的会被标 resumed，下次不重跑）
                try:
                    resume_one(record, item["result"])
                except Exception as exc:
                    print(f"  ✗ 续跑失败 {record.get('file_name')}，错误：{type(exc).__name__}: {exc}")
                    print("    修好环境后可重跑：uv run python -m scripts.mineru_probe --pending --resume")

        if not unfinished or not args.watch or time.time() >= deadline:
            if args.watch and unfinished:
                print(f"\n--watch 到时（{args.watch}s），仍有 {unfinished} 个任务未结束，退出轮询")
            break
        print(f"\n还有 {unfinished} 个任务在 MinerU 队列/解析中，{args.interval}s 后复查…")
        time.sleep(args.interval)

    print("\n提示: state=pending 说明还在 MinerU 队列里排队（不是你的代码问题）；")
    print("     若想彻底绕开云端排队，见 MinerU 本地部署手册（--backend pipeline，纯 CPU 可跑）。")


class _Tee:
    """同时写控制台与文件：PowerShell 的 `>` 会把 UTF-8 输出按 GBK 解码产生乱码，自己落一份干净的"""

    def __init__(self, stream, file_handle) -> None:
        self._stream = stream
        self._file = file_handle

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._file.write(text)
        # 必须逐次 flush：文件句柄默认 8KB 块缓冲，--watch 要跑几十分钟，
        # 不 flush 的话日志几乎不增长，看上去像工具挂了
        self._file.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()


def submit(pdf_path: Path, model_version: str, task_dir: str = "") -> str:
    """
    向 MinerU 提交单个本地 PDF（申请上传链接 → PUT 上传 → 登记）
    用途：分诊“到底是 vlm 通道堵，还是账号被降级”——不改 .env、不重启后端，
    也不走导入图（避免把没把握的任务写进库），只建一个可跟踪的 batch。
    :param pdf_path: 本地 PDF 路径
    :param model_version: 解析通道，如 vlm / pipeline
    :param task_dir: 结果想落回的 task 目录（给了就能直接 --resume 入库）
    :return: batch_id
    """
    if not pdf_path.exists():
        raise SystemExit(f"文件不存在：{pdf_path}")
    if not mineru_config.mineru_api_token:
        raise SystemExit("缺少 MINERU_API_TOKEN 配置")

    url = f"{mineru_config.mineru_base_url}/file-urls/batch"
    header = {
        "Authorization": f"Bearer {mineru_config.mineru_api_token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=header, json={
        "files": [{"name": pdf_path.name}],
        "model_version": model_version,
    }, timeout=30)
    payload = resp.json()
    if payload.get("code") != 0:
        raise SystemExit(f"提交失败: code={payload.get('code')} msg={payload.get('msg')}")

    batch_id = payload["data"]["batch_id"]
    file_url = payload["data"]["file_urls"][0]
    with open(pdf_path, "rb") as f:
        up = requests.put(file_url, data=f, timeout=300)
    if up.status_code not in (200, 201, 204):
        print(f"  ✗ 文件 PUT 上传失败：HTTP {up.status_code}（batch 已登记，先 --pending 查状态）")

    record_batch(
        batch_id=batch_id,
        file_name=pdf_path.name,
        task_id=Path(task_dir).name if task_dir else "",
        file_dir=task_dir or str(pdf_path.parent),
        import_file_path=str(pdf_path),
    )
    print(f"  ✓ 已提交 batch_id={batch_id} 文件={pdf_path.name} model_version={model_version}")
    return batch_id


def main() -> None:
    parser = argparse.ArgumentParser(description="MinerU 任务探针 / 超时捞回与续跑")
    parser.add_argument("batch_ids", nargs="*", default=[], help="导入日志或登记表里的 batch_id，可多个")
    parser.add_argument("--pending", action="store_true", help="从 runtime/mineru_pending.json 取未收尾任务")
    parser.add_argument("--download", action="store_true", help="把已 done 的结果 ZIP 下载并解压（不续跑）")
    parser.add_argument("--resume", action="store_true", help="把已 done 的结果落回 task 目录并跑完导入后半段")
    parser.add_argument("--dest", default="./temp-files/mineru", help="--download 时的下载目录")
    parser.add_argument("--raw", action="store_true", help="打印 MinerU 返回的全部原始字段")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="持续复查直到全部结束或超时")
    parser.add_argument("--interval", type=int, default=30, help="--watch 的复查间隔，默认 30 秒")
    parser.add_argument("--out", default=None, help="输出同时写入的文件（带 BOM，Get-Content 直接读不乱码）")
    parser.add_argument("--submit", default=None, metavar="PDF_PATH", help="提交本地 PDF（分诊/补交用）")
    parser.add_argument("--model", default=mineru_config.model_version, help="--submit 的解析通道：vlm / pipeline")
    parser.add_argument("--task-dir", default="", help="--submit 后希望结果落回的 task 目录（供 --resume 用）")
    parser.add_argument("--import-md", default=None, metavar="MD_OR_DIR", help="将本地已解析的 md（+同级 images/）直接入库，不碰 MinerU")
    parser.add_argument("--file-title", default="", help="--import-md 的入库标题，默认取 md 文件名")
    parser.add_argument("--task-id", default="", help="--import-md 的日志用 task_id")
    args = parser.parse_args()

    if args.out:
        # utf-8-sig 带 BOM：PowerShell 5.1 的 Get-Content 不指定 -Encoding 也能认出来
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8-sig") as f, \
                contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            run_probe(args)
        print(f"\n日志已保存: {out_path}")
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
