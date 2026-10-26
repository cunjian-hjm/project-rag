# utils/mineru_pending_utils.py
"""
MinerU 已提交任务登记表
====================================================================
要解决的问题：node_pdf_to_md 轮询超时（或后端进程重启）后，导入流程抛异常终止，
但 **MinerU 侧的解析任务并没有被取消**，它还在继续排队/解析。原实现里 batch_id
只出现在异常文本中，进程一死就再也找不回来 —— 结果就是"白等 10 分钟 + 白烧额度"，
只能重新上传重新排队。

本模块把「batch_id ↔ 任务上下文」持久化到 runtime/mineru_pending.json，
让 scripts/mineru_probe.py 随时能凭登记信息复查状态、把已完成的结果捞回来续跑导入。

并发说明：导入接口用 FastAPI BackgroundTasks 并行跑多个文件，多个线程会同时写这份
JSON，因此用进程内锁 + 临时文件原子替换（os.replace）保证不会写坏文件。
跨进程的并发写不保证（评估/恢复脚本都是单进程，够用）。
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.mineru_config import mineru_config
from tool.logger import logger

# 进程内锁：保护「读-改-写」这个非原子序列
_lock = threading.Lock()

# 任务在登记表中记录的状态。注意 timeout_pending 与 failed 的区别：
#   done           MinerU 已解析完成，可直接捞回
#   timeout_pending 我方轮询放弃，但 MinerU 侧仍在排队/解析（最值得捞回）
#   running        MinerU 正在解析
#   failed         MinerU 明确报错（err_msg 非空），捞不回
#   resumed        已捞回并跑完导入后半段
STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_TIMEOUT = "timeout_pending"
STATE_FAILED = "failed"
STATE_RESUMED = "resumed"


def _pending_path() -> Path:
    """登记表路径（可由 MINERU_PENDING_FILE 覆盖），父目录不存在则创建"""
    path = Path(mineru_config.pending_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_all() -> List[Dict[str, Any]]:
    """读取全部登记记录；文件不存在或损坏时返回空列表（绝不因为登记问题阻断导入）"""
    path = _pending_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"MinerU 登记表读取失败（将按空表处理）: {e}, path={path}")
        return []


def _save(records: List[Dict[str, Any]]) -> None:
    """原子写：先写 .tmp 再 replace，避免进程中途被杀留下半截 JSON"""
    path = _pending_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def record_batch(
        batch_id: str,
        file_name: str,
        task_id: str = "",
        file_dir: str = "",
        import_file_path: str = "",
        state: str = STATE_PENDING,
) -> Optional[Dict[str, Any]]:
    """
    登记一个已提交给 MinerU 的解析任务（按 batch_id 幂等 upsert）

    :param batch_id: MinerU 返回的批次 ID，捞回结果唯一的凭证
    :param file_name: PDF 文件名（MinerU 侧的 name）
    :param task_id: 本项目导入任务 ID，用于回显与前端提示
    :param file_dir: 该任务的本地工作目录，捞回结果时要落回这里
    :param import_file_path: 原始 PDF 本地路径（可选，仅作记录）
    :param state: 初始状态，一般为 pending
    :return: 写入后的记录；写入失败返回 None（不抛异常，导入流程优先）
    """
    if not batch_id:
        return None
    try:
        with _lock:
            records = load_all()
            now = datetime.now().isoformat(timespec="seconds")
            for rec in records:
                if rec.get("batch_id") == batch_id:
                    rec.update({"state": state, "updated_at": now})
                    record = rec
                    break
            else:
                record = {
                    "batch_id": batch_id,
                    "file_name": file_name,
                    "task_id": task_id,
                    "file_dir": file_dir,
                    "import_file_path": import_file_path,
                    "state": state,
                    "submitted_at": now,
                    "updated_at": now,
                    "err_msg": "",
                }
                records.append(record)
            _save(records)
        logger.info(f"MinerU 任务已登记: batch_id={batch_id} file={file_name} state={state}")
        return record
    except Exception as e:
        logger.warning(f"MinerU 任务登记失败（不影响导入主流程）: {e}")
        return None


def update_state(batch_id: str, state: str, err_msg: str = "") -> None:
    """更新某个 batch 的状态；记录不存在则忽略（可能是登记功能上线前的老任务）"""
    if not batch_id:
        return
    try:
        with _lock:
            records = load_all()
            changed = False
            for rec in records:
                if rec.get("batch_id") == batch_id:
                    rec["state"] = state
                    rec["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    if err_msg:
                        rec["err_msg"] = err_msg[:500]
                    changed = True
                    break
            if changed:
                _save(records)
    except Exception as e:
        logger.warning(f"MinerU 登记状态更新失败: {e}")


def list_recoverable() -> List[Dict[str, Any]]:
    """
    列出「值得再去 MinerU 查一次」的任务：done 之外的未收尾记录
    （resumed/failed 视为已终结，不再重复查询）
    """
    unfinished = (STATE_PENDING, STATE_RUNNING, STATE_DONE, STATE_TIMEOUT)
    return [r for r in load_all() if r.get("state") in unfinished and r.get("batch_id")]
