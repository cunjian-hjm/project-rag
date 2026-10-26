import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

# 项目根目录（config/ 的上一级），用于定位不随本地工作目录变化的登记文件
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class MinerUConfig:

    mineru_base_url: str
    mineru_api_token: str
    base_url_test: str
    api_token_test: str

    # ==================== 解析任务可调项（均可用 .env 覆盖）====================
    # 轮询超时：MinerU 公共云高峰期排队经常超过 10 分钟，600s 会误报失败
    poll_timeout_seconds: int
    # 轮询间隔：3 秒查一次纯属浪费（单任务 200 次请求），10 秒足够
    poll_interval_seconds: int
    # 解析模型通道：vlm 精度高但公共容量紧张；pipeline 容量大、兼容性最好
    model_version: str
    # 已提交任务的登记表：超时/进程退出后仍能凭 batch_id 把结果捞回来
    pending_file: str

mineru_config = MinerUConfig(
    mineru_base_url=os.getenv("MINERU_BASE_URL", ""),
    mineru_api_token=os.getenv("MINERU_API_TOKEN", ""),
    base_url_test="test",
    api_token_test="test",
    poll_timeout_seconds=int(os.getenv("MINERU_POLL_TIMEOUT_SECONDS") or "1800"),
    poll_interval_seconds=int(os.getenv("MINERU_POLL_INTERVAL_SECONDS") or "10"),
    model_version=os.getenv("MINERU_MODEL_VERSION") or "vlm",
    pending_file=os.getenv("MINERU_PENDING_FILE") or str(PROJECT_ROOT / "runtime" / "mineru_pending.json"),
)
