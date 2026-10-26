# eval/eval_main.py
"""
知识库 RAG 评估主程序（基于 RAGAS 0.4.3）
====================================================
对应课件《尚硅谷人工智能之知识库的评估与优化》第 6 章，共 6 个 step：

    step_1 初始化评估组件（裁判 LLM、Embedding、Metrics）
    step_2 从 qa.csv 加载测试样本（question + ground_truth）
    step_3 运行项目现有 RAG Pipeline（query_processor），采集 answer + contexts
    step_4 构建 Ragas EvaluationDataset
    step_5 调用 ragas.evaluate() 计算指标
    step_6 导出 eval_results.csv / bad_cases.csv

与课件原始代码的差异（针对本项目做的对接修正）：
    1. 课件 doc.get("text") → 本项目 reranked_docs 的内容字段是 "content"
    2. 课件 query_app.invoke() → 本项目 KBQueryWorkflow().run()（内部 compile+invoke）
    3. 课件 ragas 旧版 Dataset.from_dict → ragas 0.4 的 EvaluationDataset.from_list
    4. 新增 _vertexai_shim：修复 ragas 0.4.3 与 langchain-community 0.4.x 的断链 bug
    5. strip_image_suffix_for_eval：清除答案/上下文中的 Markdown 图片与【图片】区块，
       避免 URL 噪声拉低 faithfulness / answer_relevancy

运行前提：Milvus / MongoDB / reranker 服务 / MinIO 在线（走完整查询链路），
         .env 配置可用（裁判 LLM 复用项目 get_llm_client，即 DashScope 兼容接口）。

运行方式（在项目根目录）：
    uv run python -m eval.eval_main            # 全量评测
    uv run python -m eval.eval_main --limit 3  # 只评前 3 条（首次跑通建议）
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 环境准备：必须先于一切 ragas / 项目模块导入
# ---------------------------------------------------------------------------
# 允许从项目根目录以 `python eval/eval_main.py` 直接运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 兼容补丁：ragas 0.4.3 引用了 langchain_community 中已被移除的 ChatVertexAI
from eval import _vertexai_shim  # noqa: E402

_vertexai_shim.install()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-community.*")

import pandas as pd  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

from config.lm_config import lm_config  # noqa: E402
from eval.embeddings_adapter import ProjectBGEEmbeddings  # noqa: E402
from utils.llm_utils import get_llm_client  # noqa: E402

# 注：KBQueryWorkflow 延迟到 step_3 内部再 import——它的导入链会触发
# utils.mongo_history_utils 在 import 时就连接 MongoDB，离线环境下会直接报错；
# 真正跑评估时 Mongo/Milvus 本来就必须在线，不受影响。

# 评测集与输出文件路径（评估产出统一写到 eval/output/）
EVAL_DIR = Path(__file__).resolve().parent
QA_CSV_PATH = EVAL_DIR / "eval_doc" / "qa.csv"
OUTPUT_DIR = EVAL_DIR / "output"
RESULTS_CSV_PATH = OUTPUT_DIR / "eval_results.csv"
BAD_CASES_CSV_PATH = OUTPUT_DIR / "bad_cases.csv"

# 图片噪声清理用正则（对应课件中被调用但未给出的 strip 实现）
_MD_IMG_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")
_IMAGE_BLOCK_PATTERN = re.compile(r"【图片】.*", re.S)
_URL_PATTERN = re.compile(r"https?://\S+")


def strip_image_suffix_for_eval(text: str) -> str:
    """清除答案文本中的图片区块 / Markdown 图片 / 裸 URL，避免拉低文本类指标"""
    if not text:
        return ""
    text = _IMAGE_BLOCK_PATTERN.sub("", text)
    text = _MD_IMG_PATTERN.sub("", text)
    text = _URL_PATTERN.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 1: 初始化评估组件（LLM、Embedding、Metrics）
# ---------------------------------------------------------------------------
def step_1_init_eval_components():
    """
    裁判 LLM：复用项目 get_llm_client（DashScope OpenAI 兼容接口），
              用 LangchainLLMWrapper 包一层供 ragas 调用。
    Embedding：本地 BGE-M3（只取 dense，评测阶段不做检索，课件 6.3.1 有解释）。
    Metrics  ：faithfulness / answer_relevancy / context_precision /
               context_recall / answer_correctness 五大经典指标。
    """
    # 与生产链路对齐：显式设置环境变量，避免个别 ragas 内部路径直读 env
    os.environ.setdefault("OPENAI_API_KEY", lm_config.api_key or "")
    os.environ.setdefault("OPENAI_API_BASE", lm_config.base_url or "")
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""  # 关闭 ragas 遥测上报尝试

    llm = LangchainLLMWrapper(get_llm_client())
    embeddings = LangchainEmbeddingsWrapper(ProjectBGEEmbeddings())

    metrics = [
        faithfulness,       # 忠实度：答案是否基于检索到的上下文（answer+contexts，LLM 判定）
        answer_relevancy,   # 答案相关性：答案是否真正回答了问题（question+answer，embedding）
        context_precision,  # 上下文精确率：contexts 里有用内容占比、排得靠不靠前（question+contexts+ground_truth）
        context_recall,     # 上下文召回率：contexts 是否覆盖参考答案关键点（ground_truth+contexts，强依赖评测集质量）
        answer_correctness, # 答案正确性：答案与参考答案的语义对齐度（answer+ground_truth）
    ]
    return llm, embeddings, metrics


# ---------------------------------------------------------------------------
# Step 2: 从 CSV 加载测试样本（question + ground_truth）
# ---------------------------------------------------------------------------
def step_2_load_test_samples_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """读取评测集，CSV 列名要求：question, answer（answer 列映射为 ground_truth）"""
    test_samples: List[Dict[str, str]] = []
    if not csv_path.exists():
        print(f"未找到测试数据文件: {csv_path}")
        return test_samples

    print(f"正在从 CSV 加载测试数据: {csv_path}")
    try:
        df_qa = pd.read_csv(csv_path)
        for _, row in df_qa.iterrows():
            question = str(row["question"]).strip()
            ground_truth = str(row["answer"]).strip()
            # 跳过空行/占位注释行，防止脏数据进入评测
            if not question or not ground_truth or question.startswith("#"):
                continue
            test_samples.append({"question": question, "ground_truth": ground_truth})
        print(f"成功加载 {len(test_samples)} 条测试用例。")
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
    return test_samples


# ---------------------------------------------------------------------------
# Step 3: 运行现有 RAG Pipeline，采集 answer 与 contexts
# ---------------------------------------------------------------------------
def step_3_run_pipeline(questions: List[str]) -> Dict[str, List[Any]]:
    """
    逐题调用项目查询流程（processor/query_processor/main_graph.py 的完整 LangGraph），
    从最终 state 提取：
      - answer        → 大模型生成的答案
      - reranked_docs → 重排后进入提示词的 Top-K 文档（Ragas 的 contexts 就是它）
    注意：这会真实走一遍 商品名确认→多路检索→RRF→Rerank→生成 全链路，
         因此 Milvus / MongoDB / reranker / MCP 联网搜索 必须在运行。
    """
    answers: List[str] = []
    contexts: List[List[str]] = []

    print(f"开始运行 Pipeline 推理，共 {len(questions)} 个问题...")
    from processor.query_processor.main_graph import KBQueryWorkflow

    app = KBQueryWorkflow().compile()

    for idx, question in enumerate(questions, start=1):
        print(f"  [{idx}/{len(questions)}] {question[:50]}")
        # eval_ 前缀 session，避免污染生产会话历史（历史仍会写 MongoDB，仅隔离 key）
        input_state = {
            "original_query": question,
            "session_id": f"eval_session_{idx:03d}",
            "message_id": f"eval_message_{idx:03d}",
            "is_stream": False,  # 评估不需要流式
            "history": [],       # 单轮问答，不带上文
        }
        try:
            result_state = app.invoke(input_state)

            ans = result_state.get("answer", "") or ""
            answers.append(strip_image_suffix_for_eval(ans))

            # 本项目 reranked_docs 的字段结构（见 node_rerank.py）：
            # {content, title, chunk_id, url, source, score} —— 内容字段是 content 不是 text！
            reranked_docs = result_state.get("reranked_docs") or []
            doc_texts = []
            for doc in reranked_docs:
                content = (doc.get("content") or "").strip()
                if content:
                    # 切片正文里的 Markdown 图片/URL 对 LLM 裁判是纯噪声，一并清掉
                    doc_texts.append(strip_image_suffix_for_eval(content))
            contexts.append(doc_texts)
        except Exception as e:
            print(f"  Error processing question '{question}': {e}")
            answers.append("Error generating response")
            contexts.append([])

    return {"answer": answers, "contexts": contexts}


# ---------------------------------------------------------------------------
# Step 4: 构建 Ragas 数据集
# ---------------------------------------------------------------------------
def step_4_build_dataset(
    test_data: List[Dict[str, str]], pipeline_results: Dict[str, List[Any]]
) -> EvaluationDataset:
    """
    ragas 0.4 使用 EvaluationDataset（课件旧写法 Dataset.from_dict 已废弃）。
    实测兼容坑：ragas 0.4.3 的 from_list 不会自动把 v1 字段名映射到 v2，
    必须直接用 v2 schema 字段名，否则样本全空、所有指标全为 NaN：
        question       → user_input
        answer         → response
        contexts       → retrieved_contexts（list[str]）
        ground_truth   → reference
    """
    rows = []
    for i, item in enumerate(test_data):
        rows.append(
            {
                "user_input": item["question"],
                "response": pipeline_results["answer"][i],
                "retrieved_contexts": pipeline_results["contexts"][i],
                "reference": item["ground_truth"],
            }
        )
    return EvaluationDataset.from_list(rows)


# ---------------------------------------------------------------------------
# Step 5: 执行 Ragas 评估
# ---------------------------------------------------------------------------
def step_5_run_ragas_evaluation(
    dataset: EvaluationDataset, metrics, llm, embeddings, max_workers: int = 4
):
    """
    ragas.evaluate() 是"指标调度器"：逐行样本 × 逐指标计算分数。
    - raise_exceptions=False：单条坏数据（contexts 为空、解析失败）不中断整体，
      该格记 NaN，正好与 Step 6 的 bad case 筛选（NaN 也算 bad）配套。
    - max_workers：ragas 0.4.3 默认 16，对 DashScope 的 QPM 限流太激进（容易 429 重试风暴），
      压到 4；同时 evaluate() 非常吃内存（课件 6.11 提醒：卡死先把问题数调小）。
    - max_retries：默认 10，降到 3，避免单条坏样本耗掉大量时间与 token。
    注：ragas 0.4.3 的 RunConfig 字段只有 timeout/max_retries/max_wait/max_workers/
        exception_types/log_tenacity/seed；旧版的 section_separator 已不存在。
    """
    print("开始 Ragas 评估（LLM-as-a-Judge，耗时随问题数×指标数线性增长）...")
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=max_workers, timeout=300, max_retries=3),
    )


# ---------------------------------------------------------------------------
# Step 6: 导出结果、筛 Bad Cases
# ---------------------------------------------------------------------------
def step_6_export_results_and_bad_cases(
    results, metrics, bad_case_threshold: float = 0.5
):
    """
    导出两张表：
      eval_results.csv：全量样本分数 + 末尾追加"平均值"行（utf-8-sig 方便 Excel 打开）
      bad_cases.csv   ：answer_correctness < 阈值 或 为 NaN 的样本（定位优化 priorities）
    """
    print("\n========== 评估结果 ==========")
    print(results)

    df = results.to_pandas()
    metric_names = [m.name for m in metrics]

    # 末尾追加平均值行，直接在 CSV 里看整体表现
    avg_row = {col: "" for col in df.columns}
    label_col = next((c for c in ("question", "user_input") if c in df.columns), df.columns[0])
    avg_row[label_col] = "平均值"
    for metric in metric_names:
        if metric in df.columns:
            avg_row[metric] = pd.to_numeric(df[metric], errors="coerce").mean()

    df_to_export = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 首次运行自动创建 output 目录
    df_to_export.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存至: {RESULTS_CSV_PATH}")

    # Bad case：仅以 answer_correctness 为判据（课件 6.13 的定义）
    bad_case_metric = "answer_correctness"
    print(f"正在提取 Bad Cases ({bad_case_metric} 阈值: {bad_case_threshold})...")
    numeric_col = pd.to_numeric(df[bad_case_metric], errors="coerce")
    bad_cases_df = df[numeric_col.lt(bad_case_threshold) | numeric_col.isna()]

    if not bad_cases_df.empty:
        bad_cases_df.to_csv(BAD_CASES_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"Bad cases（{len(bad_cases_df)} 条）已保存至: {BAD_CASES_CSV_PATH}")
        print("接下来请按课件第 7 章做归因：")
        print("  context_recall/precision 低 → 检索问题（chunk/topk/rerank）")
        print("  faithfulness 低            → 生成在编（提示词约束/拒答）")
        print("  其他高但 answer_correctness 低 → 表达风格发散（提示词收敛输出结构）")
    else:
        print(f"恭喜！没有发现 {bad_case_metric} 低于阈值的 Bad Cases。")

    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="知识库 RAGAS 评估")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（0=全量），首次跑通建议 --limit 3")
    parser.add_argument("--threshold", type=float, default=0.5, help="bad case 的 answer_correctness 阈值")
    args = parser.parse_args()

    # step_1: 初始化评估组件
    ragas_llm, ragas_embeddings, ragas_metrics = step_1_init_eval_components()

    # step_2: 加载测试数据
    ragas_test_samples = step_2_load_test_samples_from_csv(QA_CSV_PATH)
    if not ragas_test_samples:
        print("评测集为空，请检查 eval/eval_doc/qa.csv")
        return
    if args.limit > 0:
        ragas_test_samples = ragas_test_samples[: args.limit]
        print(f"--limit 生效：本次仅评估前 {len(ragas_test_samples)} 条")

    # step_3: 运行 RAG pipeline 采集 answer + contexts
    questions = [item["question"] for item in ragas_test_samples]
    pipeline_results = step_3_run_pipeline(questions)

    # 全部空回答时直接退出，避免浪费裁判 LLM 的 token
    if all(not a for a in pipeline_results["answer"]):
        print("Pipeline 未产生任何答案（检查 Milvus/MongoDB/reranker 服务是否在线），终止评估。")
        return

    # step_4: 构建评估数据集
    ragas_dataset = step_4_build_dataset(ragas_test_samples, pipeline_results)

    # step_5: 执行 ragas 评估
    ragas_results = step_5_run_ragas_evaluation(
        dataset=ragas_dataset,
        metrics=ragas_metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # step_6: 导出结果并提取 bad case
    step_6_export_results_and_bad_cases(
        results=ragas_results, metrics=ragas_metrics, bad_case_threshold=args.threshold
    )


if __name__ == "__main__":
    main()
