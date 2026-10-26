# eval/embeddings_adapter.py
"""
把项目本地的 BGE-M3（utils/embedding_utils.py）适配成 LangChain Embeddings 接口，
供 ragas 的 LangchainEmbeddingsWrapper 包装后计算 answer_relevancy / answer_correctness。

说明（对应课件 6.3.1）：
- generate_embeddings() 同时产出 dense + sparse，但 Ragas 评测阶段不做检索，
  只需要语义相似度计算，因此这里只取 dense，不会导致评测失真。
- BGE-M3 是本地 torch 模型，首次调用会加载（较慢），之后走单例缓存。
"""

from typing import List

from langchain_core.embeddings import Embeddings

from utils.embedding_utils import generate_embeddings


class ProjectBGEEmbeddings(Embeddings):
    """Ragas ← LangChain ← 项目 BGE-M3 的桥接类"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return generate_embeddings(texts)["dense"]

    def embed_query(self, text: str) -> List[float]:
        dense = self.embed_documents([text])
        return dense[0] if dense else []
