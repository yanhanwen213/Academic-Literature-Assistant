"""文献知识库：RAG 层的对外门面。

完整检索链路：

    问题 →（可选）Query 改写 / 多查询展开
         → 向量检索（稠密）+ BM25 检索（稀疏）
         → RRF 融合
         →（可选）Rerank 精排
         → top-N 片段送进 prompt

每一项都能在 .env 里单独关掉（RAG_RETRIEVAL_MODE / RAG_QUERY_REWRITE /
RAG_RERANK_MODE），方便做基线对比 —— 评测脚本可以直接跑「关掉 vs 打开」两组数据。
"""

import logging
import os
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.bm25_index import BM25Index
from rag.config import (
    DEFAULT_NUM_DOCUMENTS,
    EMBEDDING_MODELS,
    OLLAMA_HOST,
    RECALL_PER_ROUTE,
    RERANK_CACHE_DIR,
    RERANK_MODEL,
    RERANK_MODE,
    RETRIEVAL_MODE,
    RRF_K,
    ollama_reachable,
)
from rag.embedder import get_embeddings
from rag.loaders import IngestResult, load_arxiv, load_pdf_bytes
from rag.rerank import build_reranker
from rag.rrf import rrf_fuse
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def _similarity(distance: Optional[float]) -> float:
    """把归一化向量下的平方 L2 距离换算成 0~1 相似度。"""
    if distance is None:
        return 0.0
    value = 1.0 - float(distance) / 2.0
    return max(0.0, min(1.0, value))


class RAGKnowledgeBase:
    """某个 embedding 模型对应的文献知识库。"""

    def __init__(
        self,
        embeddings_model: str,
        num_documents: int = DEFAULT_NUM_DOCUMENTS,
        llm=None,
        retrieval_mode: Optional[str] = None,
        rerank_mode: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> None:
        self.embeddings_model = embeddings_model
        self.num_documents = num_documents
        self.retrieval_mode = (retrieval_mode or RETRIEVAL_MODE).strip().lower()
        # 这两项可以在运行时改（前端「高级设置」），用于对比召回宽度与 RRF 融合强度
        self.recall_per_route = RECALL_PER_ROUTE
        self.rrf_k = RRF_K
        self.embeddings = get_embeddings(embeddings_model)
        self.uri = uri
        self.vector_store = get_vector_store(
            embeddings_model, self.embeddings, uri=uri
        )
        self.bm25 = BM25Index()
        self._bm25_loaded = False
        self.reranker = self._build_reranker(rerank_mode or RERANK_MODE, llm)

    # ---------------------------------------------------------------- 运行时配置
    def _build_reranker(self, mode: str, llm=None):
        """构建 reranker；任何异常都降级为「不重排」，不能拖垮会话初始化。"""
        try:
            return build_reranker(
                mode,
                llm=llm,
                model_name=RERANK_MODEL,
                cache_dir=RERANK_CACHE_DIR,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("重排初始化失败，已降级为不重排：%s", exc)
            return None

    def set_retrieval_mode(self, mode: str) -> None:
        self.retrieval_mode = (mode or "hybrid").strip().lower()
        self._bm25_loaded = False

    def set_reranker(self, mode: str, llm=None) -> None:
        self.reranker = self._build_reranker(mode, llm)

    @property
    def _dense_enabled(self) -> bool:
        return self.retrieval_mode not in {"sparse", "bm25", "keyword"}

    @property
    def _sparse_enabled(self) -> bool:
        return self.retrieval_mode in {"hybrid", "sparse", "bm25", "keyword"}
    # ---------------------------------------------------------------- 入库
    def ensure_embedder_ready(self) -> None:
        """入库前确认向量化所需的服务可用，给出比底层报错更清楚的提示。"""
        spec = EMBEDDING_MODELS.get(self.embeddings_model, {})
        if spec.get("provider") == "ollama" and not ollama_reachable():
            raise RuntimeError(
                f"本地 Ollama 未启动（{OLLAMA_HOST}），无法向量化文献。"
                "请先运行 ollama serve，并执行过 ollama pull nomic-embed-text"
            )
        if spec.get("provider") == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "当前 embedding 使用 OpenAI（text-embedding-3-small），"
                "但 .env 里没有 OPENAI_API_KEY"
            )

    def _sync_bm25(self, documents: Sequence[Document]) -> None:
        """把新入库的片段同步进 BM25 索引（混合检索时才有意义）。"""
        if not self._sparse_enabled:
            return
        try:
            if not self._bm25_loaded:
                # 首次入库：用整表重建，覆盖「历史进程已入库」的情况
                self.bm25.rebuild(self.vector_store.all_documents())
                self._bm25_loaded = True
            else:
                self.bm25.add(documents)
        except Exception as exc:  # noqa: BLE001 - 稀疏索引故障不能让入库失败
            logger.warning("BM25 索引更新失败，检索将退化为纯向量：%s", exc)
            self._bm25_loaded = False

    def add_documents(self, documents: Optional[List[Document]]) -> int:
        """按 id 覆盖写入向量库，返回入库的片段数。"""
        if not documents:
            return 0
        self.ensure_embedder_ready()
        self.vector_store.upsert_documents(documents)
        self._sync_bm25(documents)
        return len(documents)

    def remove_documents(self, ids: Sequence[str]) -> int:
        """按片段 id 删除；删完重建 BM25，避免稀疏检索命中已删片段。"""
        ids = [str(item) for item in (ids or []) if item]
        if not ids:
            return 0
        self.vector_store.delete(ids=ids)
        try:
            self.bm25.rebuild(self.vector_store.all_documents())
            self._bm25_loaded = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除后重建 BM25 索引失败：%s", exc)
            self._bm25_loaded = False
        return len(ids)

    def add_pdf_bytes(self, data: bytes, filename: str = "", **chunk_options) -> IngestResult:
        """解析 PDF（版面感知 + 章节切分）并入库。

        `chunk_options` 可覆盖分块参数（chunk_size / overlap / min_chars /
        contextual_header / dedupe），前端「高级设置」会传进来。
        """
        return self._store(load_pdf_bytes(data, filename=filename, **chunk_options))

    def add_arxiv(self, arxiv_id: str, **chunk_options) -> IngestResult:
        """按 arXiv ID 下载论文、解析并入库。"""
        return self._store(load_arxiv(arxiv_id, **chunk_options))

    def _store(self, result: IngestResult) -> IngestResult:
        """把装载结果写进向量库；没有解析出文本时不建表也不报错。"""
        if result.documents:
            self.ensure_embedder_ready()
            self.vector_store.upsert_documents(result.documents)
            self._sync_bm25(result.documents)
        return result

    # ---------------------------------------------------------------- 检索
    def _dense_search(self, query: str, k: int) -> List[Document]:
        return [document for document, _ in self._dense_search_with_score(query, k)]

    def _dense_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """向量检索并保留距离，供拒答校准与「相关性」展示使用。"""
        if not self._dense_enabled:
            return []
        try:
            return list(self.vector_store.similarity_search_with_score(query, k=k))
        except Exception as exc:  # noqa: BLE001
            logger.error("向量检索失败：%s", exc)
            return []

    def _sparse_search(self, query: str, k: int) -> List[Document]:
        if not self._sparse_enabled:
            return []
        try:
            if not self._bm25_loaded:
                self.bm25.rebuild(self.vector_store.all_documents())
                self._bm25_loaded = True
            hits = self.bm25.search(query, k=k)
            return self.bm25.documents_by_ids([doc_id for doc_id, _ in hits])
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 检索失败，退化为纯向量：%s", exc)
            return []

    def search(
        self,
        query: str,
        num_documents: Optional[int] = None,
        queries: Optional[Sequence[str]] = None,
    ) -> List[Document]:
        """检索与问题最相关的片段（混合检索 + RRF + 可选 rerank）。"""
        return [
            document
            for document, _ in self.search_with_scores(
                query, num_documents=num_documents, queries=queries
            )
        ]

    def search_with_scores(
        self,
        query: str,
        num_documents: Optional[int] = None,
        queries: Optional[Sequence[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """检索并带回相关性分数（0~1，越大越相关），供拒答校准使用。

        `queries` 传入多个改写后的问法时，每个问法都会走完整检索并一起融合。
        """
        top_n = num_documents or self.num_documents
        query_list = [query] if not queries else list(queries)
        recall_k = max(self.recall_per_route, top_n)  # 选择得到结果的数目

        ranked_lists: List[List[Document]] = []
        top_distances: List[float] = []
        for item in query_list:
            dense_scored = self._dense_search_with_score(item, recall_k)
            if dense_scored:
                top_distances.append(float(dense_scored[0][1]))
            dense = [document for document, _ in dense_scored]
            sparse = self._sparse_search(item, recall_k)
            if dense:
                ranked_lists.append(dense)
            if sparse:
                ranked_lists.append(sparse)

        if not ranked_lists:
            return []

        if len(ranked_lists) == 1:
            fused = ranked_lists[0]
        else:
            fused = rrf_fuse(ranked_lists, k=self.rrf_k)

        if self.reranker is not None:
            fused = self.reranker.rerank(query, fused, top_n=top_n)
        else:
            fused = fused[:top_n]

        best_distance = min(top_distances) if top_distances else None
        return [(document, _similarity(best_distance)) for document in fused]

    def retriever(self, num_documents: Optional[int] = None) -> BaseRetriever:
        """返回 LangChain 标准 retriever（内部就是上面的 search）。"""
        top_n = num_documents or self.num_documents

        class _KnowledgeRetriever(BaseRetriever):
            knowledge: RAGKnowledgeBase

            def _get_relevant_documents(self, query: str, **_kwargs) -> List[Document]:
                return self.knowledge.search(query, num_documents=top_n)

        return _KnowledgeRetriever(knowledge=self)

    def count(self) -> int:
        """向量库中当前的片段总数（表不存在时返回 0）。"""
        try:
            return int(self.vector_store.count())
        except Exception as exc:  # noqa: BLE001
            logger.debug("读取知识库片段数失败: %s", exc)
            return 0

    def clear(self) -> None:
        """清空向量表与稀疏索引（表不存在时视为已清空）。"""
        self.vector_store.clear()
        self.bm25.rebuild([])
        self._bm25_loaded = True