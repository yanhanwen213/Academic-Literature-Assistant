"""BM25 稀疏检索的索引实现（混合检索的其中一路）。

为什么需要它：向量检索擅长语义相近，但对专有名词、缩写、公式名、作者名这类
精确匹配经常失效（问 BM25 和 RRF 的区别时，embedding 容易把 RRF 和 RAG 混起来）。
BM25 擅长精确匹配但不理解同义改写，两路互补，用 RRF 融合即可。

只依赖标准库 + rank_bm25（纯 Python，无需编译）。索引是进程内存对象，
从向量库现有内容重建，不额外落盘。
"""

import logging
import re
from typing import Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]*|\d+(?:\.\d+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "be", "been", "by", "with", "as", "at", "from", "that", "this",
    "these", "those", "it", "its", "we", "our", "they", "their", "what", "how",
    "why", "which", "who", "when", "does", "do", "did", "can", "could",
    # 中文高频虚词（单字 bigram 里噪声很大）
    "的是", "了", "在", "和", "与", "及", "或", "也", "都", "就", "而", "被",
    "把", "对", "从", "到", "为", "以", "这", "那", "其", "有", "无", "不",
}


def tokenize(text: str) -> List[str]:
    """中英混排分词：英文按词，中文按字符 bigram。"""
    if not text:
        return []

    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    tokens = [token.strip(".-") for token in tokens if token.strip(".-")]

    if _CJK_RE.search(text):
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(run) == 1:
                tokens.append(run)
                continue
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))

    return [token for token in tokens if token and token not in _STOPWORDS]

class BM25Index:
    """内存 BM25 索引：支持整表重建与追加。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._documents: List[Document] = []
        self._ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def ready(self) -> bool:
        return self._bm25 is not None and bool(self._documents)

    def rebuild(self, documents: Sequence[Document]) -> None:
        """用给定语料重建索引（入库 / 清空后调用一次）。"""
        self._documents = list(documents)
        self._ids = [
            str(document.id or index) for index, document in enumerate(self._documents)
        ]
        self._id_to_index = {
            doc_id: index for index, doc_id in enumerate(self._ids)
        }

        if not self._documents:
            self._bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "混合检索需要 rank_bm25 依赖：pip install rank_bm25"
            ) from exc

        corpus = [tokenize(document.page_content) for document in self._documents]
        if not any(corpus):
            logger.warning("BM25 语料分词后为空，稀疏检索将不返回结果")
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        logger.info("BM25 索引就绪：%d 个片段", len(self._documents))
    def add(self, documents: Sequence[Document]) -> None:
        """追加片段；同 id 的旧内容先剔除（覆盖语义）。"""
        if not documents:
            return
        if self._bm25 is None:
            self.rebuild(list(self._documents) + list(documents))
            return

        incoming_ids = {str(document.id) for document in documents if document.id}
        merged = [
            document for document in self._documents if str(document.id) not in incoming_ids
        ]
        merged.extend(documents)
        # rank_bm25 不支持增量，直接重建（几千片段级别是毫秒级）
        self.rebuild(merged)

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """返回 [(doc_id, bm25 分数)]，按分数降序，只保留正分。"""
        if self._bm25 is None or not self._documents:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            (
                (self._ids[index], float(score))
                for index, score in enumerate(scores)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(doc_id, score) for doc_id, score in ranked[:k] if score > 0]

    def documents_by_ids(self, ids: Iterable[str]) -> List[Document]:
        """按 id 取回文档（保持传入顺序）。"""
        return [
            self._documents[self._id_to_index[doc_id]]
            for doc_id in ids
            if doc_id in self._id_to_index
        ]