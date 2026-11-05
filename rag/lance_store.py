"""LanceDB 向量库：实现 LangChain 的 `VectorStore` 抽象接口。

为什么手写而不直接用现成的 LangChain 封装：现成封装要么每次初始化都重建表
（等于清空已有文献），要么重复写入时堆重复行、清空也不生效。这里直接对接
lancedb，行为完全可控：

- 表只在第一次写入时创建，之后只追加/覆盖；
- 每条片段带稳定 id，重复导入同一篇文献是覆盖而不是新增；
- 向量在入库和检索时都做归一化，使 LanceDB 默认的 L2 距离等价于余弦距离；
- 实现 `similarity_search` 之后，可以直接 `.as_retriever()` 交给 LangChain 用。
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

_ID = "id"
_VECTOR = "vector"
_PAYLOAD = "payload"
_DELETE_BATCH = 200


def _normalize(vector: List[float]) -> List[float]:
    """L2 归一化，让 L2 距离与余弦距离同序。"""
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return list(vector)
    return [value / norm for value in vector]


def _quote(ids: Iterable[str]) -> str:
    return ", ".join("'" + str(item).replace("'", "''") + "'" for item in ids)


class LanceVectorStore(VectorStore):
    """本地文件型向量库（零服务依赖）。"""

    def __init__(self, uri: str, table_name: str, embedding: Embeddings) -> None:
        self.uri = str(uri)
        self.table_name = table_name
        self.embedding = embedding
        self._db: Any = None
        self._table: Any = None

    # ---------------------------------------------------------- 连接与建表
    def _connection(self) -> Any:
        if self._db is None:
            try:
                import lancedb
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("缺少 lancedb 依赖：pip install lancedb pyarrow") from exc
            Path(self.uri).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self.uri)
        return self._db

    @staticmethod
    def _table_names(db: Any) -> List[str]:
        try:
            response = db.list_tables()
            tables = getattr(response, "tables", None)
            if tables is not None:
                return list(tables)
        except Exception:  # noqa: BLE001 - 老版本 lancedb 没有 list_tables
            pass
        return list(db.table_names())

    def _open_table(self) -> Optional[Any]:
        """返回表句柄；每次都与磁盘上的表列表核对，避免拿到已被删除的旧表。"""
        db = self._connection()
        if self.table_name not in self._table_names(db):
            self._table = None
            return None
        if self._table is None:
            self._table = db.open_table(self.table_name)
        return self._table

    def _ensure_table(self, dimensions: int) -> Any:
        table = self._open_table()
        if table is not None:
            return table

        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field(_ID, pa.string()),
                pa.field(_VECTOR, pa.list_(pa.float32(), dimensions)),
                pa.field(_PAYLOAD, pa.string()),
            ]
        )
        self._table = self._connection().create_table(
            self.table_name, schema=schema, mode="create"
        )
        logger.info("创建向量表 %s（%d 维，%s）", self.table_name, dimensions, self.uri)
        return self._table

    # ---------------------------------------------------------- 行转换
    @staticmethod
    def _row(document: Document, vector: List[float], fallback_id: str) -> Dict[str, Any]:
        # 兜底清洗：\x00 这类控制字符会让 Arrow / JSON 写入失败
        content = (document.page_content or "").replace("\x00", " ")
        payload = {
            "content": content,
            "metadata": document.metadata or {},
        }
        return {
            _ID: str(document.id or fallback_id),
            _VECTOR: _normalize([float(value) for value in vector]),
            _PAYLOAD: json.dumps(payload, ensure_ascii=False),
        }

    def _build_rows(self, documents: List[Document]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        vectors = self.embedding.embed_documents([doc.page_content for doc in documents])
        rows: List[Dict[str, Any]] = []
        for index, (document, vector) in enumerate(zip(documents, vectors)):
            fallback_id = f"{self.table_name}-{index}-{abs(hash(document.page_content))}"
            rows.append(self._row(document, vector, fallback_id))
        return rows

    # ---------------------------------------------------------- 写入
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "请先构造 LanceVectorStore(uri, table_name, embedding)，再调用 add_documents()"
        )

    def add_texts(self, texts, metadatas=None, *, ids=None, **kwargs) -> List[str]:  # type: ignore[override]
        documents = []
        for index, text in enumerate(texts):
            metadata = metadatas[index] if metadatas else {}
            doc_id = ids[index] if ids else None
            documents.append(Document(page_content=text, metadata=metadata, id=doc_id))
        return self.add_documents(documents)

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:  # type: ignore[override]
        rows = self._build_rows(documents)
        if not rows:
            return []
        table = self._ensure_table(len(rows[0][_VECTOR]))
        table.add(rows)
        logger.debug("写入 %d 个片段到 %s", len(rows), self.table_name)
        return [row[_ID] for row in rows]

    def upsert_documents(self, documents: List[Document]) -> List[str]:
        """按 id 覆盖写入（同一篇文献重新解析后不会残留旧片段）。"""
        rows = self._build_rows(documents)
        if not rows:
            return []

        table = self._ensure_table(len(rows[0][_VECTOR]))
        ids = [row[_ID] for row in rows]
        for start in range(0, len(ids), _DELETE_BATCH):
            batch = ids[start : start + _DELETE_BATCH]
            table.delete(f"{_ID} IN ({_quote(batch)})")
        table.add(rows)
        logger.debug("覆盖写入 %d 个片段到 %s", len(rows), self.table_name)
        return ids

    # ---------------------------------------------------------- 检索
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:  # type: ignore[override]
        return [document for document, _ in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs
    ) -> List[Tuple[Document, float]]:
        table = self._open_table()
        if table is None:
            return []

        vector = self.embedding.embed_query(query)
        if not vector:
            logger.error("检索没有拿到查询向量：%s", query)
            return []

        try:
            rows = (
                table.search(_normalize([float(value) for value in vector]), vector_column_name=_VECTOR)
                .limit(k)
                .to_list()
            )
        except Exception as exc:  # noqa: BLE001 - 表结构异常时不要让问答整体挂掉
            logger.error("向量检索失败：%s", exc)
            return []

        results: List[Tuple[Document, float]] = []
        for item in rows:
            try:
                payload = json.loads(item.get(_PAYLOAD) or "{}")
            except (TypeError, ValueError):
                payload = {}
            # 兼容旧版写入的 payload 字段名（meta_data → metadata）
            metadata = payload.get("metadata") or payload.get("meta_data") or {}
            document = Document(
                page_content=payload.get("content") or "",
                metadata=metadata,
                id=str(item.get(_ID) or ""),
            )
            results.append((document, float(item.get("_distance") or 0.0)))
        return results

    # ---------------------------------------------------------- 管理
    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> Optional[bool]:  # type: ignore[override]
        if ids is None:
            return self.clear()
        table = self._open_table()
        if table is None:
            return True
        for start in range(0, len(ids), _DELETE_BATCH):
            table.delete(f"{_ID} IN ({_quote(ids[start : start + _DELETE_BATCH])})")
        return True

    def exists(self) -> bool:
        return self._open_table() is not None

    def count(self) -> int:
        table = self._open_table()
        return int(table.count_rows()) if table is not None else 0

    def all_documents(self, limit: Optional[int] = None) -> List[Document]:
        """取出全部片段（用于重建 BM25 稀疏索引）。"""
        table = self._open_table()
        if table is None:
            return []

        try:
            rows = table.to_arrow().to_pylist()
        except Exception as exc:  # noqa: BLE001
            logger.error("读取向量库全量片段失败：%s", exc)
            return []
        documents: List[Document] = []
        for item in rows:
            if limit is not None and len(documents) >= limit:
                break
            try:
                payload = json.loads(item.get(_PAYLOAD) or "{}")
            except (TypeError, ValueError):
                payload = {}
            documents.append(
                Document(
                    page_content=payload.get("content") or "",
                    metadata=payload.get("metadata") or payload.get("meta_data") or {},
                    id=str(item.get(_ID) or ""),
                )
            )
        return documents

    def clear(self) -> bool:
        """删除整张表（下次写入会重建），等价于「清空知识库」。"""
        if not self.exists():
            return True
        self._connection().drop_table(self.table_name)
        self._table = None
        logger.info("删除向量表 %s", self.table_name)
        return True
