"""向量库工厂：按 embedding 模型创建 LanceDB 向量库。

LanceDB 是嵌入式向量库，数据落在 `data/lancedb` 目录，不需要任何数据库服务，
也不需要 Docker。如果要换成 Postgres/pgvector 或其他后端，只需在这里再加一个
分支，返回一个实现了 `langchain_core.vectorstores.VectorStore` 的对象即可，
上层（knowledge_base / 对话链 / 评测）完全不用改。
"""

from pathlib import Path
from typing import Any, Optional

from langchain_core.embeddings import Embeddings

from rag.config import LANCEDB_DIR
from rag.embedder import get_collection_name


def vector_store_available(uri: Optional[str] = None) -> bool:
    """向量库是否就绪。

    LanceDB 是本地文件库，只要目录可写就可用；这里做一次轻量检查，
    保证错误在「入库」时以清楚的文案出现，而不是在页面初始化时炸掉。
    `uri` 传入时检查该知识库自己的目录。
    """
    target = Path(uri or LANCEDB_DIR)
    try:
        target.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def vector_store_hint(uri: Optional[str] = None) -> str:
    """给使用者的修复建议。"""
    target = uri or LANCEDB_DIR
    return f"本地向量库目录 {target} 不可写，请检查目录权限或磁盘空间。"


def get_vector_store(
    embeddings_model: str, embedding: Embeddings, uri: Optional[str] = None
) -> Any:
    """返回 (embedding 模型, 向量表) 一一对应的 LanceDB 向量库实例。

    `uri` 是知识库目录：不同知识库指向不同目录，互不影响。
    """
    from rag.lance_store import LanceVectorStore

    return LanceVectorStore(
        uri=uri or LANCEDB_DIR,
        table_name=get_collection_name(embeddings_model),
        embedding=embedding,
    )
