"""Embedding 模型工厂：模型名 → LangChain `Embeddings`。

- `nomic-embed-text`：本地 Ollama，免费离线（768 维）
- `text-embedding-3-small`：OpenAI（1536 维）

这里对 Ollama 做了一层包装：底层报错时返回带操作指引的异常，
而不是让上层收到一个语焉不详的 HTTP 错误。
"""

import logging
import os
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from rag.config import EMBEDDING_MODELS, OLLAMA_HOST

logger = logging.getLogger(__name__)


class LocalOllamaEmbeddings(OllamaEmbeddings):
    """本地 Ollama 向量化，失败时给出可执行的修复提示。"""

    def _wrap_error(self, exc: Exception) -> RuntimeError:
        return RuntimeError(
            f"调用本地 Ollama 失败（{self.base_url or OLLAMA_HOST}，模型 {self.model}）：{exc}。"
            f"请确认 Ollama 已启动，并执行过 ollama pull {self.model}"
        )

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = super().embed_query(text)
        except Exception as exc:  # noqa: BLE001 - 需要把底层原因一起展示
            raise self._wrap_error(exc) from exc
        if not embedding:
            raise RuntimeError(
                f"Ollama 没有返回 {self.model} 的向量，请确认已执行 ollama pull {self.model}"
            )
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = super().embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            raise self._wrap_error(exc) from exc
        if not embeddings:
            raise RuntimeError(
                f"Ollama 没有返回 {self.model} 的向量，请确认已执行 ollama pull {self.model}"
            )
        return embeddings


def supported_embeddings() -> List[str]:
    """返回前端可选的 embedding 模型列表。"""
    return list(EMBEDDING_MODELS.keys())


def _get_spec(embeddings_model: str) -> dict:
    spec = EMBEDDING_MODELS.get(embeddings_model)
    if spec is None:
        raise ValueError(
            f"不支持的 embedding 模型: {embeddings_model}，可选: {', '.join(EMBEDDING_MODELS)}"
        )
    return spec


def get_embeddings(embeddings_model: str) -> Embeddings:
    """按名称创建 LangChain 的 Embeddings 实例。"""
    spec = _get_spec(embeddings_model)
    provider = spec["provider"]

    if provider == "ollama":
        return LocalOllamaEmbeddings(model=embeddings_model, base_url=OLLAMA_HOST)

    if provider == "openai":
        # 延迟导入：只用本地 Ollama 时不必安装 langchain-openai
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=embeddings_model, api_key=api_key)

    raise ValueError(f"embedding 模型 {embeddings_model} 使用了未知 provider: {provider}")


def get_collection_name(embeddings_model: str) -> str:
    """返回该 embedding 模型对应的向量表名（维度不同，必须分表）。"""
    return _get_spec(embeddings_model)["collection"]
