"""Rerank：对融合后的候选片段做精排。

为什么 rerank 有效：召回阶段（向量 / BM25）为了不漏，宁可多给候选（top-20~50），
但真正送进 prompt 的只有 2~3 条。用交叉编码器（cross-encoder）把「问题 + 片段」
一起编码，能捕捉两者的细粒度交互，判别力远高于双塔 embedding 的点积。

两种后端：

- cross-encoder：本地 sentence-transformers 模型，效果最好，但要装 torch；
- llm：直接让大模型给每个片段打 0~10 分，零新增依赖（复用已有的 API Key），
  速度慢、有成本，但效果通常也明显好于不重排。

失败一律降级为「保持原顺序」，重排是增强，不能成为新的故障点。
"""

import logging
import re
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_LLM_SCORE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是检索结果相关性评分员。用户会给你一个问题，以及若干编号的文献片段。\n"
            "请为每个片段打 0~10 的整数分：10 = 直接回答问题；7-9 = 高度相关；"
            "4-6 = 部分相关；1-3 = 弱相关；0 = 完全无关。\n"
            "输出格式：每行「编号: 分数」，例如：\n"
            "1: 8\n2: 3\n"
            "不要输出任何解释。",
        ),
        ("human", "问题：{question}\n\n{chunks}\n\n评分："),
    ]
)


class CrossEncoderReranker:
    """本地 cross-encoder 重排（sentence-transformers）。"""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "本地 rerank 需要 sentence-transformers：pip install sentence-transformers"
                ) from exc
            logger.info("加载 rerank 模型 %s …", self.model_name)
            self._model = CrossEncoder(
                self.model_name, cache_folder=self.cache_dir, max_length=512
            )
        return self._model

    @property
    def available(self) -> bool:
        # 只用 find_spec 探测是否安装，避免在会话初始化阶段真正 import
        # sentence-transformers / torch（首次 import 往往要几秒，会拖慢登录后的加载）。
        # 真正的依赖/模型问题留到 rerank() 第一次调用时再降级。
        try:
            import importlib.util

            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:  # noqa: BLE001
            return False
    @property
    def cached(self) -> bool:
        """模型是否已在本地缓存（决定 auto 模式要不要用它，避免联网卡死）。"""
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:  # noqa: BLE001
            return False
        try:
            result = try_to_load_from_cache(self.model_name, "config.json")
        except Exception:  # noqa: BLE001
            return False
        return isinstance(result, str)

    def rerank(
        self,
        question: str,
        documents: Sequence[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        if not documents:
            return []
        try:
            model = self._load()
            pairs = [(question, document.page_content or "") for document in documents]
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cross-encoder 重排失败，保持原顺序：%s", exc)
            return list(documents)[: top_n or len(documents)]

        ranked = sorted(
            zip(documents, scores), key=lambda item: float(item[1]), reverse=True
        )
        result = [document for document, _ in ranked]
        return result[: top_n or len(result)]


# 单次请求里塞多少个片段一起打分（太多会超上下文，太少调用次数多）
_LLM_RERANK_BATCH = 8


class LLMReranker:
    """用大模型给候选片段打分重排（零新增依赖）。

    为了控制成本与延迟，候选片段按批打包进一次请求（默认 8 个一批），
    让模型一次输出所有编号的分数。
    """

    def __init__(self, llm: BaseChatModel, batch_size: int = _LLM_RERANK_BATCH) -> None:
        self.llm = llm
        self.batch_size = max(1, batch_size)

    @staticmethod
    def _format_chunks(documents: Sequence[Document]) -> str:
        blocks: List[str] = []
        for index, document in enumerate(documents, start=1):
            meta = document.metadata or {}
            header = f"[{index}]"
            if meta.get("section"):
                header += f" {meta['section']}"
            if meta.get("page"):
                header += f" p.{meta['page']}"
            blocks.append(f"{header}\n{(document.page_content or '')[:1500]}")
        return "\n\n".join(blocks)

    @staticmethod
    def _parse_scores(raw: str, count: int) -> List[float]:
        """解析「编号: 分数」格式；缺失的位置记 0 分。"""
        scores = [0.0] * count
        for line in (raw or "").splitlines():
            match = re.match(r"\s*\[?(\d+)\]?\s*[.):：\-]\s*(\d+(?:\.\d+)?)", line)
            if not match:
                continue
            index = int(match.group(1)) - 1
            if 0 <= index < count:
                scores[index] = float(match.group(2))
        return scores

    def _score_batch(self, question: str, documents: Sequence[Document]) -> List[float]:
        try:
            chain = _LLM_SCORE_PROMPT | self.llm | StrOutputParser()
            raw = chain.invoke(
                {"question": question, "chunks": self._format_chunks(documents)}
            )
        except Exception as exc:  # noqa: BLE001 - 打分失败就保持原顺序
            logger.warning("LLM 批量打分失败，该批保持原顺序：%s", exc)
            return [-float(index) for index in range(len(documents))]

        scores = self._parse_scores(raw, len(documents))
        if not any(scores):
            logger.debug("LLM 未返回可解析的分数，该批保持原顺序：%r", raw[:120])
            return [-float(index) for index in range(len(documents))]
        return scores

    def rerank(
        self,
        question: str,
        documents: Sequence[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        if not documents:
            return []

        items = list(documents)
        scored: List[Tuple[Document, float]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            scores = self._score_batch(question, batch)
            scored.extend(zip(batch, scores))

        # 同分时保留原有名次：sort 是稳定的，且 order 字段已在分值里体现
        scored.sort(key=lambda item: item[1], reverse=True)
        return [document for document, _ in scored][: top_n or len(scored)]


def build_reranker(
    mode: str,
    llm: Optional[BaseChatModel] = None,
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """按配置创建 reranker；不可用时返回 None（调用方保持原顺序）。"""
    mode = (mode or "off").strip().lower()

    if mode == "off":
        return None

    cross = CrossEncoderReranker(model_name or "BAAI/bge-reranker-base", cache_dir)

    if mode == "cross-encoder":
        if not cross.available:
            logger.warning(
                "RAG_RERANK_MODE=cross-encoder 但未安装 sentence-transformers，"
                "已降级为不重排；如需启用请 pip install sentence-transformers"
            )
            return None
        return cross

    if mode == "llm":
        return LLMReranker(llm) if llm is not None else None

    # auto：只有模型已经下载到本地时才用 cross-encoder，避免首次联网拉模型卡住页面
    if cross.available and cross.cached:
        logger.info("rerank：使用本地 cross-encoder（%s）", cross.model_name)
        return cross
    if llm is not None:
        if cross.available:
            logger.info(
                "rerank：cross-encoder 模型未下载到本地（%s），本次使用 LLM 重排；"
                "想用本地模型请先预下载，或设 RAG_RERANK_MODE=cross-encoder",
                cross.model_name,
            )
        return LLMReranker(llm)

    return None