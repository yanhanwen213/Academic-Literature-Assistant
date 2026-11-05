"""RRF（Reciprocal Rank Fusion）融合多路检索结果。

公式：score(d) = sum_i  weight_i / (k + rank_i(d))

只用名次、不用原始分数，因此向量距离（越小越好）和 BM25 分数（越大越好）
可以直接混在一起，不需要归一化，也没有需要调的权重。k 通常取 60。

顺带提供一个 RRF 风格的重排融合：把 rerank 分数也当成一路「排序」，这样
rerank 结果与多查询召回可以统一处理。
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RRF_K = 60


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = RRF_K,
    weights: Optional[Sequence[float]] = None,
    top_n: Optional[int] = None,
) -> List[Document]:
    """把多路有序结果融合成一路有序结果。

    ranked_lists：每一路都已按相关度降序排好的 Document 列表；
    weights：每一路的权重（默认等权）；
    top_n：只返回前 N 条。
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}
    store: Dict[str, Document] = {}

    for route_index, documents in enumerate(ranked_lists):
        weight = weights[route_index] if route_index < len(weights) else 1.0
        for rank, document in enumerate(documents, start=1):
            key = document.id or f"fallback::{document.page_content[:120]}"
            store.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_n is not None:
        ordered = ordered[:top_n]

    fused = [store[key] for key, _ in ordered]
    logger.debug(
        "RRF 融合：%d 路 → %d 条（去重后 %d 条）",
        len(ranked_lists),
        sum(len(items) for items in ranked_lists),
        len(fused),
    )
    return fused


def rrf_fuse_scored(
    scored_lists: Sequence[Sequence[Tuple[Document, float]]],
    k: int = RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[Document, float]]:
    """同上，但输入是 [(Document, score)] 形式，返回带 RRF 分的列表。"""
    if not scored_lists:
        return []

    if weights is None:
        weights = [1.0] * len(scored_lists)

    scores: Dict[str, float] = {}
    store: Dict[str, Document] = {}

    for route_index, items in enumerate(scored_lists):
        weight = weights[route_index] if route_index < len(weights) else 1.0
        for rank, (document, _raw_score) in enumerate(items, start=1):
            key = document.id or f"fallback::{document.page_content[:120]}"
            store.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(store[key], score) for key, score in ordered]