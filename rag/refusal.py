"""拒答校准：检索置信度不足时，明确说「文献里没有」，而不是让模型硬编。

RAG 系统的幻觉大多出现在「检索结果不相关但模型很自信」的时候。与其相信模型的
自觉，不如在生成之前就用可量化的信号做判断。这里用两个信号：

1. 稠密度相似度（1 - 归一化距离）：向量检索的原始相似度；
2. 覆盖度：问题里的关键词/token 有多少出现在召回片段里。

两者加权成一个 0~1 的置信度，低于阈值就拒答。阈值可以在 .env 里调，
评测脚本可以扫多个阈值，画出「拒答率 vs 误拒率」的权衡曲线。
"""

import difflib
import logging
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from langchain_core.documents import Document

from rag.bm25_index import tokenize
from rag.config import REFUSAL_ENABLED, REFUSAL_MIN_CONFIDENCE

logger = logging.getLogger(__name__)

# nomic-embed-text 这类模型对「无关文本」也会给出 0.6~0.8 的余弦相似度，
# 直接用原始相似度和阈值比较会永远「足够自信」。因此先把相似度按
# [FLOOR, CEIL] 线性映射到 [0, 1]：FLOOR 以下视为完全无关。
SIMILARITY_FLOOR = 0.40
SIMILARITY_CEIL = 0.95

_WORD_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_ALNUM_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")

# 中文问句的「虚词 / 提问框架」bigram：它们只表达「我问的是这篇论文」，
# 不含任何内容信息。若把它们计入覆盖率分母，问英文文献时覆盖率会被
# 平白压低（英文正文里不可能出现「论文/说了/什么」），导致误拒。
_CJK_FILLER = {
    "论文", "文章", "这篇", "那篇", "这个", "那个", "本文", "该文",
    "什么", "文说", "了什", "说什", "是什", "讲什", "有什", "为什",
    "说了", "讲了", "写了", "说了", "介绍", "说明", "告诉", "说说",
    "主要", "内容", "讲了", "如何", "怎么", "怎样", "哪些", "是否",
    "一下", "请问", "关于", "具体", "简单", "大概", "详细", "可以",
    "研究", "工作", "提出", "贡献", "结论", "结果", "实验",
}


@dataclass
class Confidence:
    """一次检索的置信度评估结果。"""

    score: float = 0.0
    similarity: float = 0.0
    coverage: float = 0.0
    top_distance: Optional[float] = None
    documents: int = 0
    reason: str = ""

    def enough(self, threshold: float = REFUSAL_MIN_CONFIDENCE) -> bool:
        return self.score >= threshold

    def describe(self) -> str:
        return (
            f"confidence={self.score:.3f}"
            f"（similarity={self.similarity:.3f} ｜ coverage={self.coverage:.3f}"
            f" ｜ 片段={self.documents}）"
        )


def _similarity_from_distance(distance: Optional[float]) -> float:
    """LanceDB 返回的是平方 L2 距离（向量已 L2 归一化）。

    归一化向量下 |a-b|^2 = 2 - 2cos，所以 cos ∈ [-1,1] → distance ∈ [0,4]。
    这里把距离压到 [0,1] 的相似度，便于和阈值比较。
    """
    if distance is None:
        return 0.0
    similarity = 1.0 - float(distance) / 2.0
    return max(0.0, min(1.0, similarity))


def _normalize_similarity(raw: float) -> float:
    """把「无关也有 0.6+」的原始相似度映射到 0~1 的可用区间。"""
    span = max(SIMILARITY_CEIL - SIMILARITY_FLOOR, 1e-6)
    return max(0.0, min(1.0, (raw - SIMILARITY_FLOOR) / span))


def _is_cjk(token: str) -> bool:
    return bool(_CJK_CHAR_RE.search(token))


def _mostly_non_cjk(text: str) -> bool:
    """正文是不是以非中文为主（英文论文的典型情况）。"""
    cjk = len(_CJK_CHAR_RE.findall(text))
    letters = len(_ALNUM_RE.findall(text))
    if letters == 0:
        return False
    return cjk / letters < 0.15


def _keyword_coverage(query: str, documents: Sequence[Document]) -> float:
    """问题里的「实词」有多少出现在召回片段里。

    三处关键修正：

    1. 中文提问框架（「论文/说了/什么/怎么/哪些」）不含内容，从分母剔除，
       否则问英文文献时这些词在英文正文里一个都不会出现，覆盖率被无端压低。
    2. 跨语言（中文提问 / 英文正文）时，中文「内容词」在英文正文里天然无法
       逐字匹配（「准确率」不会等于 accuracy）。此时把中文内容词整体视为
       「不可校验」，只用英文/数字实词判定覆盖率——否则会把本来能答的
       问题误判成「文献里没有」。若此时已无可校验的实词，退化为全部 token。
    3. 英文实词允许近似匹配（拼错的 LRFVNet / embedin），见 _near_match。

    若剔除后已无实词（如「这篇论文讲了什么」），退化为全部 token 以免分母为 0。
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    haystack = " ".join(
        (document.page_content or "").lower() for document in documents
    )
    cross_lingual = _mostly_non_cjk(haystack) and any(_is_cjk(token) for token in query_tokens)

    content_tokens = set()
    for token in query_tokens:
        if _is_cjk(token):
            if token in _CJK_FILLER:
                continue
            if cross_lingual:
                # 中文内容词无法在英文正文里逐字校验，不计入分母
                continue
        content_tokens.add(token)

    if not content_tokens:
        content_tokens = query_tokens

    haystack_tokens = set(tokenize(haystack))
    # 按首字母分桶：近似匹配只在同首字母的候选里找，
    # 避免每次都对整库几万个 token 跑 difflib（否则大库会明显变慢）。
    buckets: dict = {}
    for candidate in haystack_tokens:
        buckets.setdefault(candidate[:1], set()).add(candidate)

    hits = 0
    for token in content_tokens:
        lowered = token.lower()
        if lowered in haystack:
            hits += 1
        elif _near_match(lowered, buckets.get(lowered[:1], ())):
            hits += 1
    return hits / len(content_tokens)


def _near_match(token: str, candidates) -> bool:
    """容忍拼写错误的近似匹配（只对较长的英文词生效，且限同首字母候选）。"""
    if len(token) < 5 or _CJK_CHAR_RE.search(token) or not _ALNUM_RE.match(token):
        return False
    for candidate in candidates:
        if len(candidate) < 5 or abs(len(candidate) - len(token)) > 2:
            continue
        if _similarity_ratio(token, candidate) >= 0.85:
            return True
    return False


def _similarity_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def assess_confidence(
    query: str,
    documents: Sequence[Document],
    scored: Optional[Sequence[tuple]] = None,
) -> Confidence:
    """结合相似度与关键词覆盖，给出 0~1 的检索置信度。

    `scored` 传入 [(document, similarity)]，其中 similarity 是 0~1 的相似度
    （`RAGKnowledgeBase.search_with_scores` 返回的就是这种形式）。
    """
    if not documents:
        return Confidence(score=0.0, reason="没有检索到任何片段")

    similarity = 0.0
    raw_similarity = 0.0
    top_distance: Optional[float] = None
    if scored:
        # 前三条的平均相似度，比只看第一条更稳
        sims = [max(0.0, min(1.0, float(item[1]))) for item in scored[:3]]
        raw_similarity = sum(sims) / len(sims)
        similarity = _normalize_similarity(raw_similarity)
        top_distance = 2.0 * (1.0 - sims[0])

    coverage = _keyword_coverage(query, documents)

    if scored:
        # 几何平均：任一信号为 0（语义无关 / 关键词完全不命中）就直接判为不可信，
        # 比加权求和更能挡住「相似度虚高但内容无关」的检索结果。
        score = math.sqrt(max(similarity, 0.0) * max(coverage, 0.0))
    else:
        score = coverage

    reason = ""
    if score < REFUSAL_MIN_CONFIDENCE:
        if coverage < 0.15:
            reason = "问题中的关键词在文献片段里几乎不出现"
        elif similarity < 0.3:
            reason = "检索片段与问题语义距离过大"
        else:
            reason = "检索置信度低于阈值"

    return Confidence(
        score=score,
        similarity=similarity,
        coverage=coverage,
        top_distance=top_distance,
        documents=len(documents),
        reason=reason,
    )


def should_refuse(
    confidence: Confidence,
    threshold: float = REFUSAL_MIN_CONFIDENCE,
    enabled: Optional[bool] = None,
) -> bool:
    """是否拒答。

    `enabled` 显式传入时优先于全局开关 —— 前端「高级设置」允许在运行时开关拒答，
    不用改 .env 重启。
    """
    if enabled is None:
        enabled = REFUSAL_ENABLED
    if not enabled:
        return False
    return not confidence.enough(threshold)


def refusal_message(confidence: Confidence, threshold: float = REFUSAL_MIN_CONFIDENCE) -> str:
    """统一的拒答话术：说明原因 + 给出可操作的下一步。"""
    detail = f"（检索置信度 {confidence.score:.2f}，低于阈值 {threshold:.2f}）"
    cause = confidence.reason or "相关度不足"
    return (
        f"提供的文献中没有足够的信息回答这个问题{detail}。\n\n"
        f"原因：{cause}。\n\n"
        "建议：\n"
        "1. 换一种问法，或补充更具体的关键词（论文标题、方法名、术语）；\n"
        "2. 确认相关论文已经上传并完成入库；\n"
        "3. 如果确实是文献中没有的内容，请降低阈值（`RAG_REFUSAL_MIN_CONFIDENCE`）后再试。"
    )