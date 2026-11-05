"""引用溯源：把「模型回答里的 [1][2]」映射回具体的文献片段。

RAG 最怕的不是答错，而是答得理直气壮却查无出处。这里做三件事：

1. 给检索到的片段编号（[1] 《标题》 章节 p.页码），拼进 prompt 的参考资料；
2. 回答生成后，从回答里解析出所有 [n] 引用；
3. 校验这些编号：哪些有效、哪些越界（模型编造的编号）、有没有一条都没引。

校验结果一方面用于前端展示「依据文献」，另一方面作为「未溯源」告警，
是拒答校准之外的又一道可信度闸门。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 支持 [1] [1,2] [1、2] [1-3] 以及全角 ［1］
_CITATION_RE = re.compile(r"[\[［]\s*(\d{1,3}(?:\s*[,，、\-–]\s*\d{1,3})*)\s*[\]］]")
_SPLIT_RE = re.compile(r"\s*[,，、]\s*")
_RANGE_RE = re.compile(r"(\d{1,3})\s*[\-–]\s*(\d{1,3})")


@dataclass
class SourceRef:
    """一条可引用的来源。"""

    index: int
    document: Document
    title: str = ""
    section: str = ""
    page: str = ""
    source: str = ""
    source_key: str = ""

    @property
    def label(self) -> str:
        parts = [f"[{self.index}] 《{self.title or self.source or '未知文献'}》"]
        if self.section:
            parts.append(self.section)
        if self.page:
            parts.append(f"p.{self.page}")
        return " · ".join(parts)

    @property
    def anchor(self) -> str:
        """稳定的跳转锚点（同一条来源多次命中可去重）。"""
        return f"{self.source_key}|{self.section}|{self.page}"

    def snippet(self, limit: int = 160) -> str:
        text = (self.document.page_content or "").replace("\n", " ").strip()
        return text[:limit] + ("…" if len(text) > limit else "")


@dataclass
class CitationReport:
    """一次回答的引用校验结果。"""

    sources: List[SourceRef] = field(default_factory=list)
    used: List[int] = field(default_factory=list)
    invalid: List[int] = field(default_factory=list)
    grounded: bool = False

    @property
    def coverage(self) -> float:
        """有效引用的条数 / 提供来源条数（上限 1）。"""
        if not self.sources:
            return 0.0
        return min(len(self.used) / len(self.sources), 1.0)

    @property
    def used_sources(self) -> List[SourceRef]:
        used = set(self.used)
        return [source for source in self.sources if source.index in used]

    def summary(self) -> str:
        if not self.sources:
            return "没有可用来源"
        if not self.used:
            return f"未引用任何来源（共提供 {len(self.sources)} 条）"
        text = f"引用 {len(self.used)}/{len(self.sources)} 条来源"
        if self.invalid:
            text += f"，另有 {len(self.invalid)} 个无效编号 {self.invalid}"
        return text


def build_sources(documents: Sequence[Document], dedupe: bool = True) -> List[SourceRef]:
    """把检索片段转成带编号的来源列表（顺序与 prompt 里的编号一致）。"""
    sources: List[SourceRef] = []
    seen: Dict[str, SourceRef] = {}

    for index, document in enumerate(documents, start=1):
        meta = document.metadata or {}
        ref = SourceRef(
            index=index,
            document=document,
            title=str(meta.get("title") or ""),
            section=str(meta.get("section") or ""),
            page=str(meta.get("page") or ""),
            source=str(meta.get("source") or ""),
            source_key=str(meta.get("source_key") or ""),
        )
        if dedupe and ref.anchor in seen:
            # 同一来源同一章节重复命中：保留第一次的编号，避免编号跳号
            continue
        if dedupe:
            seen[ref.anchor] = ref
        sources.append(ref)

    return sources


def renumber(sources: Sequence[SourceRef]) -> List[SourceRef]:
    """去重后重新编号，保证 prompt 里的 [n] 与来源列表完全对应。"""
    for position, source in enumerate(sources, start=1):
        source.index = position
    return list(sources)


def parse_citation_indices(text: str) -> List[int]:
    """从回答文本里解析出所有被引用的编号（去重、保持出现顺序）。"""
    if not text:
        return []

    indices: List[int] = []
    for match in _CITATION_RE.finditer(text):
        body = match.group(1)
        for chunk in _SPLIT_RE.split(body):
            chunk = chunk.strip()
            if not chunk:
                continue
            range_match = _RANGE_RE.match(chunk)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if start <= end and end - start <= 20:
                    indices.extend(range(start, end + 1))
                continue
            if chunk.isdigit():
                indices.append(int(chunk))

    ordered: List[int] = []
    for index in indices:
        if index not in ordered:
            ordered.append(index)
    return ordered


def build_citation_report(answer: str, sources: Sequence[SourceRef]) -> CitationReport:
    """校验回答里的引用：有效 / 越界 / 是否完全未溯源。"""
    valid = {source.index for source in sources}
    cited = parse_citation_indices(answer)
    used = [index for index in cited if index in valid]
    invalid = [index for index in cited if index not in valid]

    if invalid:
        logger.warning("回答引用了不存在的来源编号：%s（实际只有 %d 条）", invalid, len(valid))

    return CitationReport(
        sources=list(sources),
        used=used,
        invalid=invalid,
        grounded=bool(used),
    )


def format_source_list(sources: Iterable[SourceRef], with_snippet: bool = True) -> str:
    """给前端 / CLI 用的来源清单（Markdown）。"""
    lines: List[str] = []
    for source in sources:
        line = f"- {source.label}"
        if with_snippet:
            line += f"\n  > {source.snippet()}"
        lines.append(line)
    return "\n".join(lines) if lines else "（无）"