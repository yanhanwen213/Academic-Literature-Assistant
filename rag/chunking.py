"""章节级切分：把 ParsedPaper 变成带完整元数据的 LangChain Document。

切分原则：

1. 不跨章节切 —— 每个片段只属于一个章节（Abstract / 3 Method …）；
2. 先按段落、再按句子，句内不再切坏（无标点的超长表格才硬切）；
3. 每个片段都带上「标题 / 作者 / 年份 / 章节 / 页码」，回答时才能给出处；
4. 入库文本默认加一行「《论文标题》 章节名」上下文，显著提升检索命中率，
   由 `RAG_CONTEXTUAL_HEADER` 控制。
"""

import hashlib
import logging
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document

from rag.cleaning import clean_section_text
from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXTUAL_HEADER,
    DEDUPE_CHUNKS,
    MIN_CHUNK_CHARS,
)
from rag.parsers import ParsedPaper, is_reference_section

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s+")
_SLUG_RE = re.compile(r"[^\w\u4e00-\u9fff]+")

logger = logging.getLogger(__name__)


def _slug(text: str, limit: int = 40) -> str:
    slug = _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    return slug[:limit] or "section"


def _hard_split(text: str, size: int) -> List[str]:
    """没有标点的超长文本（表格、代码块）按长度硬切。"""
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
    seen_paragraphs: Optional[set] = None,
) -> List[str]:
    """把一段章节正文切成若干个带重叠的片段。

    顺序：清洗 → 段落 → 句子 → 硬切 → 合并短片段 → 加重叠 → 丢弃过短片段。

    `seen_paragraphs` 传入一个集合时，全篇范围内重复的段落（页眉残留、重复的
    图表说明）会被跳过；同一篇文献的所有章节应共用同一个集合。
    """
    text = clean_section_text(text)
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]

    # 段落级去重：跨章节共享 seen_paragraphs，重复段落只保留第一次出现。
    # 必须放在「长度不超过一块就直接返回」的快捷分支之前，否则短章节里的
    # 重复段落会被整段带走。
    if seen_paragraphs is not None and paragraphs:
        unique: List[str] = []
        for paragraph in paragraphs:
            fingerprint = hashlib.md5(paragraph.encode("utf-8")).hexdigest()
            if fingerprint in seen_paragraphs:
                continue
            seen_paragraphs.add(fingerprint)
            unique.append(paragraph)
        paragraphs = unique
        if not paragraphs:
            return []
        text = "\n".join(paragraphs)

    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chars else []

    pieces: List[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            pieces.append(paragraph)
            continue

        buffer = ""
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            for part in _hard_split(sentence.strip(), chunk_size):
                if not part:
                    continue
                if not buffer:
                    buffer = part
                elif len(buffer) + len(part) + 1 <= chunk_size:
                    buffer = f"{buffer} {part}"
                else:
                    pieces.append(buffer)
                    buffer = part
        if buffer:
            pieces.append(buffer)

    # 合并相邻的短片段，避免图注、过渡句被单独切碎
    merged: List[str] = []
    for piece in pieces:
        if (
            merged
            and len(merged[-1]) + len(piece) + 1 <= chunk_size
            and len(merged[-1]) < chunk_size * 0.6
        ):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    if overlap > 0 and len(merged) > 1:
        with_overlap = [merged[0]]
        for index in range(1, len(merged)):
            tail = merged[index - 1][-overlap:].strip()
            with_overlap.append(f"{tail} {merged[index]}".strip() if tail else merged[index])
        merged = with_overlap

    # 丢弃过短片段（图注残片、页码残留、单行页眉）
    filtered = [piece for piece in merged if len(piece) >= min_chars]
    if filtered:
        return filtered
    return [merged[0]] if merged else []


def paper_to_documents(
    paper: ParsedPaper,
    source: str,
    source_key: str,
    extra_meta: Optional[Dict] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_chars: Optional[int] = None,
    contextual_header: Optional[bool] = None,
    dedupe: Optional[bool] = None,
    skip_references: bool = True,
) -> List[Document]:
    """把解析结果转成可入库的 Document 列表。

    三层去重在这里收口：

    1. 文档级：`source_key` 由文件内容指纹（PDF）或 arXiv ID 决定，
       同一篇文献重复导入会生成同一批 id，写入时是覆盖而不是新增；
    2. 片段级：片段 id = md5(来源 + 章节序号 + 片段序号)，位置固定；
    3. 文本级：同一个来源内文本完全相同的片段只保留一份（`RAG_DEDUPE_CHUNKS`）。

    另外每个片段都带「标题 / 章节 / 页码」等元数据，供引用溯源使用。
    """
    title = paper.title or source
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = CHUNK_OVERLAP if overlap is None else overlap
    min_chars = MIN_CHUNK_CHARS if min_chars is None else min_chars
    contextual_header = CONTEXTUAL_HEADER if contextual_header is None else contextual_header
    dedupe = DEDUPE_CHUNKS if dedupe is None else dedupe
    documents: List[Document] = []
    seen_paragraphs: set = set()
    seen_chunks: set = set()
    total_dropped = 0

    for section_index, section in enumerate(paper.sections):
        if skip_references and is_reference_section(section.title):
            continue
        chunks = split_text(
            section.text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chars=min_chars,
            seen_paragraphs=seen_paragraphs if dedupe else None,
        )
        section_slug = _slug(section.title)
        for chunk_index, chunk in enumerate(chunks):
            # 片段级兜底：文本完全相同的片段（含上下文头）只保留一份
            fingerprint = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            if dedupe and fingerprint in seen_chunks:
                total_dropped += 1
                continue
            seen_chunks.add(fingerprint)

            doc_id = hashlib.md5(
                f"{source_key}|{section_index}|{chunk_index}".encode("utf-8")
            ).hexdigest()
            content = chunk
            if contextual_header:
                content = f"《{title}》 {section.title}\n{chunk}"

            metadata: Dict = {
                "name": f"{source_key}#{section_slug}-{chunk_index}",
                "source": source,
                "source_key": source_key,
                "title": title,
                "authors": paper.authors,
                "year": paper.year,
                "section": section.title,
                "page": section.page,
                "chunk_index": chunk_index,
                "chunk_total": len(chunks),
            }
            if extra_meta:
                metadata.update({key: value for key, value in extra_meta.items() if value})

            documents.append(
                Document(
                    id=doc_id,
                    page_content=content.strip(),
                    metadata=metadata,
                )
            )

    if total_dropped:
        logger.info("%s：去重丢弃 %d 个重复片段", source, total_dropped)
    return documents
