"""文档装载：把 PDF 文件 / arXiv 论文变成可直接入库的 Document 列表。

这里只做「输入源 → 文档」的翻译，不负责写向量库（那是 knowledge_base 的事）。
一条 `IngestResult` 同时带回入库所需的所有信息（标题、作者、章节数、解析器），
前端据此给出「已入库 N 个片段：《标题》」这样的提示。
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

from rag.arxiv import download_arxiv_pdf, fetch_arxiv_paper, normalize_arxiv_id
from rag.chunking import paper_to_documents
from rag.config import INCLUDE_REFERENCES
from rag.parsers import ParsedPaper, extract_references, parse_pdf_bytes

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """一次装载的结果：文档列表 + 元数据摘要。"""

    documents: List[Document]
    source: str
    source_key: str
    title: str = ""
    authors: str = ""
    year: str = ""
    sections: int = 0
    parser: str = ""
    references_text: str = ""

    @property
    def added(self) -> int:
        """本次装载的片段数。"""
        return len(self.documents)


def _build_result(
    paper: ParsedPaper, documents: List[Document], source: str, source_key: str
) -> IngestResult:
    return IngestResult(
        documents=documents,
        source=source,
        source_key=source_key,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        sections=len(paper.sections),
        parser=paper.parser,
        references_text=extract_references(paper),
    )


def load_pdf_bytes(
    data: bytes,
    filename: str = "",
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_chars: Optional[int] = None,
    contextual_header: Optional[bool] = None,
    dedupe: Optional[bool] = None,
) -> IngestResult:
    """解析前端上传的 PDF 字节流。"""
    if not data:
        raise ValueError("PDF 内容为空")

    fallback_title = Path(filename).stem if filename else ""
    # 始终连参考文献一起解析：正文入库时仍会跳过它，但引用图谱需要这份列表
    paper = parse_pdf_bytes(data, fallback_title=fallback_title, include_references=True)

    # 用文件内容指纹做来源标识：同一篇文献重复上传不会重复入库，不同文件也不会撞车
    source_key = f"pdf:{hashlib.sha1(data).hexdigest()[:12]}"
    source = filename or f"{source_key}.pdf"
    documents = paper_to_documents(
        paper,
        source=source,
        source_key=source_key,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chars=min_chars,
        contextual_header=contextual_header,
        dedupe=dedupe,
        skip_references=not INCLUDE_REFERENCES,
    )

    logger.info(
        "解析 PDF %s：%s（%s，%d 节，%d 个片段）",
        source,
        paper.title or "-",
        paper.parser or "-",
        len(paper.sections),
        len(documents),
    )
    return _build_result(paper, documents, source, source_key)


def load_pdf_file(path: Union[str, Path]) -> IngestResult:
    """解析本地 PDF 文件（脚本 / 批量导入用）。"""
    file_path = Path(path)
    return load_pdf_bytes(file_path.read_bytes(), filename=file_path.name)


def load_arxiv(
    arxiv_id: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_chars: Optional[int] = None,
    contextual_header: Optional[bool] = None,
    dedupe: Optional[bool] = None,
) -> IngestResult:
    """按 arXiv ID 抓取元数据 + PDF 并解析。

    `export.arxiv.org`（API）和 `arxiv.org`（PDF）是两个不同的入口，
    有时只有其中一个可达：API 挂了就从 PDF 解析标题/作者/年份兜底。
    """
    meta = None
    try:
        meta = fetch_arxiv_paper(arxiv_id)
    except RuntimeError as exc:
        logger.warning("arXiv 元数据接口不可用，改从 PDF 解析元数据：%s", exc)

    code = meta.arxiv_id if meta else normalize_arxiv_id(arxiv_id)
    data = download_arxiv_pdf(code)
    paper = parse_pdf_bytes(
        data,
        fallback_title=meta.title if meta else f"arXiv {code}",
        include_references=True,
    )

    # arXiv 的元数据比从 PDF 里猜的准，优先用它
    if meta is not None:
        if meta.title:
            paper.title = meta.title
        if meta.authors:
            paper.authors = meta.authors
        if meta.year:
            paper.year = meta.year
    elif not paper.title:
        paper.title = f"arXiv {code}"

    source_key = f"arxiv:{code}"
    source = f"arXiv:{code}"
    extra_meta: Dict[str, Optional[str]] = {
        "arxiv_id": code,
        "url": f"https://arxiv.org/abs/{code}",
    }
    documents = paper_to_documents(
        paper,
        source=source,
        source_key=source_key,
        extra_meta=extra_meta,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chars=min_chars,
        contextual_header=contextual_header,
        dedupe=dedupe,
        skip_references=not INCLUDE_REFERENCES,
    )

    logger.info(
        "导入 arXiv:%s：%s（%d 节，%d 个片段，元数据来自%s）",
        code,
        paper.title or "-",
        len(paper.sections),
        len(documents),
        "arXiv API" if meta is not None else "PDF",
    )
    return _build_result(paper, documents, source, source_key)
