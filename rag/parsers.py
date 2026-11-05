"""版面感知的学术 PDF 解析。

目标不是「把 PDF 变成一坨文字」，而是尽量还原论文结构：
标题 / 作者 / 年份 / 章节（Abstract、1 Introduction、2 Related Work …），
供后续按章节切分与引用溯源使用。

优先使用 PyMuPDF（能拿到字号、加粗、坐标，用来识别标题层级、页眉页脚）；
未安装或解析失败时退化为 pypdf 的纯文本解析（只能靠正则识别章节标题）。
"""

import logging
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from rag.cleaning import clean_text
from rag.config import INCLUDE_REFERENCES

logger = logging.getLogger(__name__)


@dataclass
class ParsedSection:
    """论文中的一个章节（含它所在的起始页）。"""

    title: str
    page: int
    text: str


@dataclass
class ParsedPaper:
    """一篇论文的结构化解析结果。"""

    title: str = ""
    authors: str = ""
    year: str = ""
    page_count: int = 0
    parser: str = ""
    sections: List[ParsedSection] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(section.text) for section in self.sections)


# 常见章节名（小写、去掉编号后比较），中英文都覆盖
_KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "related works", "background",
    "preliminaries", "preliminary", "method", "methods", "methodology",
    "approach", "proposed method", "proposed approach", "our approach",
    "model", "models", "architecture", "system design", "implementation",
    "experiment", "experiments", "experimental setup", "experimental results",
    "results", "result", "evaluation", "results and discussion",
    "results and analysis", "ablation study", "ablation studies", "analysis",
    "discussion", "limitations", "conclusion", "conclusions", "future work",
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "appendix", "摘要", "引言", "绪论", "相关工作", "研究背景", "方法",
    "研究方法", "实验", "实验结果", "结果与分析", "讨论", "结论",
    "总结与展望", "参考文献", "致谢", "附录",
}

# 从这里开始的内容默认不入库（参考文献既长又噪声大）
_REFERENCES_HEADINGS = {
    "references", "bibliography", "reference", "参考文献", "引用文献",
}

_NUMBERING_RE = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,3}|[IVX]{1,5})[\.\)、]?\s+")
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,3}[\.\)、]\s*\S.*$")
_PAGE_NUMBER_RE = re.compile(r"^[\s\-–—·•]*\d{1,4}[\s\-–—·•]*$")
_ARXIV_MARGIN_RE = re.compile(r"^arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

_TITLE_NOISE = ("untitled", "microsoft word", "microsoft power", ".doc", ".tex", "document1")


def _clean_inline(text: Optional[str]) -> str:
    return clean_text(text)


def _normalize_line(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip()).lower()


def _strip_numbering(text: str) -> str:
    return _NUMBERING_RE.sub("", text).strip()


def _heading_key(text: str) -> str:
    key = _strip_numbering(text)
    return key.strip().strip("—-–:：.。").strip().lower()


def is_reference_section(title: str) -> bool:
    """这个章节标题是不是「参考文献 / References」。"""
    return _heading_key(title) in _REFERENCES_HEADINGS


def extract_references(paper: ParsedPaper) -> str:
    """把参考文献章节拼成一段纯文本（引用图谱的本地兜底用）。"""
    parts = [
        section.text
        for section in paper.sections
        if is_reference_section(section.title) and (section.text or "").strip()
    ]
    return "\n".join(parts).strip()


def _join_lines(texts: List[str]) -> str:
    """把 PDF 的硬换行拼回自然段落，处理行尾连字符断词。"""
    parts: List[str] = []
    for raw in texts:
        text = clean_text(raw)
        if not text:
            continue
        if parts and parts[-1].endswith("-") and text[:1].islower():
            parts[-1] = parts[-1][:-1] + text
        else:
            parts.append(text)
    return " ".join(parts)


# --------------------------------------------------------------------- 行提取
def _page_lines(page) -> List[Dict]:
    """取出一页里所有文本行，附带字号 / 加粗 / 纵坐标。"""
    lines: List[Dict] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            # 顺手丢掉纯符号行（装饰线、孤立项目符号），它们对检索只会造成噪声
            if not any(char.isalnum() for char in text):
                continue
            sizes = [span.get("size", 0.0) for span in spans if span.get("text", "").strip()]
            lines.append(
                {
                    "text": text,
                    "size": round(max(sizes), 1) if sizes else 0.0,
                    "bold": any(span.get("flags", 0) & 16 for span in spans),
                    "y": float(line.get("bbox", (0.0, 0.0, 0.0, 0.0))[1]),
                }
            )
    return lines


def _drop_headers_and_footers(pages: List[List[Dict]], heights: List[float]) -> List[List[Dict]]:
    """去掉页眉、页脚、页码与 arXiv 侧边标记。"""
    margin_counts: Dict[str, int] = {}
    for lines, height in zip(pages, heights):
        if not height:
            continue
        for line in lines:
            if line["y"] < height * 0.08 or line["y"] > height * 0.92:
                key = _normalize_line(line["text"])
                margin_counts[key] = margin_counts.get(key, 0) + 1

    repeated = {key for key, count in margin_counts.items() if count >= 2}

    cleaned: List[List[Dict]] = []
    for lines, height in zip(pages, heights):
        kept: List[Dict] = []
        for line in lines:
            text = line["text"]
            in_margin = bool(height) and (line["y"] < height * 0.08 or line["y"] > height * 0.92)
            key = _normalize_line(text)

            if _ARXIV_MARGIN_RE.match(text):
                continue
            if _PAGE_NUMBER_RE.match(text) and (in_margin or len(text.strip()) <= 3):
                continue
            if in_margin and key in repeated:
                continue
            kept.append(line)
        cleaned.append(kept)
    return cleaned


def _body_font_size(pages: List[List[Dict]]) -> float:
    """用「长行」的字号众数估计正文大小，标题/图表文字不会带偏它。"""
    weights: Dict[float, int] = {}
    for lines in pages:
        for line in lines:
            if len(line["text"]) >= 25 and line["size"] > 0:
                weights[line["size"]] = weights.get(line["size"], 0) + len(line["text"])
    if not weights:
        return 0.0
    return max(weights, key=weights.get)


# --------------------------------------------------------------------- 结构识别
def _detect_heading(line: Dict, body_size: float) -> Optional[str]:
    """判断一行是不是章节标题。"""
    text = line["text"].strip()
    if not text or len(text) > 90:
        return None
    if text.endswith((".", "。", ",", "，", ";", "；", ":", "：")):
        return None

    key = _heading_key(text)
    if key in _KNOWN_HEADINGS:
        return text
    if _NUMBERED_HEADING_RE.match(text) and (body_size <= 0 or line["size"] >= body_size * 0.95):
        return text
    if body_size > 0 and line["size"] >= body_size * 1.15 and len(text) <= 70:
        return text
    return None


def _detect_title_block(
    lines: List[Dict], body_size: float, page_height: float
) -> Tuple[str, str, int]:
    """从第一页顶部找出标题与作者，返回 (标题, 作者, 已消费的行数)。"""
    if not lines:
        return "", "", 0

    limit = page_height * 0.4 if page_height else float("inf")
    head_indexes = [index for index, line in enumerate(lines) if line["y"] <= limit]
    if not head_indexes:
        head_indexes = list(range(min(len(lines), 6)))

    top_size = max(lines[index]["size"] for index in head_indexes)
    if top_size <= 0:
        return "", "", 0

    starts = [
        index
        for index in head_indexes
        if lines[index]["size"] >= top_size - 0.6 and len(lines[index]["text"]) > 3
    ]
    if not starts:
        return "", "", 0

    start = min(starts)
    end = start
    for index in range(start, len(lines)):
        line = lines[index]
        if line["size"] >= top_size - 0.6 and len(line["text"]) > 3:
            end = index
        else:
            break

    title = _join_lines([lines[index]["text"] for index in range(start, end + 1)])

    authors_parts: List[str] = []
    consumed = end + 1
    for index in range(end + 1, min(end + 8, len(lines))):
        line = lines[index]
        key = _heading_key(line["text"])
        if key.startswith("abstract") or key in _KNOWN_HEADINGS:
            break
        if body_size > 0 and line["size"] < body_size * 0.85:
            break
        if len(line["text"]) > 200:
            break
        authors_parts.append(line["text"])
        consumed = index + 1

    authors = _join_lines(authors_parts)
    return _clean_inline(title), _clean_inline(authors), consumed


def _build_sections(
    pages: List[List[Dict]],
    body_size: float,
    include_references: bool,
    skip_first_page_lines: int = 0,
) -> List[ParsedSection]:
    """按章节标题把整篇论文拼成若干段。"""
    sections: List[ParsedSection] = []
    current_title = "Abstract"
    current_page = 1
    buffer: List[str] = []
    stopped = False

    def flush() -> None:
        text = _join_lines(buffer).strip()
        if text:
            sections.append(ParsedSection(title=current_title, page=current_page, text=text))
        buffer.clear()

    for page_number, lines in enumerate(pages, start=1):
        if stopped:
            break
        page_lines = lines[skip_first_page_lines:] if page_number == 1 else lines
        for line in page_lines:
            heading = _detect_heading(line, body_size)
            if heading:
                key = _heading_key(heading)
                if key in _REFERENCES_HEADINGS and not include_references:
                    flush()
                    stopped = True
                    break
                flush()
                current_title = heading
                current_page = page_number
                continue
            if not buffer:
                current_page = page_number
            buffer.append(line["text"])

    flush()
    return sections


def _metadata_title(metadata: Dict) -> str:
    title = _clean_inline(metadata.get("title"))
    if len(title) < 6:
        return ""
    lowered = title.lower()
    if any(noise in lowered for noise in _TITLE_NOISE):
        return ""
    return title


def _metadata_year(metadata: Dict) -> str:
    for key in ("creationDate", "modDate"):
        match = re.search(r"(19|20)\d{2}", metadata.get(key) or "")
        if match:
            return match.group(0)
    return ""


# --------------------------------------------------------------------- 两个解析器
_MUPDF_QUIET = False


def _silence_mupdf(fitz_module) -> None:
    """关掉 MuPDF 往 stderr 打的解析噪音。

    有些期刊导出的 PDF 里 colorspace 是坏的（日志里的
    `unknown colorspace: R9`），MuPDF 每页都会报一次。文本层通常还能正常
    提取，这些日志只是噪音，不影响解析结果。
    """
    global _MUPDF_QUIET
    if _MUPDF_QUIET:
        return
    try:
        fitz_module.TOOLS.mupdf_display_errors(False)
        fitz_module.TOOLS.mupdf_display_warnings(False)
    except Exception:  # noqa: BLE001 - 老版本没有这个开关就保持原样
        pass
    _MUPDF_QUIET = True


def _parse_with_pymupdf(
    data: bytes, include_references: bool, fallback_title: str
) -> ParsedPaper:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover - 老版本只有 fitz
        import fitz  # type: ignore

    _silence_mupdf(fitz)
    document = fitz.open(stream=data, filetype="pdf")
    try:
        pages = [_page_lines(page) for page in document]
        heights = [float(page.rect.height or 0.0) for page in document]
        metadata = dict(document.metadata or {})
        page_count = int(document.page_count)
    finally:
        document.close()

    pages = _drop_headers_and_footers(pages, heights)
    body_size = _body_font_size(pages)
    title, authors, consumed = _detect_title_block(
        pages[0] if pages else [],
        body_size,
        heights[0] if heights else 0.0,
    )
    sections = _build_sections(
        pages,
        body_size,
        include_references=include_references,
        skip_first_page_lines=consumed,
    )

    return ParsedPaper(
        title=_clean_inline(title) or _metadata_title(metadata) or fallback_title,
        authors=_clean_inline(authors),
        year=_metadata_year(metadata),
        page_count=page_count,
        parser="pymupdf",
        sections=sections,
    )


def _parse_with_pypdf(
    data: bytes, include_references: bool, fallback_title: str
) -> ParsedPaper:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要安装 PyMuPDF 或 pypdf：pip install pymupdf") from exc

    reader = PdfReader(BytesIO(data))
    pages: List[List[Dict]] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [
            {"text": line.strip(), "size": 0.0, "bold": False, "y": 0.0}
            for line in text.splitlines()
        ]
        pages.append([line for line in lines if line["text"]])

    title = pages[0][0]["text"] if pages and pages[0] else ""
    sections = _build_sections(
        pages, 0.0, include_references=include_references, skip_first_page_lines=0
    )

    return ParsedPaper(
        title=_clean_inline(title) or fallback_title,
        authors="",
        year="",
        page_count=len(pages),
        parser="pypdf",
        sections=sections,
    )


def parse_pdf_bytes(
    data: bytes,
    fallback_title: str = "",
    include_references: bool = INCLUDE_REFERENCES,
) -> ParsedPaper:
    """解析 PDF 字节流；PyMuPDF 不可用时自动退化到 pypdf。"""
    if not data:
        raise ValueError("PDF 内容为空")

    try:
        return _parse_with_pymupdf(data, include_references, fallback_title)
    except ImportError:
        logger.info("未安装 PyMuPDF，改用 pypdf 解析")
    except Exception as exc:  # noqa: BLE001 - 坏文件、加密 PDF 等都要能降级
        logger.warning("PyMuPDF 解析失败（%s），改用 pypdf", exc)

    return _parse_with_pypdf(data, include_references, fallback_title)
