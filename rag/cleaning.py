"""文本清洗：把 PDF 提取出来的脏文本整理成可入库的干净文本。

PDF 抽出来的文字常见问题：控制字符（\\x00 等）、软连字符、零宽字符、连字
（ﬁ→fi）、缺字体导致的 `(cid:123)` 伪字符、全角/不同宽度空格、连续空行。
这些都会污染向量表示、拉低检索质量，也会让回答里冒出奇怪符号。

清洗原则：只做无损或近无损的规范化，不改句意，也不破坏段落换行
（章节切分依赖换行判断段落边界）。
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# 除 \t \n 之外的控制字符：PDF 中常见，且会让 JSON / Arrow 写入报错
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 软连字符、零宽空格、零宽连接符、BOM 等不可见字符
_INVISIBLE_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
# 常见英文连字：PDF 里 fi / fl 常被抽成单个字符
_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }
)
# 缺字体时 PDF 会抽成 (cid:123) 这类伪字符
_CID_RE = re.compile(r"\(cid:\d+\)")
# 各种宽度的空格统一成普通空格（全角空格、不换行空格等）
_SPACE_RE = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
# 三个及以上连续换行压成空一行
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: Optional[str], normalize_unicode: bool = True) -> str:
    """规范化单段文本：去控制符与不可见字符、展开连字、压缩空白。"""
    if not text:
        return ""

    cleaned = _CID_RE.sub(" ", text)
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _INVISIBLE_RE.sub("", cleaned)
    cleaned = cleaned.translate(_LIGATURES)

    if normalize_unicode:
        # NFKC：全角字母数字/标点 → 半角，兼容字符归一（中文汉字不受影响）
        cleaned = unicodedata.normalize("NFKC", cleaned)

    cleaned = _SPACE_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_section_text(text: Optional[str], max_chars: int = 200_000) -> str:
    """清洗整节正文；单节异常长时截断，避免一条脏数据拖垮整次入库。"""
    cleaned = clean_text(text)
    if max_chars and len(cleaned) > max_chars:
        logger.warning("章节正文过长（%d 字），已截断到 %d 字", len(cleaned), max_chars)
        cleaned = cleaned[:max_chars]
    return cleaned
