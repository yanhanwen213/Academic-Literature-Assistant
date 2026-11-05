"""arXiv 论文导入：把「arXiv ID / 论文链接」变成可入库的 PDF 与元数据。

相比抓网页，arXiv 有官方 API：一条 ID 对应一篇论文，元数据（标题、作者、年份、
摘要）都是结构化的，PDF 也能直接下载，因此更适合做学术文献助手的数据源。

注意：arXiv 会限流，短时间内大量导入建议放慢速度。
"""

import logging
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

from rag.config import ARXIV_API_BASE, ARXIV_PROXY, ARXIV_TIMEOUT

logger = logging.getLogger(__name__)

_USER_AGENT = "AcademicLiteratureAssistant/1.0 (arXiv API client)"
_ATOM = "{http://www.w3.org/2005/Atom}"
_WHITESPACE_RE = re.compile(r"\s+")

# arXiv 的网关经常 5 秒返回一个空 body 的 502，重试一两次通常就好了
_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_TIMES = 1
_RETRY_BACKOFF = 2.0

# 新式 ID：2401.00001 ；老式 ID：hep-th/9901001
_NEW_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_OLD_ID_RE = re.compile(r"([a-zA-Z\-]+(?:\.[A-Za-z\-]+)?/\d{7})(v\d+)?")


def _build_opener() -> urllib.request.OpenerDirector:
    """arXiv 专用 opener：只有配了 ARXIV_PROXY 才走代理。

    这样代理只影响 arXiv（国内常被墙），不会把 DeepSeek / 本地 Ollama
    也一起塞进代理里。
    """
    if ARXIV_PROXY:
        handler = urllib.request.ProxyHandler(
            {"http": ARXIV_PROXY, "https": ARXIV_PROXY}
        )
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()


_OPENER = _build_opener()


def _api_bases() -> List[str]:
    if ARXIV_API_BASE:
        return [ARXIV_API_BASE]
    # 先 https（官方推荐），再退回 http（部分校园网只放行 80 端口）
    return [
        "https://export.arxiv.org/api/query",
        "http://export.arxiv.org/api/query",
    ]


def _hint_for_proxy() -> str:
    return (
        "如果 arXiv 在你所在网络被墙/被网关拦截，可以在 .env 里加 "
        "ARXIV_PROXY=http://127.0.0.1:7890（换成你自己的代理端口），"
        "或者直接下载 PDF 再上传。"
    )


def _http_error_message(code: int, what: str, url: str) -> str:
    if code in (502, 503, 504):
        return (
            f"下载 {what} 失败：HTTP {code}（{url}）。"
            f"这是 arXiv 服务端或你所在网络的出口网关暂时不可用，重试通常能恢复；"
            f"反复出现说明出口被拦。{_hint_for_proxy()}"
        )
    if code == 429:
        return (
            f"下载 {what} 失败：HTTP 429（{url}）。arXiv 限流了，"
            "隔几分钟再试；批量导入请放慢速度。"
        )
    if code == 404:
        return f"下载 {what} 失败：HTTP 404（{url}）。arXiv 上可能没有这篇论文。"
    if code in (401, 403):
        return f"下载 {what} 失败：HTTP {code}（{url}）。访问被拒绝，可能需要代理。{_hint_for_proxy()}"
    return f"下载 {what} 失败：HTTP {code}（{url}）"


def _network_error_message(what: str, url: str, reason) -> str:
    text = str(reason or "")
    if "SSL" in text or "EOF" in text or "CERTIFICATE" in text.upper():
        return (
            f"无法连接 {what}：TLS 握手被中断（{text}）。"
            f"这种报错通常是网络层拦截 arXiv，而不是证书问题。{_hint_for_proxy()}"
        )
    return (
        f"无法连接 {what}：{text}（{url}）。请检查网络或代理设置。{_hint_for_proxy()}"
    )



@dataclass
class ArxivPaper:
    """arXiv 上的论文元数据。"""

    arxiv_id: str
    title: str = ""
    authors: str = ""
    year: str = ""
    abstract: str = ""
    pdf_url: str = ""

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def _clean(text: Optional[str]) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def normalize_arxiv_id(raw: str) -> str:
    """从 ID / abs 链接 / PDF 链接里提取标准 arXiv ID（去掉版本号）。"""
    text = (raw or "").strip()
    if not text:
        raise ValueError("请输入 arXiv ID 或论文链接，例如 2401.00001")

    match = _NEW_ID_RE.search(text) or _OLD_ID_RE.search(text)
    if match is None:
        raise ValueError(
            f"无法识别的 arXiv ID：{raw}"
            "（示例：2401.00001 或 https://arxiv.org/abs/2401.00001）"
        )
    return match.group(1)


def _get_bytes(
    url: str, what: str, timeout: int, retries: int = _RETRY_TIMES
) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            with _OPENER.open(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRY_STATUS and attempt < retries:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning(
                    "%s 返回 HTTP %s，%.0fs 后重试（第 %d 次）",
                    what,
                    exc.code,
                    wait,
                    attempt + 1,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(_http_error_message(exc.code, what, url)) from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < retries:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning("%s 连接失败（%s），%.0fs 后重试", what, exc.reason, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(
                _network_error_message(what, url, exc.reason)
            ) from exc
        except OSError as exc:
            if attempt < retries:
                wait = _RETRY_BACKOFF * (attempt + 1)
                logger.warning("%s 超时（%s），%.0fs 后重试", what, exc, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"下载 {what} 超时或网络异常：{exc}（{url}）。{_hint_for_proxy()}"
            ) from exc
    raise RuntimeError(
        f"下载 {what} 失败：{last_error}（{url}）。{_hint_for_proxy()}"
    )


def fetch_arxiv_paper(arxiv_id: str, timeout: int = ARXIV_TIMEOUT) -> ArxivPaper:
    """调用 arXiv API 拿论文元数据（https → http 双通道 + 重试）。"""
    normalized = normalize_arxiv_id(arxiv_id)

    payload: Optional[bytes] = None
    errors: List[str] = []
    for base in _api_bases():
        url = f"{base}?id_list={normalized}&max_results=1"
        try:
            payload = _get_bytes(url, what="arXiv API", timeout=timeout)
            break
        except RuntimeError as exc:
            errors.append(str(exc))
            logger.warning("arXiv API 通道失败：%s", exc)

    if payload is None:
        raise RuntimeError(
            "arXiv API 当前不可达：" + "；".join(errors)
        )

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"arXiv 返回的数据无法解析：{exc}") from exc

    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        raise RuntimeError(f"arXiv 上找不到论文 {normalized}")

    entry_id = entry.findtext(f"{_ATOM}id") or ""
    if "api/errors" in entry_id:
        detail = _clean(entry.findtext(f"{_ATOM}summary"))
        raise RuntimeError(f"arXiv 返回错误：{detail or '未找到该论文'}")

    authors = ", ".join(
        _clean(author.findtext(f"{_ATOM}name"))
        for author in entry.findall(f"{_ATOM}author")
    ).strip(", ")

    published = entry.findtext(f"{_ATOM}published") or ""
    pdf_url = ""
    for link in entry.findall(f"{_ATOM}link"):
        if (link.get("title") or "").lower() == "pdf":
            pdf_url = link.get("href") or ""
            break

    return ArxivPaper(
        arxiv_id=normalized,
        title=_clean(entry.findtext(f"{_ATOM}title")),
        authors=authors,
        year=published[:4] if len(published) >= 4 else "",
        abstract=_clean(entry.findtext(f"{_ATOM}summary")),
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{normalized}",
    )


def download_arxiv_pdf(arxiv_id: str, timeout: int = ARXIV_TIMEOUT) -> bytes:
    """下载 arXiv 论文 PDF 的原始字节。"""
    normalized = normalize_arxiv_id(arxiv_id)
    url = f"https://arxiv.org/pdf/{normalized}"
    data = _get_bytes(url, what="arXiv PDF", timeout=timeout, retries=2)
    if not data.startswith(b"%PDF"):
        preview = data[:80].decode("utf-8", "ignore").replace("\n", " ")
        raise RuntimeError(
            "arXiv 没有返回 PDF 文件（通常是访问过于频繁被限流，或返回了网页）："
            f"{preview}。请稍后重试，或直接下载 PDF 后上传。"
        )
    return data
