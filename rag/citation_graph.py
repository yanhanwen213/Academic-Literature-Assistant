"""引用图谱：把「论文-引用」关系建成有向图，支持多跳检索。

单步 RAG 只能回答「这篇论文说了什么」，回答不了：
「谁改进了它？」「它和 X 的差异在哪？」「这条技术路线是怎么演进的？」

这些问题的答案不在任何一段正文里，而在**引用网络**里。做法：

1. 抽取引用关系：优先用 Semantic Scholar API（有真实 cited-by），
   拿不到时退化为从 PDF 里解析的参考文献（只能拿到「它引用了谁」）；
2. 用 networkx 建有向图：边 A → B 表示「A 引用了 B」（B 是被引用者）；
3. 提供三个多跳工具：get_references / get_citations / trace_lineage。
"""

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rag.config import (
    GRAPH_HTTP_TIMEOUT,
    GRAPH_MAX_HOPS,
    GRAPH_PATH,
    SEMANTIC_SCHOLAR_API,
    SEMANTIC_SCHOLAR_API_KEY,
)

logger = logging.getLogger(__name__)

_USER_AGENT = "AcademicLiteratureAssistant/1.0 (citation graph)"

# Semantic Scholar 的 key 额度是「1 请求/秒，全端点累计」。
# 因此所有 S2 请求都过同一道闸门：两次请求之间至少间隔 _S2_MIN_INTERVAL 秒。
# 有 key 时 1.1s（留 10% 余量），没 key 时官方是 5 分钟 100 次（≈3s/次）。
_DEFAULT_INTERVAL_WITH_KEY = 1.1
_DEFAULT_INTERVAL_WITHOUT_KEY = 3.1
try:
    _S2_MIN_INTERVAL = float(
        os.getenv("SEMANTIC_SCHOLAR_MIN_INTERVAL", "") or 0
    ) or (
        _DEFAULT_INTERVAL_WITH_KEY
        if SEMANTIC_SCHOLAR_API_KEY
        else _DEFAULT_INTERVAL_WITHOUT_KEY
    )
except ValueError:
    _S2_MIN_INTERVAL = _DEFAULT_INTERVAL_WITH_KEY

_S2_LOCK = threading.Lock()
_S2_LAST_CALL = 0.0

# 自适应限速：429 多说明额度比 1 req/s 更紧（S2 免费额度在高峰期会被共享限流），
# 命中 429 就把间隔拉长，成功后再慢慢收紧回配置值。
_S2_INTERVAL_MAX = 8.0
_S2_INTERVAL_CURRENT = _S2_MIN_INTERVAL

# 429 时最多重试几次；优先按 Retry-After，其次指数退避
_HTTP_RETRIES = int(os.getenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "3") or 3)
_HTTP_BACKOFF_SECONDS = 2.0
_HTTP_MAX_WAIT_SECONDS = 20.0

# 同一篇论文在一次运行里只解析一次 S2 ID
_RESOLVE_CACHE: Dict[str, str] = {}


def s2_request_interval() -> float:
    """当前生效的 S2 请求间隔（秒），前端用来估算耗时。"""
    return _S2_INTERVAL_CURRENT


_S2_STATS = {"requests": 0, "throttled": 0, "retries": 0}


def s2_stats() -> Dict[str, int]:
    """本次进程内的 S2 请求统计（给管理后台的健康检查用）。"""
    return dict(_S2_STATS)


def _slow_down() -> None:
    """命中 429 后拉长间隔，并记录到日志（用户能看到“慢速模式”）。"""
    global _S2_INTERVAL_CURRENT
    with _S2_LOCK:
        updated = min(_S2_INTERVAL_MAX, max(_S2_INTERVAL_CURRENT * 1.6, 2.0))
        if updated > _S2_INTERVAL_CURRENT + 0.05:
            _S2_INTERVAL_CURRENT = updated
            logger.warning(
                "Semantic Scholar 仍在限流，请求间隔调整为 %.1fs（上限 %.1fs）",
                _S2_INTERVAL_CURRENT,
                _S2_INTERVAL_MAX,
            )


def _speed_up() -> None:
    """请求成功后慢慢收紧回配置值，避免一次 429 就永久变慢。"""
    global _S2_INTERVAL_CURRENT
    with _S2_LOCK:
        if _S2_INTERVAL_CURRENT > _S2_MIN_INTERVAL:
            _S2_INTERVAL_CURRENT = max(
                _S2_MIN_INTERVAL, _S2_INTERVAL_CURRENT * 0.92
            )

# PDF 参考文献条目通常以 [12] / 12. / (12) 开头
_REF_ENTRY_START = re.compile(r"^\s*(?:\[\s*\d{1,3}\s*\]|\(?\d{1,3}\s*[\.\)])\s+")
_MAX_LOCAL_REFS = 120


@dataclass
class PaperNode:
    """图里的一个论文节点。"""

    paper_id: str
    title: str = ""
    year: str = ""
    authors: str = ""
    venue: str = ""
    abstract: str = ""
    in_library: bool = False
    source_key: str = ""


def _split_reference_entries(text: str, limit: int = _MAX_LOCAL_REFS) -> List[str]:
    """把参考文献章节切成一條條引文。

    PDF 抽出来的参考文献有三种常见排版：
    1. 全部挤在同一个物理行里，靠编号 `/ [12]` 分隔——按编号正则切；
    2. 一条一行、带编号——按行切；
    3. 一条一行、无编号——按行切。
    """
    raw = (text or "").strip()
    if not raw:
        return []

    bracket_markers = re.findall(r"\[\s*\d{1,3}\s*\]", raw)
    if len(bracket_markers) >= 3:
        parts = re.split(r"(?=\[\s*\d{1,3}\s*\]\s*)", raw)
        entries = []
        for part in parts:
            cleaned = _REF_ENTRY_START.sub("", part, count=1)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) > 12:
                entries.append(cleaned)
        return entries[:limit]

    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]

    numbered = [line for line in lines if _REF_ENTRY_START.match(line)]
    if len(numbered) >= 3:
        entries = []
        for line in lines:
            match = _REF_ENTRY_START.match(line)
            if match:
                entries.append(line[match.end():].strip())
            elif entries:
                entries[-1] = f"{entries[-1]} {line}".strip()
        return [entry for entry in entries if len(entry) > 12][:limit]

    return [line for line in lines if len(line) > 40][:limit]


def _guess_year(text: str) -> str:
    """从引文字符串里猜年份：先剔除「2017–2025」这类页码范围，再取最后一个 19xx/20xx。"""
    cleaned = re.sub(r"(?:19|20)\d{2}\s*[–—\-~]\s*(?:19|20)\d{2}", " ", text or "")
    years = re.findall(r"(?:19|20)\d{2}", cleaned)
    return years[-1] if years else ""


def _throttle() -> None:
    """全局限速：保证两次 S2 请求之间至少间隔 _S2_MIN_INTERVAL 秒。

    锁在 sleep 期间一直持有，所以多线程调用也会被串行化——1 req/s 的额度
    是「全端点累计」的，并发发请求只会一起 429。
    """
    global _S2_LAST_CALL
    with _S2_LOCK:
        wait = _S2_INTERVAL_CURRENT - (time.monotonic() - _S2_LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _S2_LAST_CALL = time.monotonic()


def _retry_wait(exc: urllib.error.HTTPError, attempt: int) -> float:
    """429 时优先听 Retry-After，其次指数退避（带抖动）。"""
    retry_after = ""
    try:
        retry_after = (exc.headers or {}).get("Retry-After", "")
    except Exception:  # noqa: BLE001
        retry_after = ""
    if retry_after:
        try:
            return min(max(float(retry_after), _S2_INTERVAL_CURRENT), _HTTP_MAX_WAIT_SECONDS)
        except ValueError:
            pass
    base = _HTTP_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 0.4)
    return min(max(base, _S2_INTERVAL_CURRENT), _HTTP_MAX_WAIT_SECONDS)


def _http_get_json(url: str, timeout: int) -> Optional[dict]:
    headers = {"User-Agent": _USER_AGENT}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    request = urllib.request.Request(url, headers=headers)

    for attempt in range(_HTTP_RETRIES + 1):
        _throttle()
        _S2_STATS["requests"] += 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                _speed_up()
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 429 = 限流。按 Retry-After / 指数退避重试；其余 4xx/5xx 直接放弃
            if exc.code == 429:
                _S2_STATS["throttled"] += 1
                _slow_down()
                if attempt < _HTTP_RETRIES:
                    _S2_STATS["retries"] += 1
                    wait = _retry_wait(exc, attempt)
                    logger.warning(
                        "Semantic Scholar 限流（429），%.1fs 后重试（第 %d/%d 次）",
                        wait,
                        attempt + 1,
                        _HTTP_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                logger.warning(
                    "Semantic Scholar 连续 429，放弃本次请求（%s）。"
                    "如果反复出现：确认没有别的进程/脚本在用同一个 key，"
                    "或把 SEMANTIC_SCHOLAR_MIN_INTERVAL 调大。",
                    url,
                )
                return None
            logger.warning("Semantic Scholar 返回 HTTP %s：%s", exc.code, url)
            return None
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            logger.warning("Semantic Scholar 请求失败：%s（%s）", exc, url)
            return None
    return None

class CitationGraph:
    """论文引用图（networkx 有向图 + JSON 持久化）。

    用 networkx 而不是自己写邻接表：BFS、最短路径、子图都是现成的，
    答辩时每一行都能对应到教科书上的图算法。
    """

    def __init__(self, path: str = GRAPH_PATH) -> None:
        self.path = Path(path)
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("引用图谱需要 networkx：pip install networkx") from exc
        self.nx = nx
        self.graph = nx.DiGraph()
        self._load()

    # ------------------------------------------------------------ 持久化
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("引用图谱文件损坏，忽略：%s", exc)
            return

        for node in payload.get("nodes", []):
            paper_id = node.get("id")
            if not paper_id:
                continue
            self.graph.add_node(
                paper_id,
                title=node.get("title", ""),
                year=node.get("year", ""),
                authors=node.get("authors", ""),
                venue=node.get("venue", ""),
                abstract=node.get("abstract", ""),
                in_library=bool(node.get("in_library")),
                source_key=node.get("source_key", ""),
            )
        for edge in payload.get("edges", []):
            source, target = edge.get("source"), edge.get("target")
            if source and target:
                self.graph.add_edge(source, target)
        logger.info(
            "引用图谱已加载：%d 节点，%d 边",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nodes = [
            {
                "id": paper_id,
                "title": data.get("title", ""),
                "year": data.get("year", ""),
                "authors": data.get("authors", ""),
                "venue": data.get("venue", ""),
                "abstract": data.get("abstract", ""),
                "in_library": bool(data.get("in_library")),
                "source_key": data.get("source_key", ""),
            }
            for paper_id, data in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": source, "target": target}
            for source, target in self.graph.edges()
        ]
        self.path.write_text(
            json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------ 建图
    def has_paper(self, paper_id: str) -> bool:
        return self.graph.has_node(paper_id)

    def has_edge(self, source: str, target: str) -> bool:
        """source 是否直接引用了 target。"""
        return self.graph.has_edge(source, target)

    def clear(self) -> None:
        """清空整张图（清空当前知识库时一并重置）。"""
        self.graph.clear()
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError as exc:  # noqa: BLE001
            logger.warning("删除引用图文件失败：%s", exc)

    def add_paper(self, node: PaperNode) -> None:
        self.graph.add_node(
            node.paper_id,
            title=node.title,
            year=node.year,
            authors=node.authors,
            venue=node.venue,
            abstract=node.abstract,
            in_library=node.in_library,
            source_key=node.source_key,
        )

    def mark_in_library(self, paper_id: str, source_key: str = "") -> None:
        if self.graph.has_node(paper_id):
            self.graph.nodes[paper_id]["in_library"] = True
            if source_key:
                self.graph.nodes[paper_id]["source_key"] = source_key

    def mark_out_of_library(self, source_key: str) -> int:
        """文献从库里删掉后，把对应节点的「在库」标记去掉（保留引用边）。"""
        changed = 0
        for _node_id, data in self.graph.nodes(data=True):
            if data.get("source_key") == source_key and data.get("in_library"):
                data["in_library"] = False
                changed += 1
        if changed:
            self.save()
        return changed

    def sync_in_library(self, source_keys) -> int:
        """把「在库」标记和向量库的实际内容对齐。

        典型场景：直接清空过向量库、或某篇文献被单独删除，图谱里还留着
        `in_library=True` 的残留节点，导致「文献库 5 篇 / 图谱在库 6 篇」
        这种对不上的数字。
        """
        keys = {str(key) for key in (source_keys or []) if key}
        changed = 0
        for _node_id, data in self.graph.nodes(data=True):
            key = str(data.get("source_key") or "")
            if not key:
                continue
            should_be = key in keys
            if bool(data.get("in_library")) != should_be:
                data["in_library"] = should_be
                changed += 1
        if changed:
            self.save()
            logger.info("引用图谱「在库」标记已与向量库对齐：%d 个节点更新", changed)
        return changed

    def in_library_nodes(self) -> List[PaperNode]:
        """库里所有论文在图中的节点（按标题排序）。"""
        nodes = [
            node
            for node_id in self.graph.nodes
            if (node := self.node(node_id)) is not None and node.in_library
        ]
        return sorted(nodes, key=lambda item: item.title.lower())

    def find_by_title(self, title: str) -> str:
        """按标题（或 source_key）模糊匹配节点 id，找不到返回空串。"""
        needle = (title or "").strip().lower()
        if not needle:
            return ""
        if self.graph.has_node(title):
            return title
        for node_id, data in self.graph.nodes(data=True):
            haystack = str(data.get("title") or "").lower()
            if haystack and (needle in haystack or haystack in needle):
                return node_id
            if needle in str(data.get("source_key") or "").lower():
                return node_id
        return ""

    def degrees(self, paper_id: str) -> Dict[str, int]:
        """出边（它引用了多少）/ 入边（被多少引用）。"""
        if not self.graph.has_node(paper_id):
            return {"references": 0, "citations": 0}
        return {
            "references": self.graph.out_degree(paper_id),
            "citations": self.graph.in_degree(paper_id),
        }

    def ego_snapshot(
        self, paper_id: str, hops: int = 2, max_nodes: int = 40, direction: str = "both"
    ) -> Dict[str, object]:
        """以某篇论文为中心的邻域子图，供前端画图。

        direction: both（双向）/ out（只看它引用了谁）/ in（只看谁引用了它）
        返回纯 dict，前端不需要认识 networkx / PaperNode。
        """
        if not self.graph.has_node(paper_id):
            return {
                "center": paper_id,
                "nodes": [],
                "edges": [],
                "truncated": False,
                "total_nodes": 0,
                "total_edges": 0,
            }

        hops = max(1, min(int(hops or 1), 3))
        direction = direction if direction in {"both", "out", "in"} else "both"

        def _neighbors(node_id: str) -> List[str]:
            if direction == "out":
                return list(self.graph.successors(node_id))
            if direction == "in":
                return list(self.graph.predecessors(node_id))
            return list(self.graph.successors(node_id)) + list(
                self.graph.predecessors(node_id)
            )

        seen = {paper_id}
        frontier = {paper_id}
        for _depth in range(hops):
            next_frontier = set()
            for node_id in frontier:
                for neighbor in _neighbors(node_id):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        # 超出上限时优先保留中心节点和在库节点，其余按引用度数排序截断
        ordered = sorted(
            seen,
            key=lambda node_id: (
                node_id != paper_id,
                not bool(self.graph.nodes[node_id].get("in_library")),
                -self.graph.degree(node_id),
            ),
        )
        kept = set(ordered[: max(2, max_nodes)])
        truncated = len(seen) > len(kept)

        nodes = []
        for node_id in kept:
            data = self.graph.nodes[node_id]
            nodes.append(
                {
                    "id": node_id,
                    "title": str(data.get("title") or node_id),
                    "year": str(data.get("year") or ""),
                    "in_library": bool(data.get("in_library")),
                    "local_ref": str(data.get("venue") or "") == "本地参考文献",
                    "is_center": node_id == paper_id,
                    "references": self.graph.out_degree(node_id),
                    "citations": self.graph.in_degree(node_id),
                }
            )
        nodes.sort(key=lambda item: (not item["is_center"], not item["in_library"]))

        edges = [
            {"source": source, "target": target}
            for source, target in self.graph.edges()
            if source in kept and target in kept
        ]
        return {
            "center": paper_id,
            "nodes": nodes,
            "edges": edges,
            "truncated": truncated,
            "total_nodes": len(seen),
            "total_edges": self.graph.number_of_edges(),
        }

    def add_edge(self, source: str, target: str) -> None:
        if source and target and source != target:
            self.graph.add_edge(source, target)

    def node(self, paper_id: str) -> Optional[PaperNode]:
        if not self.graph.has_node(paper_id):
            return None
        data = self.graph.nodes[paper_id]
        return PaperNode(
            paper_id=paper_id,
            title=data.get("title", ""),
            year=data.get("year", ""),
            authors=data.get("authors", ""),
            venue=data.get("venue", ""),
            abstract=data.get("abstract", ""),
            in_library=bool(data.get("in_library")),
            source_key=data.get("source_key", ""),
        )
    # ------------------------------------------------------------ 多跳查询
    def references(self, paper_id: str) -> List[PaperNode]:
        """它引用了谁（沿出边）。"""
        if not self.graph.has_node(paper_id):
            return []
        return [
            self.node(target)
            for target in self.graph.successors(paper_id)
            if self.node(target)
        ]

    def citations(self, paper_id: str) -> List[PaperNode]:
        """谁引用了它（沿入边）。"""
        if not self.graph.has_node(paper_id):
            return []
        return [
            self.node(source)
            for source in self.graph.predecessors(paper_id)
            if self.node(source)
        ]

    def trace_lineage(
        self, paper_id: str, hops: int = GRAPH_MAX_HOPS
    ) -> Dict[str, List[PaperNode]]:
        """多跳追溯：BFS 展开前驱（它引用的）与后继（引用它的）。

        ancestors = 它引用的、以及被引用者引用的…（向前追溯）
        descendants = 引用它的、以及引用它的引用者…（向后追溯）
        """
        if not self.graph.has_node(paper_id):
            return {"ancestors": [], "descendants": []}

        hops = max(1, min(hops, 4))
        ancestors: List[PaperNode] = []
        descendants: List[PaperNode] = []

        seen = {paper_id}
        frontier = {paper_id}
        for _depth in range(hops):
            next_frontier = set()
            for node_id in frontier:
                for target in self.graph.successors(node_id):
                    if target in seen:
                        continue
                    seen.add(target)
                    next_frontier.add(target)
                    node = self.node(target)
                    if node:
                        ancestors.append(node)
            frontier = next_frontier
            if not frontier:
                break

        seen = {paper_id}
        frontier = {paper_id}
        for _depth in range(hops):
            next_frontier = set()
            for node_id in frontier:
                for source in self.graph.predecessors(node_id):
                    if source in seen:
                        continue
                    seen.add(source)
                    next_frontier.add(source)
                    node = self.node(source)
                    if node:
                        descendants.append(node)
            frontier = next_frontier
            if not frontier:
                break

        return {"ancestors": ancestors, "descendants": descendants}

    def shortest_path(self, source: str, target: str) -> List[PaperNode]:
        """两篇论文在引用网络上的最短路径（用于讲演进链）。"""
        if not (self.graph.has_node(source) and self.graph.has_node(target)):
            return []
        try:
            path = self.nx.shortest_path(self.graph.to_undirected(), source, target)
        except Exception:  # noqa: BLE001 - 不连通
            return []
        return [self.node(node_id) for node_id in path if self.node(node_id)]

    def stats(self) -> Dict[str, int]:
        return {
            "papers": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "in_library": sum(
                1 for _, data in self.graph.nodes(data=True) if data.get("in_library")
            ),
        }

_S2_FIELDS = "title,year,authors,venue,abstract,externalIds,citationCount"


def _node_from_s2(item: dict) -> Optional[PaperNode]:
    """把 Semantic Scholar 的一条 paper 数据变成节点。"""
    if not isinstance(item, dict):
        return None
    paper_id = item.get("paperId") or ""
    if not paper_id:
        return None
    authors = item.get("authors") or []
    names = ", ".join(
        author.get("name", "") for author in authors if isinstance(author, dict)
    ).strip(", ")
    return PaperNode(
        paper_id=paper_id,
        title=(item.get("title") or "").strip(),
        year=str(item.get("year") or ""),
        authors=names,
        venue=(item.get("venue") or "").strip(),
        abstract=(item.get("abstract") or "").strip(),
    )


def _cache_key(arxiv_id: str = "", title: str = "") -> str:
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip().lower()}"
    return "title:" + re.sub(r"\s+", " ", (title or "").strip().lower())[:200]


def _resolve_paper(
    arxiv_id: str = "", title: str = "", timeout: int = GRAPH_HTTP_TIMEOUT
) -> "tuple[str, Optional[dict]]":
    """把 arXiv ID / 标题解析成 S2 paperId，并尽量把元数据一起带回来。

    返回 (paperId, payload)：
    - payload 非空时可直接当元数据用，省掉一次 `/paper/{id}` 请求；
    - 命中进程内缓存时 payload 为 None（元数据从图里已有节点补）。
    """
    key = _cache_key(arxiv_id, title)
    if key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[key], None

    if arxiv_id:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/arXiv:{arxiv_id}?fields={_S2_FIELDS}"
        payload = _http_get_json(url, timeout)
        if payload and payload.get("paperId"):
            _RESOLVE_CACHE[key] = payload["paperId"]
            return payload["paperId"], payload

    if title:
        query = urllib.parse.quote(title[:200])
        url = (
            f"{SEMANTIC_SCHOLAR_API}/paper/search?query={query}&limit=1"
            f"&fields={_S2_FIELDS}"
        )
        payload = _http_get_json(url, timeout)
        if payload:
            data = payload.get("data") or []
            if data and data[0].get("paperId"):
                item = dict(data[0])
                _RESOLVE_CACHE[key] = item["paperId"]
                return item["paperId"], item
    return "", None


def _resolve_paper_id(arxiv_id: str = "", title: str = "", timeout: int = GRAPH_HTTP_TIMEOUT) -> str:
    """兼容旧接口：只返回 paperId。"""
    paper_id, _payload = _resolve_paper(arxiv_id=arxiv_id, title=title, timeout=timeout)
    return paper_id

class CitationGraphBuilder:
    """从 Semantic Scholar 抓引用关系，把图建全。

    为什么用 Semantic Scholar 而不是解析 PDF 的参考文献：
    PDF 只能告诉你「它引用了谁」（出边），拿不到「谁引用了它」（入边）；
    而 S2 两边都有，且是结构化的。免费额度足够做一个演示用的图。
    """

    def __init__(self, graph: CitationGraph) -> None:
        self.graph = graph

    def _known_node_id(self, source_key: str, title: str) -> str:
        """如果这篇论文以前已经解析成 S2 paperId，直接复用，省一次解析请求。

        只有真正的 S2 paperId 才复用它；`pdf:xxx` / `local:xxx` 这类本地
        标识说明当时没解析成功，仍要重新解析。
        """
        if source_key:
            for node_id, data in self.graph.graph.nodes(data=True):
                if data.get("source_key") != source_key:
                    continue
                if node_id and not node_id.startswith(("pdf:", "arxiv:", "local:")):
                    return node_id
        needle = (title or "").strip().lower()[:80]
        if needle:
            for node_id, data in self.graph.graph.nodes(data=True):
                if node_id.startswith(("pdf:", "local:")):
                    continue
                node_title = str(data.get("title") or "").strip().lower()
                if node_title and (needle in node_title or node_title in needle):
                    if data.get("source_key") in ("", source_key):
                        return node_id
        return ""

    def _add_paper(self, node: PaperNode) -> None:
        existing = self.graph.node(node.paper_id)
        if existing is None:
            self.graph.add_paper(node)
            return
        # 已有节点只补空字段，避免抓来的信息覆盖本地更准的元数据
        data = self.graph.graph.nodes[node.paper_id]
        for field_name in ("title", "year", "authors", "venue", "abstract"):
            if not data.get(field_name) and getattr(node, field_name):
                data[field_name] = getattr(node, field_name)

    def ingest(
        self,
        arxiv_id: str = "",
        title: str = "",
        source_key: str = "",
        references_text: str = "",
    ) -> Dict[str, int]:
        """抓一篇论文的 references + citations 并入图。

        即使 Semantic Scholar 不可达（无网络 / 被限流），也至少把「本地这篇论文」
        作为孤节点登记进图，这样：
        - 侧栏能显示「共 N 篇文献」，而不是误导性的 0；
        - 多跳工具能明确回答「这篇论文在库里，但引用关系未抓取」，而不是
          「图里找不到这篇论文」。

        `references_text` 是 PDF 里解析出的「参考文献」章节原文。S2 没给出引用边时
        （新论文常见：referenceCount 有值但 references 为空），用它兜底建有向边，
        保证「它引用了谁」这一侧永远有数据。
        """
        seed_id = self._known_node_id(source_key, title)
        seed_payload: Optional[dict] = None
        if not seed_id:
            seed_id, seed_payload = _resolve_paper(arxiv_id=arxiv_id, title=title)
        if not seed_id:
            # 退化：用本地标识建孤节点，保留标题/年份/来源，等网络恢复可补边
            local_id = source_key or f"local:{title.strip()[:80] or 'untitled'}"
            if not self.graph.has_paper(local_id):
                self.graph.add_paper(
                    PaperNode(
                        paper_id=local_id,
                        title=title,
                        in_library=True,
                        source_key=source_key,
                    )
                )
            else:
                self.graph.mark_in_library(local_id, source_key=source_key)
            # 即使 S2 上没有这篇论文，「它引用了谁」也能从 PDF 参考文献里拿到
            local_refs = (
                self._add_local_references(local_id, references_text)
                if references_text.strip()
                else 0
            )
            self.graph.save()
            logger.warning(
                "Semantic Scholar 未能定位论文（%s / %s），已登记为本地节点 %s，"
                "并从 PDF 补出 %d 条引用边",
                arxiv_id or "-",
                title or "-",
                local_id,
                local_refs,
            )
            return {
                "references": local_refs,
                "citations": 0,
                "local_only": 1,
                "local_references": local_refs,
            }

        # 如果之前失败时留下过同名本地节点，把它的边迁移到正式的 S2 节点上
        self._merge_local_node(seed_id, source_key, title)

        known = self.graph.node(seed_id)
        if seed_payload is None and (known is None or not known.title):
            # 图里还没有这篇的元数据，才值得再花一次请求
            seed_payload = _http_get_json(
                f"{SEMANTIC_SCHOLAR_API}/paper/{seed_id}?fields={_S2_FIELDS}",
                GRAPH_HTTP_TIMEOUT,
            )
        seed_node = _node_from_s2(seed_payload or {"paperId": seed_id})
        if seed_node is None:
            seed_node = PaperNode(paper_id=seed_id)
        self._add_paper(seed_node)
        if source_key:
            self.graph.graph.nodes[seed_id]["in_library"] = True
            self.graph.graph.nodes[seed_id]["source_key"] = source_key

        # 本地缩略名（pdf:xxx / arxiv:xxx）也登记成节点，方便前端按来源定位
        if source_key and source_key != seed_id:
            self._add_paper(
                PaperNode(
                    paper_id=seed_id,
                    title=title,
                    in_library=True,
                    source_key=source_key,
                )
            )

        references = self._fetch_edges(seed_id, "references", outbound=True)
        citations = self._fetch_edges(seed_id, "citations", outbound=False)

        local_refs = 0
        if references == 0 and references_text.strip():
            # S2 常常只给元数据不给引用列表（新论文尤其明显），用 PDF 里的
            # 参考文献兜底，保证「它引用了谁」这一侧不是空的。
            local_refs = self._add_local_references(seed_id, references_text)

        self.graph.save()
        logger.info(
            "引用图更新 %s：references=%d（本地兜底 %d） citations=%d",
            seed_id,
            references,
            local_refs,
            citations,
        )
        return {
            "references": references + local_refs,
            "citations": citations,
            "local_references": local_refs,
        }

    def _add_local_references(self, seed_id: str, references_text: str) -> int:
        """把 PDF 参考文献章节解析成「seed → 引文」的边（无 S2 数据时兜底）。"""
        entries = _split_reference_entries(references_text)
        added = 0
        for entry in entries:
            title = entry[:240]
            node_id = "ref:" + hashlib.md5(entry.encode("utf-8")).hexdigest()[:16]
            if not self.graph.has_paper(node_id):
                self.graph.add_paper(
                    PaperNode(
                        paper_id=node_id,
                        title=title,
                        year=_guess_year(entry),
                        venue="本地参考文献",
                    )
                )
            self.graph.add_edge(seed_id, node_id)
            added += 1
        if added:
            logger.info("由 PDF 参考文献补出 %d 条引用边（%s）", added, seed_id)
        return added

    def _merge_local_node(self, seed_id: str, source_key: str, title: str) -> None:
        """把此前 S2 不可用时登记的本地孤节点合并到正式节点。

        否则同一篇论文会以「pdf:xxxx」和 S2 paperId 两个身份并存。
        """
        local_id = ""
        if source_key and source_key != seed_id and self.graph.has_paper(source_key):
            local_id = source_key
        elif title:
            needle = title.strip().lower()[:80]
            for node_id, data in self.graph.graph.nodes(data=True):
                if node_id == seed_id:
                    continue
                node_title = str(data.get("title") or "").strip().lower()
                if node_title and needle and (needle in node_title or node_title in needle):
                    local_id = node_id
                    break

        if not local_id:
            return

        # 迁移边：既保留 local -> x，也保留 x -> local
        for target in list(self.graph.graph.successors(local_id)):
            if target != seed_id:
                self.graph.add_edge(seed_id, target)
        for source in list(self.graph.graph.predecessors(local_id)):
            if source != seed_id:
                self.graph.add_edge(source, seed_id)
        # 本地节点上的 in_library 标记迁移过去
        if self.graph.graph.nodes[local_id].get("in_library"):
            self.graph.graph.add_node(seed_id)
            self.graph.graph.nodes[seed_id]["in_library"] = True
        self.graph.graph.remove_node(local_id)
        logger.info("已把本地节点 %s 合并到 Semantic Scholar 节点 %s", local_id, seed_id)

    def _fetch_edges(self, seed_id: str, direction: str, outbound: bool) -> int:
        """抓一个方向的引用列表（一次请求拿 100 条，够演示用）。

        注意 Semantic Scholar 的 key 是 1 请求/秒、全端点共享，所以这里刻意
        只取一页：多翻一页就多占 1 秒额度，对演示没有额外价值。
        """
        url = (
            f"{SEMANTIC_SCHOLAR_API}/paper/{seed_id}/{direction}"
            f"?fields={_S2_FIELDS}&limit=100"
        )
        payload = _http_get_json(url, GRAPH_HTTP_TIMEOUT)
        if not payload:
            return 0

        count = 0
        for item in payload.get("data") or []:
            paper = item.get("citedPaper") or item.get("citingPaper") or {}
            node = _node_from_s2(paper)
            if node is None:
                continue
            self._add_paper(node)
            if outbound:
                self.graph.add_edge(seed_id, node.paper_id)
            else:
                self.graph.add_edge(node.paper_id, seed_id)
            count += 1
        return count
        return count