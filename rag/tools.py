"""多步工具链所用的工具集：把「检索」拆成可组合的原子能力。

单步 RAG 只能「检索一次 → 生成」，遇到下面这类问题就力不从心：

- 「我上传了哪些文献？」        → 需要列目录，不是语义检索；
- 「第 3 节具体怎么做的？」      → 需要按章节精读，而不是向量近邻；
- 「这两篇论文的方法有什么不同？」→ 需要分别检索再对比，多步推理。

为此提供四个工具，交给模型自己决定调用顺序：

1. `search_literature(query)`  混合检索（向量 + BM25 + RRF + rerank）
2. `list_papers()`             列出已入库文献（标题 / 章节数 / 片段数）
3. `read_section(source_key, section)` 按来源 + 章节读取正文
4. `compare_papers(topic)`     同一主题下按文献分组，便于横向对比

工具返回的都是纯文本（给模型读），同时把命中的 Document 记进 collector，
供最终的引用溯源使用 —— 这样多步链路的答案也能落到具体章节页码上。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

from rag.config import TOOL_MAX_OUTPUT_CHARS

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """一次工具调用的记录（用于前端展示与调试）。"""

    step: int
    name: str
    args: Dict[str, Any]
    output: str
    documents: int = 0

    @property
    def summary(self) -> str:
        arg_text = ", ".join(f"{key}={value}" for key, value in self.args.items())
        if self.name == "list_papers":
            # list_papers 只列目录，不往引用 collector 里加片段，
            # 所以 "命中 0 条" 是预期的，这里换成更直观的描述避免误解。
            return f"第 {self.step} 步 · {self.name}({arg_text}) → 已列出库内文献目录"
        return f"第 {self.step} 步 · {self.name}({arg_text}) → 命中 {self.documents} 条"


@dataclass
class SourceCollector:
    """收集工具链路里出现过的片段，最后统一编号做引用溯源。"""

    documents: List[Document] = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def add(self, documents: Sequence[Document]) -> None:
        for document in documents:
            key = document.id or f"{document.metadata.get('source_key')}|{document.page_content[:80]}"
            if key in self._seen:
                continue
            self._seen.add(key)
            self.documents.append(document)

    def top(self, limit: int) -> List[Document]:
        return self.documents[:limit]

    def __len__(self) -> int:
        return len(self.documents)


def _truncate(text: str) -> str:
    if len(text) <= TOOL_MAX_OUTPUT_CHARS:
        return text
    return text[:TOOL_MAX_OUTPUT_CHARS] + "\n…（输出过长已截断）"


def _format_hits(documents: Sequence[Document], start: int = 1) -> str:
    if not documents:
        return "没有检索到相关片段。"
    blocks: List[str] = []
    for offset, document in enumerate(documents):
        meta = document.metadata or {}
        header = f"[{start + offset}]"
        if meta.get("title"):
            header += f" 《{meta['title']}》"
        if meta.get("section"):
            header += f" · {meta['section']}"
        if meta.get("page"):
            header += f" · p.{meta['page']}"
        header += f" · 来源 {meta.get('source_key') or meta.get('source') or '未知'}"
        blocks.append(f"{header}\n{(document.page_content or '').strip()}")
    return "\n\n".join(blocks)


class LiteratureTools:
    """基于知识库的一组工具，可被模型按需调用。"""

    def __init__(
        self,
        knowledge,
        citation_graph=None,
        extraction_store=None,
        figure_lister=None,
        figure_describer=None,
    ) -> None:
        self.knowledge = knowledge
        self.collector = SourceCollector()
        self.calls: List[ToolCall] = []
        self.citation_graph = citation_graph
        self.extraction_store = extraction_store
        # 多模态：列出 PDF 图表 / 用视觉模型描述图表（由 ChatSession 注入）
        self.figure_lister = figure_lister
        self.figure_describer = figure_describer
        # 最近一次 compare_methods 生成的表格，供前端直接展示
        self.last_table: str = ""

    # ------------------------------------------------------------ 工具实现
    def search_literature(self, query: str) -> str:
        documents = self.knowledge.search(query)
        self.collector.add(documents)
        return _truncate(_format_hits(documents))

    def list_papers(self) -> str:
        try:
            documents = self.knowledge.vector_store.all_documents()
        except Exception as exc:  # noqa: BLE001
            return f"读取文献目录失败：{exc}"
        if not documents:
            return "知识库里还没有任何文献。"

        grouped: Dict[str, Dict[str, Any]] = {}
        for document in documents:
            meta = document.metadata or {}
            key = str(meta.get("source_key") or meta.get("source") or "未知")
            entry = grouped.setdefault(
                key,
                {
                    "title": meta.get("title") or meta.get("source") or key,
                    "year": meta.get("year") or "",
                    "authors": meta.get("authors") or "",
                    "sections": set(),
                    "chunks": 0,
                },
            )
            entry["sections"].add(str(meta.get("section") or "-"))
            entry["chunks"] += 1

        lines = [f"共 {len(grouped)} 篇文献："]
        for index, (key, entry) in enumerate(grouped.items(), start=1):
            header = f"{index}. 《{entry['title']}》"
            if entry["year"]:
                header += f"（{entry['year']}）"
            header += f" · 来源 {key} · {entry['chunks']} 个片段"
            lines.append(header)
            sections = [name for name in sorted(entry["sections"]) if name and name != "-"]
            if sections:
                lines.append(f"   章节：{'、'.join(sections)}")
        return _truncate("\n".join(lines))

    def read_section(self, source_key: str, section: str = "") -> str:
        """按来源（和可选的章节名）读取正文。"""
        try:
            documents = self.knowledge.vector_store.all_documents()
        except Exception as exc:  # noqa: BLE001
            return f"读取向量库失败：{exc}"

        needle = (source_key or "").strip().lower()
        section_needle = (section or "").strip().lower()
        matched: List[Document] = []
        for document in documents:
            meta = document.metadata or {}
            haystack = " ".join(
                str(meta.get(field) or "")
                for field in ("source_key", "source", "title")
            ).lower()
            if needle and needle not in haystack:
                continue
            if section_needle and section_needle not in str(meta.get("section") or "").lower():
                continue
            matched.append(document)

        if not matched:
            available: List[str] = []
            for document in documents:
                meta = document.metadata or {}
                key = str(meta.get("source_key") or "")
                name = str(meta.get("section") or "")
                entry = f"{key} / {name}"
                if entry not in available:
                    available.append(entry)
            hint = "；".join(available[:12]) or "（无）"
            return f"没有找到匹配的章节。可用组合（来源 / 章节）：{hint}"

        matched.sort(key=lambda item: int((item.metadata or {}).get("chunk_index") or 0))
        self.collector.add(matched)
        return _truncate(_format_hits(matched))

    def compare_papers(self, topic: str) -> str:
        """围绕一个主题分别检索，按文献分组输出，便于横向对比。"""
        documents = self.knowledge.search(topic, num_documents=max(4, self.knowledge.num_documents))
        if not documents:
            return f"没有检索到与「{topic}」相关的片段，无法对比。"
        self.collector.add(documents)

        grouped: Dict[str, List[Document]] = {}
        for document in documents:
            meta = document.metadata or {}
            key = str(meta.get("title") or meta.get("source_key") or "未知文献")
            grouped.setdefault(key, []).append(document)

        if len(grouped) == 1:
            only = next(iter(grouped))
            return _truncate(
                f"只检索到 1 篇文献《{only}》的相关片段，无法横向对比。\n\n"
                + _format_hits(grouped[only])
            )

        blocks: List[str] = [f"围绕「{topic}」检索到 {len(grouped)} 篇文献："]
        for title, items in grouped.items():
            blocks.append(f"\n== 《{title}》 ==")
            for document in items:
                meta = document.metadata or {}
                where = " / ".join(
                    part
                    for part in (str(meta.get("section") or ""), f"p.{meta.get('page')}" if meta.get("page") else "")
                    if part
                )
                blocks.append(f"- [{where}] {(document.page_content or '').strip()[:400]}")

        relationships = self._pairwise_relationships(list(grouped))
        if relationships:
            blocks.append("\n== 这几篇之间的引用关系（来自引用图谱）==")
            blocks.append(relationships)
        return _truncate("\n".join(blocks))

    def _pairwise_relationships(self, titles: Sequence[str]) -> str:
        """对比时顺带给出这几篇之间的直接引用关系。"""
        if not self.citation_graph or len(titles) < 2:
            return ""
        resolved = [
            (title, self._resolve_paper(title)) for title in titles
        ]
        resolved = [(title, node_id) for title, node_id in resolved if node_id]
        lines: List[str] = []
        for index in range(len(resolved)):
            for other in range(index + 1, len(resolved)):
                left_title, left_id = resolved[index]
                right_title, right_id = resolved[other]
                if left_id == right_id:
                    continue
                if self.citation_graph.has_edge(left_id, right_id):
                    lines.append(f"- 《{left_title}》引用了《{right_title}》")
                elif self.citation_graph.has_edge(right_id, left_id):
                    lines.append(f"- 《{right_title}》引用了《{left_title}》")
                else:
                    lines.append(f"- 《{left_title}》与《{right_title}》之间无直接引用关系")
        if not lines:
            return ""
        lines.append(
            "（说明：这是引用网络中的事实关系，只表示文献之间存在引用，"
            "不等于方法上确有借鉴。）"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------ 引用图谱工具
    def _resolve_paper(self, paper: str) -> str:
        """把用户给的论文名/ID 映射成图里的节点 id。

        先直接当 id 用；不行再按标题模糊匹配节点。
        """
        if not self.citation_graph:
            return ""
        paper = (paper or "").strip()
        if not paper:
            return ""
        if self.citation_graph.has_paper(paper):
            return paper

        needle = paper.lower()
        for node_id, data in self.citation_graph.graph.nodes(data=True):
            title = str(data.get("title") or "").lower()
            if needle and (needle in title or title in needle):
                return node_id
            if needle and needle in str(data.get("source_key") or "").lower():
                return node_id
        return ""

    @staticmethod
    def _format_nodes(nodes: Sequence, empty: str) -> str:
        if not nodes:
            return empty
        lines: List[str] = []
        for index, node in enumerate(nodes, start=1):
            title = (node.title or node.paper_id).strip()
            if len(title) > 120:
                title = title[:120] + "…"
            header = f"{index}. 《{title}》"
            if node.year:
                header += f"（{node.year}）"
            if node.authors:
                header += f" · {node.authors[:60]}"
            if node.in_library:
                header += " · ✅已在库中"
            elif node.venue == "本地参考文献":
                header += " · 📄PDF 参考文献（未在 S2 建索引）"
            lines.append(header)
        return "\n".join(lines)

    def get_references(self, paper: str) -> str:
        """它引用了谁（沿引用边向前追溯）。"""
        if not self.citation_graph:
            return "引用图谱未启用。"
        node_id = self._resolve_paper(paper)
        if not node_id:
            return f"引用图谱里找不到论文「{paper}」。请先导入该论文的 arXiv 版本以构建引用关系。"
        refs = self.citation_graph.references(node_id)
        node = self.citation_graph.node(node_id)
        title = node.title if node else paper
        if not refs:
            return (
                f"《{title}》的引用关系尚未抓取到。\n"
                f"说明：该论文已在库中登记，但没有从 Semantic Scholar 取到参考文献"
                f"（可能是限流、无网络，或该文献未被 S2 收录）。"
                f"可配置 SEMANTIC_SCHOLAR_API_KEY 后重新导入以补全。"
            )
        body = f"《{title}》引用了以下 {len(refs)} 篇论文：\n" + self._format_nodes(
            refs, "（它没有记录到任何被引论文）"
        )
        if any(node.venue == "本地参考文献" for node in refs):
            body += (
                "\n\n说明：这些条目来自该论文 PDF 的 References 章节，"
                "因为 Semantic Scholar 还没为它建立引用索引（新论文常见）。"
            )
        return body

    def get_citations(self, paper: str) -> str:
        """谁引用了它（沿引用边向后追溯）。"""
        if not self.citation_graph:
            return "引用图谱未启用。"
        node_id = self._resolve_paper(paper)
        if not node_id:
            return f"引用图谱里找不到论文「{paper}」。"
        node = self.citation_graph.node(node_id)
        cites = self.citation_graph.citations(node_id)
        title = node.title if node else paper
        if not cites:
            return (
                f"《{title}》的被引关系尚未抓取到。\n"
                f"说明：该论文已在库中登记，但没有从 Semantic Scholar 取到引用它的论文"
                f"（可能是限流、无网络，或还没有公开的引用记录）。"
                f"可配置 SEMANTIC_SCHOLAR_API_KEY 后重新导入以补全。"
            )
        return f"以下 {len(cites)} 篇论文引用了《{title}》：\n" + self._format_nodes(
            cites, "（还没有论文引用它，或引用数据未抓取）"
        )

    def trace_lineage(self, paper: str, hops: int = 2) -> str:
        """多跳追溯：它的前驱（引用链上游）与后继（下游改进）。"""
        if not self.citation_graph:
            return "引用图谱未启用。"
        node_id = self._resolve_paper(paper)
        if not node_id:
            return f"引用图谱里找不到论文「{paper}」。"
        node = self.citation_graph.node(node_id)
        title = node.title if node else paper
        lineage = self.citation_graph.trace_lineage(node_id, hops=hops)
        ancestors = lineage.get("ancestors", [])
        descendants = lineage.get("descendants", [])

        if not ancestors and not descendants:
            return (
                f"《{title}》暂无可用引用脉络。\n"
                f"说明：该论文已在库中登记，但引用关系未抓取到"
                f"（Semantic Scholar 限流 / 无网络 / 未被收录）。"
                f"配置 SEMANTIC_SCHOLAR_API_KEY 后重新导入可补全多跳链路。"
            )

        blocks = [f"《{title}》的引用脉络（{hops} 跳）："]
        blocks.append(f"\n== 它继承的工作（向前追溯，{len(ancestors)} 篇）==")
        blocks.append(self._format_nodes(ancestors, "（无）"))
        blocks.append(f"\n== 后续引用/改进它的工作（向后追溯，{len(descendants)} 篇）==")
        blocks.append(self._format_nodes(descendants, "（无）"))
        if ancestors and descendants:
            blocks.append(
                "\n提示：可以据此梳理技术演进链 —— "
                "从它继承的方法，到它，再到改进它的后续工作。"
            )
        return _truncate("\n".join(blocks))

    # ------------------------------------------------------------ 方法对比工具
    def compare_methods(self, papers: str = "") -> str:
        """从结构化抽取表生成 Markdown 方法对比表。

        `papers` 为空则对比全部已抽取的论文；否则按逗号分隔的关键词筛选。
        """
        if not self.extraction_store:
            return "结构化抽取未启用。"
        profiles = self.extraction_store.all()
        if not profiles:
            return "还没有任何论文做过结构化抽取，无法生成对比表。"

        if papers.strip():
            keywords = [item.strip() for item in re.split(r"[,，、;；]", papers) if item.strip()]
            selected = []
            for keyword in keywords:
                selected.extend(self.extraction_store.find(keyword))
            # 去重并保持顺序
            seen = set()
            picked = []
            for profile in selected:
                if profile.source_key in seen:
                    continue
                seen.add(profile.source_key)
                picked.append(profile)
            if picked:
                profiles = picked

        from rag.extraction import render_comparison_table

        table = render_comparison_table(profiles)
        self.last_table = table
        names = "、".join(p.method_name or p.title for p in profiles)
        return f"已生成 {len(profiles)} 种方法（{names}）的对比表：\n\n{table}"

    # ------------------------------------------------------------ 多模态图表工具
    def list_figures(self, source_key: str) -> str:
        """列出某篇论文 PDF 中抽取到的图表（页码 / 序号 / 尺寸）。"""
        if not self.figure_lister:
            return "图表提取未启用或当前文献没有保留原始 PDF。"
        try:
            figures = self.figure_lister((source_key or "").strip())
        except Exception as exc:  # noqa: BLE001
            return f"读取图表失败：{exc}"
        if not figures:
            return (
                f"《{source_key}》没有可用的图表：可能没有保留原始 PDF，"
                "或 PDF 里的图片过小（图标 / logo 已被过滤）。"
            )
        lines = [f"《{source_key}》共提取到 {len(figures)} 张图表："]
        for figure in figures:
            lines.append(
                f"- 第 {figure.get('page', '?')} 页 · 图 {figure.get('index', '?')} · "
                f"{figure.get('width', '?')}×{figure.get('height', '?')} · {figure.get('ext', 'png')}"
            )
        lines.append("可用 describe_figure 工具让视觉模型解释其中一张。")
        return _truncate("\n".join(lines))

    def describe_figure(
        self,
        source_key: str,
        page: int = 0,
        figure_index: int = 0,
        question: str = "",
    ) -> str:
        """让视觉模型解释某篇论文的一张图表（按页码或图表序号定位）。"""
        if not self.figure_lister or not self.figure_describer:
            return "图表理解未启用：请配置视觉模型（RAG_VISION_MODEL）并保留原始 PDF。"
        try:
            figures = self.figure_lister((source_key or "").strip())
        except Exception as exc:  # noqa: BLE001
            return f"读取图表失败：{exc}"
        if not figures:
            return f"《{source_key}》没有可用的图表。"

        target = None
        if page:
            target = next((item for item in figures if int(item.get("page") or 0) == int(page)), None)
        if target is None and figure_index:
            target = next(
                (item for item in figures if int(item.get("index") or 0) == int(figure_index)),
                None,
            )
        if target is None:
            target = figures[0]

        try:
            description = self.figure_describer(
                (source_key or "").strip(), target, question
            )
        except Exception as exc:  # noqa: BLE001
            return f"图表描述失败：{exc}"
        header = (
            f"《{source_key}》第 {target.get('page', '?')} 页 · 图 "
            f"{target.get('index', '?')}（{target.get('width', '?')}×{target.get('height', '?')}）：\n\n"
        )
        return _truncate(header + (description or "（没有返回描述）"))

    # ------------------------------------------------------------ 工具定义
    def as_langchain_tools(self) -> List[StructuredTool]:
        """转成给模型绑定用的工具列表。"""
        tools = [
            # 把普通 Python 函数包装成工具对象，并干三件事——① 包装成 BaseTool ② 记录 name/description ③ 从函数签名自动推断参数 schema。
            StructuredTool.from_function(
                func=self.search_literature,
                name="search_literature",
                description=(
                    "在已上传的文献里做语义 + 关键词混合检索，返回最相关的片段。"
                    "适合回答具体问题。参数 query 是检索用的自然语言问句。"
                ),
            ),
            StructuredTool.from_function(
                func=self.list_papers,
                name="list_papers",
                description=(
                    "列出当前知识库里所有文献的标题、年份、片段数和章节名。"
                    "当用户问「我上传了哪些论文」「有几篇文献」时使用。"
                ),
            ),
            StructuredTool.from_function(
                func=self.read_section,
                name="read_section",
                description=(
                    "按来源标识（source_key，如 pdf:1a2b3c4d5e6f 或 arxiv:2401.00001）"
                    "读取某篇文献的正文，可再用 section 参数限定章节名（如 Method）。"
                    "当用户要求「读一下第几节」「把某篇论文的某章完整说明」时使用。"
                ),
            ),
            StructuredTool.from_function(
                func=self.compare_papers,
                name="compare_papers",
                description=(
                    "围绕一个主题分别检索并按文献分组返回，用于横向对比多篇论文。"
                    "参数 topic 是对比的主题，例如「损失函数设计」。"
                    "如果这几篇论文之间存在直接引用关系，结果末尾会自动附上"
                    "「引用关系」一节（谁引用了谁）。"
                ),
            ),
        ]

        if self.citation_graph is not None:
            tools.extend(
                [
                    StructuredTool.from_function(
                        func=self.get_references,
                        name="get_references",
                        description=(
                            "查询某篇论文引用了哪些论文（沿引用网络向前追溯）。"
                            "参数 paper 是论文标题或图节点 id。"
                            "当用户问「它基于哪些工作」「它引用了什么」时使用。"
                        ),
                    ),
                    StructuredTool.from_function(
                        func=self.get_citations,
                        name="get_citations",
                        description=(
                            "查询哪些论文引用了某篇论文（沿引用网络向后追溯）。"
                            "当用户问「谁改进了它」「有哪些后续工作」时使用。"
                        ),
                    ),
                    StructuredTool.from_function(
                        func=self.trace_lineage,
                        name="trace_lineage",
                        description=(
                            "多跳追溯某篇论文的引用脉络：继承的工作与后续改进。"
                            "参数 paper 是论文标题或 id，hops 是跳数（默认 2）。"
                            "当用户问「这条技术路线是怎么演进的」时使用。"
                        ),
                    ),
                ]
            )

        if self.extraction_store is not None:
            tools.append(
                StructuredTool.from_function(
                    func=self.compare_methods,
                    name="compare_methods",
                    description=(
                        "从结构化抽取结果生成 Markdown 方法对比表（方法名/核心思想/"
                        "数据集/指标/参数量/优点/局限）。"
                        "参数 papers 用逗号分隔论文名可筛选，留空则对比全部。"
                        "当用户问「对比这些方法」「做一张对比表」时使用。"
                    ),
                )
            )

        if self.figure_lister is not None:
            tools.append(
                StructuredTool.from_function(
                    func=self.list_figures,
                    name="list_figures",
                    description=(
                        "列出某篇论文 PDF 中抽取到的图表（页码 / 序号 / 尺寸）。"
                        "参数 source_key 是文献来源标识。"
                        "当用户问「这篇论文有哪些图」「图表在第几页」时使用。"
                    ),
                )
            )

        if self.figure_lister is not None and self.figure_describer is not None:
            tools.append(
                StructuredTool.from_function(
                    func=self.describe_figure,
                    name="describe_figure",
                    description=(
                        "用视觉模型解释某篇论文的一张图表（结构 / 趋势 / 关键数值）。"
                        "参数 source_key 是文献来源标识，page 是页码，"
                        "figure_index 是图表序号，question 是具体想问的角度。"
                        "当用户问「这张图说明了什么」「解释图 2」时使用。"
                    ),
                )
            )

        return tools

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        """按名字执行工具（工具链循环里调用）。"""
        handlers: Dict[str, Callable[..., str]] = {
            "search_literature": self.search_literature,
            "list_papers": self.list_papers,
            "read_section": self.read_section,
            "compare_papers": self.compare_papers,
            "get_references": self.get_references,
            "get_citations": self.get_citations,
            "trace_lineage": self.trace_lineage,
            "compare_methods": self.compare_methods,
            "list_figures": self.list_figures,
            "describe_figure": self.describe_figure,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"未知工具：{name}。可用工具：{', '.join(handlers)}"
        try:
            return handler(**args)  # 函数执行后的结果
        except TypeError as exc:
            return f"工具 {name} 的参数不正确：{exc}"
        except Exception as exc:  # noqa: BLE001 - 工具故障不能让整条链路挂掉
            logger.warning("工具 %s 执行失败：%s", name, exc)
            return f"工具 {name} 执行失败：{exc}"

    def record(self, step: int, name: str, args: Dict[str, Any], output: str) -> ToolCall:
        # 造一条调用记录（ToolCall 实例），存进 self.calls 并返回给调用方
        call = ToolCall(step=step, name=name, args=args, output=output)
        self.calls.append(call)
        return call


_TOOL_HINTS = (
    ("list_papers", ("上传", "哪些文献", "几篇", "文献列表", "有哪些论文", "list")),
    ("trace_lineage", ("演进", "脉络", "技术路线", "发展历程", "lineage", "evolution")),
    ("get_citations", ("谁引用", "谁改进", "后续工作", "被引用", "cited by", "follow-up")),
    ("get_references", ("参考文献", "引用了哪些", "基于哪些", "references", "build on")),
    ("compare_methods", ("对比表", "对比这些方法", "方法对比", "comparison table")),
    ("describe_figure", ("这张图", "图表", "figure", "曲线图", "柱状图", "架构图", "流程图")),
    ("compare_papers", ("对比", "比较", "区别", "差异", "compare", "difference")),
    ("read_section", ("第", "章节", "section", "全文", "原文", "读一下")),
)


def needs_tools(question: str) -> bool:
    """粗粒度判断这个问题要不要走多步工具链。

    只做「明显需要多步」的识别，剩下的一律走单步 RAG（更快更便宜）。
    """
    text = (question or "").strip().lower()
    if not text:
        return False
    for _tool, keywords in _TOOL_HINTS:
        if any(keyword.lower() in text for keyword in keywords):
            return True
    return False


def suggest_tools(question: str) -> List[str]:
    """给出建议优先使用的工具（用于提示模型，不强制）。"""
    text = (question or "").strip().lower()
    picked: List[str] = []
    for tool, keywords in _TOOL_HINTS:
        if any(keyword.lower() in text for keyword in keywords):
            picked.append(tool)
    return picked