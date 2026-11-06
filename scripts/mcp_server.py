"""把「学术文献助手」的工具暴露成 MCP Server（stdio）。

用途：让任何支持 MCP 的客户端（Claude Desktop、Cursor、其他 Agent）直接调用
本项目的检索 / 引用图谱 / 方法对比 / 图表理解能力，体现「MCP 协议集成」这一评分点。

启动方式：

    pip install "mcp>=1.0"
    python -m scripts.mcp_server

可选环境变量：

    MCP_LIBRARY=默认知识库        # 使用哪个知识库
    DEFAULT_EMBEDDING_MODEL=nomic-embed-text
    OLLAMA_HOST=http://127.0.0.1:11434

客户端配置示例（Claude Desktop / Cursor）：

    {
      "mcpServers": {
        "academic-literature": {
          "command": "python",
          "args": ["-m", "scripts.mcp_server"],
          "cwd": "D:/path/to/Academic_Literature_Assistant"
        }
      }
    }
"""

import os
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag import libraries  # noqa: E402
from rag.citation_graph import CitationGraph  # noqa: E402
from rag.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    PAPERS_DIR,
)
from rag.extraction import ExtractionStore  # noqa: E402
from rag.multimodal import describe_figure, extract_pdf_figures  # noqa: E402
from rag.tools import LiteratureTools  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "未安装 MCP SDK，无法启动 MCP Server。请先执行：\n"
        '    pip install "mcp>=1.0"\n'
        f"原始错误：{exc}"
    ) from exc


mcp = FastMCP("academic-literature-assistant")
_TOOLS: Optional[LiteratureTools] = None
_LLMS: dict = {}


def _pdf_path(source_key: str) -> Optional[Path]:
    if not source_key:
        return None
    path = Path(PAPERS_DIR) / f"{source_key.replace(':', '_')}.pdf"
    return path if path.exists() else None


def _list_figures(source_key: str):
    path = _pdf_path(source_key)
    return extract_pdf_figures(path.read_bytes()) if path else []


def _describe_figure(source_key: str, figure: dict, question: str = "") -> str:
    from agent.config import DEFAULT_LLM_MODEL
    from agent.llm import build_vision_llm
    from rag.multimodal import resolve_vision_model

    vision_model = resolve_vision_model(DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL
    if vision_model not in _LLMS:
        _LLMS[vision_model] = build_vision_llm(vision_model)
    return describe_figure(
        _LLMS[vision_model], vision_model, figure, question=question
    )


def _tools() -> LiteratureTools:
    """延迟构建工具集：只有真正调用工具时才连接向量库。"""
    global _TOOLS
    if _TOOLS is not None:
        return _TOOLS

    library = libraries.get_library(os.getenv("MCP_LIBRARY"))
    from rag.knowledge_base import RAGKnowledgeBase

    knowledge = RAGKnowledgeBase(
        embeddings_model=DEFAULT_EMBEDDING_MODEL, uri=library.lancedb
    )

    graph = None
    try:
        graph = CitationGraph(path=library.graph)
    except Exception:  # noqa: BLE001
        pass

    store = None
    try:
        store = ExtractionStore(library.extraction)
    except Exception:  # noqa: BLE001
        pass

    _TOOLS = LiteratureTools(
        knowledge,
        citation_graph=graph,
        extraction_store=store,
        figure_lister=_list_figures,
        figure_describer=_describe_figure,
    )
    return _TOOLS


@mcp.tool()
def search_literature(query: str) -> str:
    """在已入库文献中做混合检索（向量 + BM25 + RRF + 可选重排），返回相关片段。"""
    return _tools().search_literature(query)


@mcp.tool()
def list_papers() -> str:
    """列出知识库中所有文献的标题、年份、片段数与章节名。"""
    return _tools().list_papers()


@mcp.tool()
def read_section(source_key: str, section: str = "") -> str:
    """按来源标识与章节名读取某篇论文的正文。"""
    return _tools().read_section(source_key, section)


@mcp.tool()
def compare_papers(topic: str) -> str:
    """围绕一个主题分别检索并按文献分组，用于横向对比多篇论文。"""
    return _tools().compare_papers(topic)


@mcp.tool()
def get_references(paper: str) -> str:
    """查询某篇论文引用了哪些论文（沿引用网络向前追溯）。"""
    return _tools().get_references(paper)


@mcp.tool()
def get_citations(paper: str) -> str:
    """查询哪些论文引用了某篇论文（沿引用网络向后追溯）。"""
    return _tools().get_citations(paper)


@mcp.tool()
def trace_lineage(paper: str, hops: int = 2) -> str:
    """多跳追溯某篇论文的引用脉络：继承的工作与后续改进。"""
    return _tools().trace_lineage(paper, hops)


@mcp.tool()
def compare_methods(papers: str = "") -> str:
    """从结构化抽取结果生成 Markdown 方法对比表。"""
    return _tools().compare_methods(papers)


@mcp.tool()
def list_figures(source_key: str) -> str:
    """列出某篇论文 PDF 中抽取到的图表。"""
    return _tools().list_figures(source_key)


@mcp.tool()
def describe_pdf_figure(
    source_key: str, page: int = 0, figure_index: int = 0, question: str = ""
) -> str:
    """用视觉模型解释某篇论文的一张图表（按页码或图表序号定位）。"""
    return _tools().describe_figure(
        source_key, page=page, figure_index=figure_index, question=question
    )


if __name__ == "__main__":
    mcp.run()
