"""RAG 检索层：文档解析、清洗、切分、向量化、混合检索与评测。

这一层只关心「文献怎么进库、怎么被检索出来」，不包含任何 LLM 对话逻辑，
也不依赖前端与 Web 框架，可以被对话链、脚本、测试直接复用。
"""

from rag.bm25_index import BM25Index, tokenize
from rag.citation_graph import CitationGraph, CitationGraphBuilder, PaperNode
from rag.citations import (
    CitationReport,
    SourceRef,
    build_citation_report,
    build_sources,
    parse_citation_indices,
)
from rag.cleaning import clean_section_text, clean_text
from rag.embedder import get_collection_name, get_embeddings, supported_embeddings
from rag.extraction import (
    ExtractionStore,
    MethodExtractor,
    MethodProfile,
    render_comparison_table,
)
from rag.knowledge_base import RAGKnowledgeBase
from rag.loaders import IngestResult, load_arxiv, load_pdf_bytes, load_pdf_file
from rag.parsers import ParsedPaper, ParsedSection, parse_pdf_bytes
from rag.query_rewrite import QueryRewriter
from rag.refusal import Confidence, assess_confidence, should_refuse
from rag.rerank import CrossEncoderReranker, LLMReranker, build_reranker
from rag.rrf import rrf_fuse, rrf_fuse_scored
from rag.tool_chain import ToolChainRunner
from rag.tools import LiteratureTools, needs_tools
from rag.vector_store import get_vector_store

__all__ = [
    "BM25Index",
    "CitationGraph",
    "CitationGraphBuilder",
    "CitationReport",
    "Confidence",
    "ExtractionStore",
    "LiteratureTools",
    "MethodExtractor",
    "MethodProfile",
    "PaperNode",
    "SourceRef",
    "ToolChainRunner",
    "assess_confidence",
    "build_citation_report",
    "build_sources",
    "needs_tools",
    "parse_citation_indices",
    "render_comparison_table",
    "should_refuse",
    "CrossEncoderReranker",
    "IngestResult",
    "LLMReranker",
    "ParsedPaper",
    "ParsedSection",
    "QueryRewriter",
    "RAGKnowledgeBase",
    "build_reranker",
    "clean_section_text",
    "clean_text",
    "get_collection_name",
    "get_embeddings",
    "get_vector_store",
    "load_arxiv",
    "load_pdf_bytes",
    "load_pdf_file",
    "parse_pdf_bytes",
    "rrf_fuse",
    "rrf_fuse_scored",
    "supported_embeddings",
    "tokenize",
]