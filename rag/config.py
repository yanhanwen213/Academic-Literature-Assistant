"""RAG 层的可配置项，统一从环境变量读取。"""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# 无论从哪个目录启动，都固定读取项目根目录的 .env（utf-8-sig 兼容带 BOM 的文件）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", encoding="utf-8-sig")


def _project_path(name: str, default_relative: str) -> str:
    """读取路径型配置；相对路径一律相对项目根目录解析。"""
    raw = (os.getenv(name) or "").strip()
    path = Path(raw) if raw else Path(default_relative)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


# 向量库数据目录（LanceDB 是本地文件库，不需要任何数据库服务）
LANCEDB_DIR = _project_path("LANCEDB_DIR", "data/lancedb")

# 会话历史 SQLite 文件
SQLITE_PATH = _project_path("SQLITE_PATH", "data/assistant.db")

# 本地 Ollama 服务地址（用于 nomic-embed-text 向量化）
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _port_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ollama_reachable(timeout: float = 1.0) -> bool:
    """快速探测本地 Ollama 是否已启动。"""
    parsed = urlparse(OLLAMA_HOST)
    return _port_open(parsed.hostname or "127.0.0.1", parsed.port or 11434, timeout)


# 不同的 embedding 模型维度不同，必须落在不同的向量表里
EMBEDDING_MODELS = {
    "nomic-embed-text": {
        "provider": "ollama",
        "dimensions": 768,
        "collection": "rag_documents_ollama",
    },
    "text-embedding-3-small": {
        "provider": "openai",
        "dimensions": 1536,
        "collection": "rag_documents_openai",
    },
}

# 默认使用本地 Ollama 的 nomic-embed-text（免费、离线、768 维）
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")

# 每次提问时检索到 prompt 中的文献片段数
DEFAULT_NUM_DOCUMENTS = int(os.getenv("RAG_NUM_DOCUMENTS", "2"))

# ---------------------------------------------------------------- 切分策略
# 每个片段的字符数上限（英文约 150~200 词，中文约 500~600 字）
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
# 相邻片段的重叠字符数，避免答案正好被切断
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
# 过短的片段（图注、页码、页眉残留）直接丢弃，减少噪声
MIN_CHUNK_CHARS = int(os.getenv("RAG_MIN_CHUNK_CHARS", "40"))
# 同一篇文献内文本完全相同的片段只保留一份
DEDUPE_CHUNKS = _env_bool("RAG_DEDUPE_CHUNKS", True)
# 入库文本前加一行「《论文标题》 章节名」，提升检索命中率
CONTEXTUAL_HEADER = _env_bool("RAG_CONTEXTUAL_HEADER", True)
# 是否把「参考文献」章节也入库（默认不入库，噪声大且对问答无用）
INCLUDE_REFERENCES = _env_bool("RAG_INCLUDE_REFERENCES", False)

# ---------------------------------------------------------------- arXiv 导入
ARXIV_TIMEOUT = int(os.getenv("ARXIV_TIMEOUT", "30"))
# arXiv 被墙/网关拦截时，给 arXiv 请求单独配代理（不影响 LLM / Semantic Scholar）
# 例如 ARXIV_PROXY=http://127.0.0.1:7890
ARXIV_PROXY = (os.getenv("ARXIV_PROXY") or "").strip()
# API 地址（默认 https，失败会退回 http；被劫持时也可换成镜像）
ARXIV_API_BASE = (os.getenv("ARXIV_API_BASE") or "").strip().rstrip("/")

# ---------------------------------------------------------------- 检索增强
# 检索模式：dense（纯向量）/ hybrid（BM25 + 向量 + RRF 融合）
RETRIEVAL_MODE = (os.getenv("RAG_RETRIEVAL_MODE") or "hybrid").strip().lower()
# 融合前每一路各自召回多少个候选（越大召回越全，但 rerank 开销越大）
RECALL_PER_ROUTE = int(os.getenv("RAG_RECALL_PER_ROUTE", "20"))
# RRF 的平滑常数，业界常用 60（越小越看重头部结果）
RRF_K = int(os.getenv("RAG_RRF_K", "60"))

# Query 改写：off / multi_query / history（多轮补全）/ both
QUERY_REWRITE = (os.getenv("RAG_QUERY_REWRITE") or "both").strip().lower()
# 一次生成几个查询变体（含原句）
REWRITE_NUM_QUERIES = int(os.getenv("RAG_REWRITE_NUM_QUERIES", "2"))

# Rerank：off / llm / cross-encoder / auto（优先 cross-encoder，失败回退 llm）
RERANK_MODE = (os.getenv("RAG_RERANK_MODE") or "auto").strip().lower()
# cross-encoder 模型名（中英混合文献建议 bge-reranker-base）
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
# rerank 后保留多少条送进 prompt
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "3"))
# cross-encoder 本地模型缓存目录
RERANK_CACHE_DIR = _project_path("RAG_RERANK_CACHE_DIR", "data/models")

# ---------------------------------------------------------------- 引用溯源
# 是否强制模型在回答里带「[编号]」引用，并对回答做引用校验
CITATION_ENABLED = _env_bool("RAG_CITATION_ENABLED", True)
# 回答里一条引用都没有时，是否提示「未溯源」
CITATION_REQUIRED = _env_bool("RAG_CITATION_REQUIRED", True)

# ---------------------------------------------------------------- 拒答校准
# 是否启用拒答（检索置信度不足时直接说「文献里没有」，而不是让模型硬编）
REFUSAL_ENABLED = _env_bool("RAG_REFUSAL_ENABLED", True)
# 检索置信度阈值（0~1），低于它就拒答
REFUSAL_MIN_CONFIDENCE = float(os.getenv("RAG_REFUSAL_MIN_CONFIDENCE", "0.30"))

# ---------------------------------------------------------------- 多步工具链
# 工具模式：auto（按问题复杂度自动选）/ on（总是多步）/ off（单步 RAG）
TOOL_MODE = (os.getenv("RAG_TOOL_MODE") or "auto").strip().lower()
# 工具链最多走几步（防止无限循环）
TOOL_MAX_STEPS = int(os.getenv("RAG_TOOL_MAX_STEPS", "4"))
# 单次工具返回最多多少字符（避免撑爆上下文）
TOOL_MAX_OUTPUT_CHARS = int(os.getenv("RAG_TOOL_MAX_OUTPUT_CHARS", "4000"))
# 工具链最终答案最多带多少条参考资料
TOOL_FINAL_DOCUMENTS = int(os.getenv("RAG_TOOL_FINAL_DOCUMENTS", "4"))

# ---------------------------------------------------------------- 引用图谱
# 引用关系图（networkx 序列化为 JSON）的存放目录
GRAPH_PATH = _project_path("RAG_GRAPH_PATH", "data/citation_graph.json")
# 是否启用引用图谱相关工具（get_references / get_citations / trace_lineage）
GRAPH_ENABLED = _env_bool("RAG_GRAPH_ENABLED", True)
# 抓引用关系的数据源：semantic_scholar（有真实引用）/ pdf（只能拿到论文自己的参考文献）
GRAPH_SOURCE = (os.getenv("RAG_GRAPH_SOURCE") or "semantic_scholar").strip().lower()
# Semantic Scholar API（免费，无需 key；有 key 配额更高）
SEMANTIC_SCHOLAR_API = os.getenv(
    "SEMANTIC_SCHOLAR_API", "https://api.semanticscholar.org/graph/v1"
)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
# 网络超时（秒）
GRAPH_HTTP_TIMEOUT = int(os.getenv("RAG_GRAPH_HTTP_TIMEOUT", "15"))
# 多跳追溯默认深度上限
GRAPH_MAX_HOPS = int(os.getenv("RAG_GRAPH_MAX_HOPS", "2"))

# ---------------------------------------------------------------- 结构化抽取
# 抽取结果（方法对比表）的 SQLite 文件
EXTRACTION_PATH = _project_path("RAG_EXTRACTION_PATH", "data/extraction.db")
# 是否在入库后自动做结构化抽取（会额外消耗 LLM 调用）
EXTRACTION_AUTO = _env_bool("RAG_EXTRACTION_AUTO", True)
# 抽取失败时的最大重试次数
EXTRACTION_MAX_RETRIES = int(os.getenv("RAG_EXTRACTION_MAX_RETRIES", "2"))
# 每篇论文最多喂给抽取模型的字符数（控制成本）
EXTRACTION_MAX_CHARS = int(os.getenv("RAG_EXTRACTION_MAX_CHARS", "6000"))

# ---------------------------------------------------------------- 多模态（文本 + 图像 + 语音）
# 总开关：关闭后前端不再展示多模态入口，后端也不会调用视觉 / 语音接口
MULTIMODAL_ENABLED = _env_bool("RAG_MULTIMODAL_ENABLED", True)
# 视觉模型名；留空时按模型名自动判断（gpt-4o / *-vl / gemini / claude-3 等）
MULTIMODAL_VISION_MODEL = (os.getenv("RAG_VISION_MODEL") or "").strip()
# 一次提问最多附带几张图片、单张图片大小上限（MB）
MULTIMODAL_MAX_IMAGES = int(os.getenv("RAG_MULTIMODAL_MAX_IMAGES", "3"))
MULTIMODAL_MAX_IMAGE_MB = float(os.getenv("RAG_MULTIMODAL_MAX_IMAGE_MB", "8"))
# 是否把上传的 PDF 落盘，供「PDF 图表理解」按需提取图片
STORE_PDF = _env_bool("RAG_STORE_PDF", True)
PAPERS_DIR = _project_path("RAG_PAPERS_DIR", "data/papers")
# PDF 图表提取：最小宽高（像素）、最大张数、最小字节数（过滤图标 / logo）
FIGURE_MIN_WIDTH = int(os.getenv("RAG_FIGURE_MIN_WIDTH", "160"))
FIGURE_MIN_HEIGHT = int(os.getenv("RAG_FIGURE_MIN_HEIGHT", "120"))
FIGURE_MAX_COUNT = int(os.getenv("RAG_FIGURE_MAX_COUNT", "8"))
FIGURE_MIN_BYTES = int(os.getenv("RAG_FIGURE_MIN_BYTES", "4096"))
# 语音转写（OpenAI Whisper 兼容接口）：模型名、接口地址、读取哪个环境变量里的 Key
ASR_MODEL = os.getenv("RAG_ASR_MODEL", "whisper-1")
ASR_BASE_URL = (
    os.getenv("RAG_ASR_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.openai.com/v1"
).strip().rstrip("/")
ASR_API_KEY_ENV = (os.getenv("RAG_ASR_API_KEY_ENV") or "OPENAI_API_KEY").strip()
ASR_TIMEOUT = int(os.getenv("RAG_ASR_TIMEOUT", "90"))

# ---------------------------------------------------------------- AI 安全与可控
# 总开关：关闭后不做提示注入 / 内容过滤 / 输出脱敏
SAFETY_ENABLED = _env_bool("RAG_SAFETY_ENABLED", True)
# 是否拦截「忽略以上指令 / 输出系统提示」这类提示注入
SAFETY_BLOCK_PROMPT_INJECTION = _env_bool("RAG_SAFETY_BLOCK_PROMPT_INJECTION", True)
# 是否拦截明确的违法 / 暴力制作类请求（保守词表，避免误伤学术讨论）
SAFETY_CONTENT_FILTER = _env_bool("RAG_SAFETY_CONTENT_FILTER", True)
# 是否对模型输出里的 API Key / Token / 邮箱等敏感信息做脱敏
SAFETY_REDACT_SECRETS = _env_bool("RAG_SAFETY_REDACT_SECRETS", True)
# 单条提问的最大字符数（超长提问视为滥用）
SAFETY_MAX_QUESTION_CHARS = int(os.getenv("RAG_SAFETY_MAX_QUESTION_CHARS", "8000"))

# ---------------------------------------------------------------- MCP（Model Context Protocol）
# 外部 MCP Server 配置（JSON 字符串），例如：
# MCP_SERVERS={"filesystem":{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."],"transport":"stdio"}}
# 留空则不接入外部 MCP；需要安装 langchain-mcp-adapters。
MCP_SERVERS = (os.getenv("MCP_SERVERS") or "").strip()
