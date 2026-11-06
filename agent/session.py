"""会话：把「向量检索 + LangChain RAG 链 + 会话历史」封装成一个对象。

前端只跟 `ChatSession` 打交道：

- 入库：`add_pdf` / `add_arxiv` / `clear_knowledge_base`
- 问答：`stream_answer`（逐块产出文本，前端直接 `st.write_stream`）
- 历史：`list_runs` / `load_run` / `reset`
"""

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from agent.config import (
    DEFAULT_LLM_MODEL,
    ENABLE_STORAGE,
    NUM_HISTORY_MESSAGES,
    WELCOME_MESSAGE,
)
from agent.llm import build_llm, build_vision_llm
from agent.prompts import format_documents, format_sources
from agent.rag_chain import build_rag_chain
from agent.storage import SqliteRunStore, backup_file
from rag.citations import (
    CitationReport,
    build_citation_report,
    build_sources,
    renumber,
)
from rag import libraries
from rag.citation_graph import CitationGraph, CitationGraphBuilder
from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CITATION_ENABLED,
    CITATION_REQUIRED,
    CONTEXTUAL_HEADER,
    DEFAULT_EMBEDDING_MODEL,
    DEDUPE_CHUNKS,
    EXTRACTION_AUTO,
    GRAPH_ENABLED,
    MIN_CHUNK_CHARS,
    PAPERS_DIR,
    QUERY_REWRITE,
    REFUSAL_ENABLED,
    REFUSAL_MIN_CONFIDENCE,
    REWRITE_NUM_QUERIES,
    SAFETY_ENABLED,
    STORE_PDF,
    TOOL_FINAL_DOCUMENTS,
    TOOL_MODE,
)
from rag.extraction import ExtractionStore, MethodExtractor
from rag.knowledge_base import RAGKnowledgeBase
from rag.loaders import IngestResult
from rag.mcp_client import load_mcp_tools
from rag.multimodal import (
    MultimodalAnswerer,
    describe_figure,
    extract_pdf_figures,
    normalize_images,
    resolve_vision_model,
    transcribe_audio,
)
from rag.query_rewrite import QueryRewriter
from rag.refusal import (Confidence, assess_confidence, refusal_message, should_refuse)
from rag.safety import (
    SafetyReport,
    inspect_question,
    safety_rejection_message,
    scrub_output,
)
from rag.tool_chain import ToolChainResult, ToolChainRunner
from rag.tools import needs_tools
from rag.vector_store import vector_store_available, vector_store_hint

logger = logging.getLogger(__name__)


class ChatSession:
    """一个对话会话：持有模型、知识库、RAG 链与对话记录。"""

    def __init__(
        self,
        llm_model: Optional[str] = None,
        embeddings_model: Optional[str] = None,
        library: Optional[str] = None,
        owner: str = "",
        role: str = "member",
    ) -> None:
        _t0 = time.perf_counter()
        self.session_id = uuid.uuid4().hex
        # 会话归属：历史记录按 owner 过滤，管理员可看全部
        self.owner = (owner or "").strip()
        self.role = (role or "member").strip()
        self.is_admin = self.role == "admin"
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.embeddings_model = embeddings_model or DEFAULT_EMBEDDING_MODEL
        # 当前知识库：决定向量目录、引用图与抽取库的位置
        self.library = libraries.get_library(library)
        self.library_name = self.library.name

        self.llm: BaseChatModel = build_llm(self.llm_model)
        self.chain: Runnable = build_rag_chain(self.llm)
        # 多模态：视觉模型可以是另一个模型（RAG_VISION_MODEL），例如文本用 deepseek-chat、
        # 图片用同一 provider 下的视觉模型；构造失败时回退到当前模型，由 available 判定降级。
        vision_model = resolve_vision_model(self.llm_model)
        vision_llm: BaseChatModel = self.llm
        if vision_model and vision_model != self.llm_model:
            try:
                vision_llm = build_vision_llm(vision_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("视觉模型 %s 初始化失败，回退到 %s：%s", vision_model, self.llm_model, exc)
                vision_model = ""
        self.multimodal = MultimodalAnswerer(
            vision_llm, vision_model or self.llm_model
        )
        # 知识库需要 llm 才能做 LLM 重排，必须在 llm 之后构造
        self.knowledge = RAGKnowledgeBase(
            embeddings_model=self.embeddings_model,
            llm=self.llm,
            uri=self.library.lancedb,
        )

        # 运行时开关（前端「高级设置」可以随时改，不用重启）
        self.rewrite_mode = QUERY_REWRITE
        self.rewrite_num_queries = REWRITE_NUM_QUERIES
        self.refusal_enabled = REFUSAL_ENABLED
        self.refusal_threshold = REFUSAL_MIN_CONFIDENCE
        self.citation_enabled = CITATION_ENABLED
        self.citation_required = CITATION_REQUIRED
        self.tool_mode = TOOL_MODE
        self.graph_enabled = GRAPH_ENABLED
        self.extraction_enabled = EXTRACTION_AUTO
        self.safety_enabled = SAFETY_ENABLED
        self.chunk_options: Dict[str, object] = {
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "min_chars": MIN_CHUNK_CHARS,
            "contextual_header": CONTEXTUAL_HEADER,
            "dedupe": DEDUPE_CHUNKS,
        }

        self.rewriter = QueryRewriter(
            self.llm,
            enable_history=self.rewrite_mode in {"history", "both"},
            enable_multi_query=self.rewrite_mode in {"multi_query", "both"},
            num_queries=self.rewrite_num_queries,
        )
        self.rerank_mode = None

        # 引用图谱与结构化抽取：任一初始化失败都不影响基本问答
        self.citation_graph: Optional[CitationGraph] = None
        self.graph_builder: Optional[CitationGraphBuilder] = None
        self.extraction_store: Optional[ExtractionStore] = None
        self.extractor: Optional[MethodExtractor] = None
        self._init_graph()
        self._init_extraction()

        self.tool_runner = ToolChainRunner(
            self.llm,
            self.knowledge,
            citation_graph=self.citation_graph,
            extraction_store=self.extraction_store,
            figure_lister=self.extract_paper_figures,
            figure_describer=self.describe_paper_figure,
            extra_tools=load_mcp_tools(),
        )
        # 最近一次回答的可信度信息（引用校验 / 置信度 / 工具调用 / 对比表）
        self.last_report: Optional[CitationReport] = None
        self.last_confidence: Optional[Confidence] = None
        self.last_tool_trace: List[str] = []
        self.last_table: str = ""
        self.last_refused: bool = False
        self.last_safety: Optional[SafetyReport] = None
        self.last_images: int = 0

        # 向量库（本地文件库）与历史存储的可用性，供前端提示
        self._db_available = vector_store_available(self.library.lancedb)
        self._storage_available = ENABLE_STORAGE
        self._store: Optional[SqliteRunStore] = None
        if ENABLE_STORAGE:
            try:
                self._store = SqliteRunStore()
            except Exception as exc:  # noqa: BLE001 - 存不了历史也要能聊天
                logger.warning("会话历史存储不可用，降级为临时对话: %s", exc)
                self._storage_available = False

        self._count_cache: tuple = (0.0, 0)
        self._papers_cache: Optional[List[Dict[str, Any]]] = None
        self._lock = threading.RLock()
        self.last_active = time.time()
        self.run_id: Optional[str] = None
        self._pending_create = False
        self._start_run([{"role": "assistant", "content": WELCOME_MESSAGE}])
        logger.info(
            "ChatSession 初始化完成：%.2fs（模型 %s ｜ 知识库 %s）",
            time.perf_counter() - _t0,
            self.llm_model,
            self.library_name,
        )

    # ---------------------------------------------------------------- 子系统
    def _init_graph(self) -> None:
        """按需创建引用图谱（路径跟随当前知识库）。"""
        if not self.graph_enabled:
            self.citation_graph = None
            self.graph_builder = None
            return
        try:
            self.citation_graph = CitationGraph(path=self.library.graph)
            self.graph_builder = CitationGraphBuilder(self.citation_graph)
        except Exception as exc:  # noqa: BLE001
            logger.warning("引用图谱不可用，相关工具将禁用：%s", exc)
            self.citation_graph = None
            self.graph_builder = None

    def _init_extraction(self) -> None:
        """按需创建结构化抽取库（路径跟随当前知识库）。"""
        if not self.extraction_enabled:
            self.extraction_store = None
            self.extractor = None
            return
        try:
            self.extraction_store = ExtractionStore(self.library.extraction)
            self.extractor = MethodExtractor(self.llm, self.extraction_store)
        except Exception as exc:  # noqa: BLE001
            logger.warning("结构化抽取不可用：%s", exc)
            self.extraction_store = None
            self.extractor = None

    def _sync_tool_runner(self) -> None:
        if getattr(self, "tool_runner", None) is None:
            return
        self.tool_runner.citation_graph = self.citation_graph
        self.tool_runner.extraction_store = self.extraction_store

    # ---------------------------------------------------------------- 运行时设置
    def configure(self, **options) -> None:
        """应用前端「高级设置」里的选项（全部可在运行时生效，不重建会话）。"""
        knowledge = self.knowledge

        if "retrieval_mode" in options and options["retrieval_mode"]:
            knowledge.set_retrieval_mode(str(options["retrieval_mode"]))
        if "rerank_mode" in options and options["rerank_mode"]:
            mode = str(options["rerank_mode"])
            if mode != self.rerank_mode:
                knowledge.set_reranker(mode, llm=self.llm)
                self.rerank_mode = mode
        if "num_documents" in options and options["num_documents"]:
            knowledge.num_documents = int(options["num_documents"])
        if "recall_per_route" in options and options["recall_per_route"]:
            knowledge.recall_per_route = int(options["recall_per_route"])
        if "rrf_k" in options and options["rrf_k"]:
            knowledge.rrf_k = int(options["rrf_k"])

        if "rewrite_mode" in options and options["rewrite_mode"]:
            mode = str(options["rewrite_mode"])
            self.rewrite_mode = mode
            self.rewriter.enable_history = mode in {"history", "both"}
            self.rewriter.enable_multi_query = mode in {"multi_query", "both"}
        if "rewrite_num_queries" in options and options["rewrite_num_queries"]:
            self.rewrite_num_queries = int(options["rewrite_num_queries"])
            self.rewriter.num_queries = max(1, self.rewrite_num_queries)

        if "refusal_enabled" in options and options["refusal_enabled"] is not None:
            self.refusal_enabled = bool(options["refusal_enabled"])
        if "refusal_threshold" in options and options["refusal_threshold"] is not None:
            self.refusal_threshold = float(options["refusal_threshold"])

        if "citation_enabled" in options and options["citation_enabled"] is not None:
            self.citation_enabled = bool(options["citation_enabled"])
        if "citation_required" in options and options["citation_required"] is not None:
            self.citation_required = bool(options["citation_required"])

        if "tool_mode" in options and options["tool_mode"]:
            self.tool_mode = str(options["tool_mode"])

        if "graph_enabled" in options and options["graph_enabled"] is not None:
            enabled = bool(options["graph_enabled"])
            if enabled != self.graph_enabled:
                self.graph_enabled = enabled
                self._init_graph()
                self._sync_tool_runner()
        if "extraction_enabled" in options and options["extraction_enabled"] is not None:
            enabled = bool(options["extraction_enabled"])
            if enabled != self.extraction_enabled:
                self.extraction_enabled = enabled
                self._init_extraction()
                self._sync_tool_runner()
        if "safety_enabled" in options and options["safety_enabled"] is not None:
            self.safety_enabled = bool(options["safety_enabled"])

        for key in ("chunk_size", "overlap", "min_chars", "contextual_header", "dedupe"):
            if key in options and options[key] is not None:
                self.chunk_options[key] = options[key]

    # ---------------------------------------------------------------- 状态
    @property
    def db_available(self) -> bool:
        return self._db_available

    @property
    def storage_available(self) -> bool:
        return self._storage_available

    @property
    def multimodal_available(self) -> bool:
        return bool(self.multimodal.available)

    @property
    def messages(self) -> List[Dict[str, str]]:
        with self._lock:
            return [dict(message) for message in self._messages]

    # ---------------------------------------------------------------- 运行
    def _start_run(self, messages: List[Dict[str, str]]) -> None:
        """开一个新的 run：生成 id，等第一条提问落库时再写进 SQLite。

        不在这里立刻建行，是为了避免「只打开页面、什么都没问」也在历史里
        留一条空会话。
        """
        self._messages = [dict(message) for message in messages]
        self.run_id = uuid.uuid4().hex
        self._pending_create = self._store is not None

    def _persist(self) -> None:
        if self._store is None or self.run_id is None:
            return
        try:
            if self._pending_create:
                self._store.create_run(self.run_id, self._messages, owner=self.owner)
                self._pending_create = False
            else:
                self._store.save(self.run_id, self._messages, owner=self.owner)
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存会话历史失败: %s", exc)

    def reset(self) -> None:
        """开启全新的 run（清空上下文，保留知识库）。"""
        with self._lock:
            self._start_run([{"role": "assistant", "content": WELCOME_MESSAGE}])

    def _history_owner_filter(self) -> Optional[str]:
        """管理员看全部历史；普通用户只看自己的。"""
        return None if self.is_admin else self.owner

    def _ensure_run_access(self, run_id: str) -> None:
        """越权访问别人的会话时直接拒绝（前端之外的第二道防线）。"""
        if self._store is None:
            return
        owner = self._store.owner_of(run_id)
        if owner is None:
            raise RuntimeError(f"找不到会话记录：{run_id}")
        if not self.is_admin and owner != self.owner:
            # owner 为空的旧数据（启用账号体系之前）只对管理员可见
            raise PermissionError("无权访问该会话")

    def load_run(self, run_id: str) -> None:
        """切换到某个历史 run。"""
        with self._lock:
            if self._store is None:
                raise RuntimeError("会话历史存储未启用，无法加载历史对话")
            self._ensure_run_access(run_id)
            messages = self._store.load(run_id)
            if messages is None:
                raise RuntimeError(f"找不到会话记录：{run_id}")
            self.run_id = run_id
            self._pending_create = False
            self._messages = messages or [
                {"role": "assistant", "content": WELCOME_MESSAGE}
            ]

    def list_runs(self) -> List[str]:
        """历史 run id 列表（按最近使用排序）。"""
        if self._store is None:
            return []
        try:
            return self._store.list_runs(owner=self._history_owner_filter())
        except Exception as exc:  # noqa: BLE001
            logger.debug("读取历史列表失败: %s", exc)
            return []

    def list_run_meta(self) -> List[Dict[str, Any]]:
        """历史会话列表（含标题/时间/条数），供侧栏展示。"""
        if self._store is None:
            return []
        try:
            return self._store.list_run_meta(owner=self._history_owner_filter())
        except Exception as exc:  # noqa: BLE001
            logger.debug("读取历史摘要失败: %s", exc)
            return []

    def rename_run(self, run_id: str, title: str) -> str:
        """给会话改名；标题留空则恢复成「按首条提问自动命名」。"""
        if self._store is None:
            raise RuntimeError("会话历史存储未启用，无法重命名")
        self._ensure_run_access(run_id)
        self._store.rename(run_id, title)
        metas = self.list_run_meta()
        for meta in metas:
            if meta["run_id"] == run_id:
                return meta["title"]
        return title.strip()

    def delete_run(self, run_id: str) -> None:
        """删除一个历史会话；删的是当前会话时自动开一个新对话。"""
        with self._lock:
            if self._store is None:
                raise RuntimeError("会话历史存储未启用，无法删除会话")
            self._ensure_run_access(run_id)
            self._store.delete(run_id)
            if run_id == self.run_id:
                self._start_run([{"role": "assistant", "content": WELCOME_MESSAGE}])

    # ---------------------------------------------------------------- 对话
    def _history(self) -> List[BaseMessage]:
        """把最近几轮对话转成 LangChain 消息，供 prompt 使用。"""
        history: List[BaseMessage] = []
        # 跳过第一句欢迎语
        for message in self._messages[1:][-(NUM_HISTORY_MESSAGES * 2) :]:
            content = message.get("content") or ""
            role = message.get("role")
            if not content:
                continue
            if role == "user":
                history.append(HumanMessage(content=content))
            elif role == "assistant":
                history.append(AIMessage(content=content))
        return history

    def _use_tools(self, question: str) -> bool:
        """是否走多步工具链。"""
        if self.tool_mode == "off":
            return False
        if self.tool_mode == "on":
            return True
        return needs_tools(question)

    def _retrieve(
        self, query: str, queries: List[str], history: List[BaseMessage]
    ) -> tuple:
        """返回 (documents, scored_pairs, tool_result)。

        `scored_pairs` 是 [(document, similarity)]，用于拒答校准。
        """
        tool_result: Optional[ToolChainResult] = None  # tool_result为变量名，Optional[ToolChainResult]为类型，None为赋值
        if self._use_tools(query):
            tool_result = self.tool_runner.run(query, history)
            if tool_result.documents:
                documents = tool_result.documents[:TOOL_FINAL_DOCUMENTS]
                return documents, [], tool_result
            if (tool_result.evidence or "").strip():
                # 工具直接给出了答案（例如 list_papers 列目录），不需要语义检索回退
                return [], [], tool_result
            logger.info("工具链未收集到片段，回退单步检索")

        scored = self.knowledge.search_with_scores(query, queries=queries)
        documents = [document for document, _ in scored]
        # 纯 BM25 模式没有向量距离，相似度恒为 0，交给覆盖率单独判定，
        # 否则会被拒答校准当成「语义无关」永久拒答。
        pairs = scored if self.knowledge._dense_enabled else []
        return documents, pairs, tool_result

    def stream_answer(
        self,
        question: str,
        images: Optional[Sequence] = None,
    ) -> Iterator[str]:
        """流式回答：安全检查 → 改写 → 检索 → 拒答校准 → 生成 → 引用校验 → 脱敏。

        `images` 可传 `bytes` 或 `(bytes, mime)`，用于图表 / 公式截图问答；传了图片时
        会改用视觉模型，并把检索到的文献作为文本上下文一起送入（多模态 RAG）。
        """

        def _generate() -> Iterator[str]:
            with self._lock:
                self.last_active = time.time()
                image_error = ""
                try:
                    image_payloads = normalize_images(images)
                except ValueError as exc:
                    image_payloads = []
                    image_error = str(exc)

                display_question = question
                if image_payloads:
                    display_question = f"{question}\n\n（附带 {len(image_payloads)} 张图片）"
                self._messages.append({"role": "user", "content": display_question})
                history = self._history()
                answer = ""
                self.last_report = None
                self.last_confidence = None
                self.last_tool_trace = []
                self.last_table = ""
                self.last_refused = False
                self.last_safety = None
                self.last_images = len(image_payloads)
                redactions = 0
                try:
                    # 0) 安全闸门：提示注入 / 高危请求直接拦截（可解释、可关闭）
                    safety = inspect_question(question, enabled=self.safety_enabled)
                    self.last_safety = safety
                    if image_error:
                        message = (
                            f"**图片附件无效**：{image_error}\n\n"
                            "请重新选择 PNG / JPEG / WEBP 图片后再试。"
                        )
                        self.last_report = build_citation_report(message, [])
                        answer = message
                        yield message
                        return
                    if safety.blocked:
                        message = safety_rejection_message(safety)
                        self.last_report = build_citation_report(message, [])
                        self.last_refused = True
                        answer = message
                        yield message
                        return

                    # 1) Query 改写：结合历史补全追问，并生成多个检索问法
                    try:
                        queries = self.rewriter.rewrite(question, history)
                    except Exception as exc:  # noqa: BLE001 - 改写失败就用原问题
                        logger.warning("查询改写失败，使用原问题：%s", exc)
                        queries = [question]
                    prompt_question = queries[0] if queries else question  # 如果改写输出为空，则还是会原始question

                    # 2) 检索：多步工具链或单步（向量 + BM25 → RRF → rerank）
                    documents, scored, tool_result = self._retrieve(
                        prompt_question, queries, history
                    )
                    if tool_result is not None and tool_result.calls:
                        self.last_tool_trace = [
                            call.summary for call in tool_result.calls
                        ]
                    if tool_result is not None and tool_result.table:
                        self.last_table = tool_result.table

                    # 3) 拒答校准：置信度不足就直接说「文献里没有」，不让模型硬编。
                    #    但工具链已经查证过的内容（例如「我上传了哪些文献」由 list_papers
                    #    直接给出目录）不再做拒答判断 —— 那类问题本身就没有语义相近的片段。
                    #    附带图片时以图片为主要依据，也不做文本拒答。
                    confidence = assess_confidence(
                        prompt_question,
                        documents,
                        scored=scored or None,
                    )
                    self.last_confidence = None if (image_payloads and not documents) else confidence

                    tool_evidence = (tool_result.evidence or "").strip() if tool_result else ""
                    tool_verified = bool(tool_result and tool_result.calls and tool_evidence)

                    # 工具链已直接查证（如「我上传了哪些文献」）且没有检索片段时，
                    # 置信度面板显示 0.00 会误导；这类回答以工具证据为准，隐藏检索置信度。
                    if tool_verified and not documents:
                        self.last_confidence = None

                    if (
                        not tool_verified
                        and not image_payloads
                        and should_refuse(
                            confidence,
                            self.refusal_threshold,
                            enabled=self.refusal_enabled,
                        )
                    ):
                        message = refusal_message(confidence, self.refusal_threshold)
                        self.last_report = build_citation_report(
                            message, build_sources(documents)
                        )
                        self.last_refused = True
                        answer = message
                        yield message
                        return

                    # 4) 引用溯源：给片段统一编号，生成时要求模型标 [n]
                    sources = renumber(build_sources(documents, dedupe=True))
                    self.last_report = CitationReport(sources=sources)
                    context = format_documents([source.document for source in sources])

                    # 5) 生成：有图片走多模态视觉模型，否则走工具链 / RAG 链
                    if image_payloads and not self.multimodal.available:
                        message = (
                            f"**当前模型 `{self.llm_model}` 未识别为视觉模型。**\n\n"
                            "图片问答需要视觉模型：请在 `.env` 设置 `RAG_VISION_MODEL`（如 `gpt-4o`），"
                            "或把 `LLM_PROVIDER` / `DEFAULT_LLM_MODEL` 换成支持图片的模型后重试。"
                        )
                        self.last_report = build_citation_report(message, sources)
                        answer = message
                        yield message
                        return

                    stream = (
                        self.multimodal.stream(
                            prompt_question, context, image_payloads, history
                        )
                        if image_payloads
                        else self.tool_runner.stream_final(
                            self.chain,
                            prompt_question,
                            context,
                            history,
                            extra_evidence=tool_evidence,
                        )
                    )
                    for chunk in stream:
                        if not chunk:
                            continue
                        # 输出侧脱敏：把 API Key / Token / 邮箱等敏感信息抹掉后再展示
                        chunk, count = scrub_output(chunk, enabled=self.safety_enabled)
                        redactions += count
                        if not chunk:
                            continue
                        answer += chunk
                        yield chunk

                    # 6) 校验回答里的引用编号，并把结果挂到会话上供前端展示
                    if self.citation_enabled and sources:
                        self.last_report = build_citation_report(answer, sources)
                        if self.citation_required and not self.last_report.grounded:
                            logger.info("回答未引用任何来源：%s", question)

                    if sources:
                        used = self.last_report.used_sources or sources
                        footer = "\n\n---\n**依据文献**\n" + format_sources(
                            [source.document for source in used]
                        )
                        answer += footer
                        yield footer

                    # 7) 如果工具生成了方法对比表，直接追加到回答末尾
                    if self.last_table:
                        table_block = "\n\n**方法对比表**\n\n" + self.last_table
                        answer += table_block
                        yield table_block
                finally:
                    # 记录本次脱敏数量，供前端「安全」面板展示
                    if self.last_safety is not None and redactions:
                        self.last_safety.redactions += redactions
                    # 客户端中途断开也把已生成的部分记下来
                    if answer:
                        self._messages.append({"role": "assistant", "content": answer})
                        self._persist()

        return _generate()

    # ---------------------------------------------------------------- 知识库
    def _require_db(self) -> None:
        """入库前快速确认向量库就绪，避免用户白等。"""
        if vector_store_available(self.library.lancedb):
            self._db_available = True
            return
        self._db_available = False
        raise RuntimeError(
            f"向量数据库未就绪，文献暂时无法入库。{vector_store_hint(self.library.lancedb)}"
        )

    def add_pdf(self, filename: str, data: bytes) -> IngestResult:
        with self._lock:
            self._require_db()
            result = self.knowledge.add_pdf_bytes(
                data, filename=filename, **self.chunk_options
            )
            self._count_cache = (0.0, self._count_cache[1] + result.added)
            self._papers_cache = None
            # 落盘一份 PDF，供之后的「图表理解 / 多模态」按需提取图片
            self._store_pdf(result.source_key, data)
            self._post_ingest(result)
            return result

    def _store_pdf(self, source_key: str, data: bytes) -> None:
        """保存原始 PDF，供多模态图表提取使用；失败不影响文本入库。"""
        if not STORE_PDF or not source_key or not data:
            return
        try:
            path = Path(PAPERS_DIR) / f"{source_key.replace(':', '_')}.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
        except OSError as exc:
            logger.warning("原始 PDF 保存失败（不影响问答）：%s", exc)

    def extract_paper_figures(self, source_key: str) -> List[dict]:
        """从已保存的 PDF 中抽取图表，供前端 / API 选择后描述。"""
        if not source_key:
            return []
        path = Path(PAPERS_DIR) / f"{source_key.replace(':', '_')}.pdf"
        if not path.exists() and source_key.startswith("arxiv:"):
            # 兼容旧数据：早期导入的 arXiv 论文只解析未落盘 PDF，
            # 打开「图表理解」时按需补下载一次。
            code = source_key.split(":", 1)[1]
            try:
                from rag.arxiv import download_arxiv_pdf

                data = download_arxiv_pdf(code)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            except Exception as exc:  # noqa: BLE001 - 下载失败则视为无图表
                logger.warning("按需下载 arXiv PDF 失败（图表理解不可用）：%s", exc)
        if not path.exists():
            return []
        try:
            return extract_pdf_figures(path.read_bytes())
        except OSError as exc:
            logger.warning("读取 PDF 图表失败：%s", exc)
            return []

    def describe_paper_figure(
        self,
        source_key: str,
        figure: dict,
        question: str = "",
    ) -> str:
        """用视觉模型解释某篇论文的一张图表。"""
        context = ""
        for document in self.knowledge.vector_store.all_documents():
            if (document.metadata or {}).get("source_key") == source_key:
                context += (document.page_content or "") + "\n"
                if len(context) >= 3000:
                    break
        return describe_figure(
            self.multimodal.llm,
            self.multimodal.model,
            figure,
            question=question,
            context=context[:3000],
        )

    def transcribe_audio(self, data: bytes, filename: str = "audio.webm") -> str:
        """把语音提问转成文字；转写后仍由 `stream_answer` 走安全 + RAG。"""
        return transcribe_audio(data, filename=filename)

    def add_arxiv(self, arxiv_id: str) -> IngestResult:
        with self._lock:
            self._require_db()
            result = self.knowledge.add_arxiv(arxiv_id, **self.chunk_options)
            self._count_cache = (0.0, self._count_cache[1] + result.added)
            self._papers_cache = None
            self._post_ingest(result, arxiv_id=arxiv_id)
            return result

    def _post_ingest(self, result: IngestResult, arxiv_id: str = "") -> None:
        """入库后的两个增强步骤：结构化抽取 + 引用图谱，失败不影响入库。"""
        if not result.documents:
            return

        # 1) 结构化抽取：为「方法对比表」准备数据
        if self.extractor is not None:
            try:
                self.extractor.extract_from_documents(
                    source_key=result.source_key,
                    documents=result.documents,
                    title=result.title,
                    year=result.year,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("结构化抽取失败（不影响入库）：%s", exc)

        # 2) 引用图谱：把这篇论文的引用关系并入图
        if self.graph_builder is not None:
            try:
                arxiv_code = ""
                if arxiv_id:
                    arxiv_code = arxiv_id
                elif result.source_key.startswith("arxiv:"):
                    arxiv_code = result.source_key[len("arxiv:"):]
                self.graph_builder.ingest(
                    arxiv_id=arxiv_code,
                    title=result.title,
                    source_key=result.source_key,
                    references_text=result.references_text,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("引用图谱更新失败（不影响入库）：%s", exc)

    def clear_knowledge_base(self) -> None:
        with self._lock:
            self._require_db()
            # 先把手写的数据（图谱/抽取）备份一份，清空后还能捞回来
            if self.citation_graph is not None:
                backup_file(str(self.citation_graph.path))
            if self.extraction_store is not None:
                backup_file(str(self.extraction_store.db_path))
            self.knowledge.clear()
            # 引用图与抽取结果也属于这套知识库，一并清空，避免「清了向量库
            # 但侧栏还显示旧图谱」的割裂感
            if self.citation_graph is not None:
                try:
                    self.citation_graph.clear()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("清空引用图谱失败：%s", exc)
            if self.extraction_store is not None:
                try:
                    self.extraction_store.clear()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("清空抽取结果失败：%s", exc)
            self._count_cache = (time.time(), 0)
            self._papers_cache = None

    # ---------------------------------------------------------------- 文献库
    def paper_overview(self) -> List[Dict[str, Any]]:
        """按论文汇总向量库里的片段（标题/章节数/片段数/图谱度数）。

        结果缓存到内存：每次 rerun 都全表扫描会让聊天变卡，入库/删除时失效。
        """
        with self._lock:
            if self._papers_cache is not None:
                return [dict(row) for row in self._papers_cache]
            try:
                documents = self.knowledge.vector_store.all_documents()
            except Exception as exc:  # noqa: BLE001
                logger.debug("读取文献列表失败: %s", exc)
                return []

            grouped: Dict[str, Dict[str, Any]] = {}
            for document in documents:
                meta = document.metadata or {}
                key = str(meta.get("source_key") or meta.get("source") or "-")
                entry = grouped.setdefault(
                    key,
                    {
                        "source_key": key,
                        "title": str(meta.get("title") or meta.get("source") or key),
                        "year": str(meta.get("year") or ""),
                        "authors": str(meta.get("authors") or ""),
                        "chunks": 0,
                        "sections": set(),
                        "pages": 0,
                    },
                )
                entry["chunks"] += 1
                entry["sections"].add(str(meta.get("section") or "-"))
                try:
                    entry["pages"] = max(entry["pages"], int(meta.get("page") or 0))
                except (TypeError, ValueError):
                    pass
                if not entry["title"] and meta.get("title"):
                    entry["title"] = str(meta["title"])

            extracted_keys = set()
            if self.extraction_store is not None:
                try:
                    extracted_keys = {
                        profile.source_key for profile in self.extraction_store.all()
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.debug("读取抽取结果失败: %s", exc)

            rows: List[Dict[str, Any]] = []
            for key, entry in grouped.items():
                node_id = ""
                references = citations = 0
                if self.citation_graph is not None:
                    try:
                        node_id = (
                            self.citation_graph.find_by_title(entry["title"])
                            or self.citation_graph.find_by_title(key)
                        )
                        if node_id:
                            degrees = self.citation_graph.degrees(node_id)
                            references = degrees["references"]
                            citations = degrees["citations"]
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("查询图谱度数失败: %s", exc)
                rows.append(
                    {
                        "source_key": key,
                        "title": entry["title"],
                        "year": entry["year"],
                        "authors": entry["authors"],
                        "chunks": entry["chunks"],
                        "sections": len([s for s in entry["sections"] if s and s != "-"]),
                        "section_names": sorted(
                            s for s in entry["sections"] if s and s != "-"
                        ),
                        "pages": entry["pages"],
                        "extracted": key in extracted_keys,
                        "graph_node": node_id,
                        "references": references,
                        "citations": citations,
                    }
                )
            rows.sort(key=lambda row: (row["title"] or "").lower())

            # 图谱的「在库」标记跟着向量库走，避免两处数字对不上
            if self.citation_graph is not None:
                try:
                    self.citation_graph.sync_in_library(list(grouped.keys()))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("同步图谱在库标记失败: %s", exc)

            self._papers_cache = rows
            return [dict(row) for row in rows]

    def delete_paper(self, source_key: str) -> Dict[str, int]:
        """删除单篇文献：向量片段 + 抽取结果 + 图谱的「在库」标记。

        引用边不清除：那篇论文依然存在于学术网络里，只是不在库里了。
        """
        source_key = (source_key or "").strip()
        if not source_key:
            raise RuntimeError("没有指定要删除的文献")
        with self._lock:
            self._require_db()
            documents = self.knowledge.vector_store.all_documents()
            ids = [
                str(document.id)
                for document in documents
                if (document.metadata or {}).get("source_key") == source_key
            ]
            if not ids:
                raise RuntimeError(f"库里没有来源为 {source_key} 的片段")

            if self.citation_graph is not None:
                backup_file(str(self.citation_graph.path))
            if self.extraction_store is not None:
                backup_file(str(self.extraction_store.db_path))

            removed = self.knowledge.remove_documents(ids)
            self._count_cache = (0.0, max(0, self._count_cache[1] - removed))
            self._papers_cache = None

            if self.extraction_store is not None:
                try:
                    self.extraction_store.delete(source_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("删除抽取结果失败：%s", exc)

            graph_nodes = 0
            if self.citation_graph is not None:
                try:
                    graph_nodes = self.citation_graph.mark_out_of_library(source_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("更新引用图谱失败：%s", exc)

            return {"chunks": removed, "graph_nodes": graph_nodes}

    def refresh_paper_graph(self, source_key: str) -> Dict[str, int]:
        """只重抓某一篇的引用关系（比整库刷新快得多）。"""
        if self.graph_builder is None:
            raise RuntimeError("引用图谱未启用")
        source_key = (source_key or "").strip()
        if not source_key:
            raise RuntimeError("没有找到这篇文献的来源标识，无法刷新引用")
        title = ""
        for row in self.paper_overview():
            if row["source_key"] == source_key:
                title = row["title"]
                break
        with self._lock:
            arxiv_code = (
                source_key[len("arxiv:"):] if source_key.startswith("arxiv:") else ""
            )
            return self.graph_builder.ingest(
                arxiv_id=arxiv_code, title=title, source_key=source_key
            )

    def graph_overview(self) -> List[Dict[str, str]]:
        """图谱里在库的论文（前端下拉框用），按被引次数倒序。"""
        if self.citation_graph is None:
            return []
        try:
            nodes = self.citation_graph.in_library_nodes()
        except Exception as exc:  # noqa: BLE001
            logger.debug("读取图谱节点失败: %s", exc)
            return []

        rows: List[Dict[str, Any]] = []
        for node in nodes:
            try:
                degrees = self.citation_graph.degrees(node.paper_id)
            except Exception:  # noqa: BLE001
                degrees = {"references": 0, "citations": 0}
            title = node.title or node.paper_id
            label = f"{title}（{node.year}）" if node.year else title
            label += f" · 被引 {degrees['citations']}"
            rows.append(
                {
                    "id": node.paper_id,
                    "title": title,
                    "label": label,
                    "references": degrees["references"],
                    "citations": degrees["citations"],
                }
            )
        rows.sort(key=lambda row: (-row["citations"], row["title"].lower()))
        return rows

    def graph_snapshot(
        self,
        paper_id: str = "",
        hops: int = 2,
        max_nodes: int = 40,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """某篇论文的邻域子图（前端画图用）。"""
        if self.citation_graph is None:
            return {"center": "", "nodes": [], "edges": [], "truncated": False,
                    "total_nodes": 0, "total_edges": 0}
        target = paper_id or ""
        if not target:
            candidates = self.graph_overview()
            target = candidates[0]["id"] if candidates else ""
        if not target:
            return {"center": "", "nodes": [], "edges": [], "truncated": False,
                    "total_nodes": 0, "total_edges": 0}
        try:
            return self.citation_graph.ego_snapshot(
                target, hops=hops, max_nodes=max_nodes, direction=direction
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("生成图谱快照失败：%s", exc)
            return {"center": target, "nodes": [], "edges": [], "truncated": False,
                    "total_nodes": 0, "total_edges": 0}

    def graph_stats(self) -> Dict[str, int]:
        """引用图谱规模（前端展示用）。"""
        if self.citation_graph is None:
            return {}
        try:
            return self.citation_graph.stats()
        except Exception:  # noqa: BLE001
            return {}

    def refresh_graph(
        self,
        delay: float = 0.0,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, int]:
        """把库里每篇论文的引用关系重新抓一遍。

        S2 对新论文的引用列表常常滞后（只给元数据、references 为空），隔一段时间
        重新抓一次就能补上。请求节流统一由 `rag/citation_graph.py` 里的全局限速器
        负责（1 请求/秒），这里不再自己 sleep，避免双重等待。

        `on_progress(done, total, title)` 用于前端显示进度。
        """
        if self.graph_builder is None or self.citation_graph is None:
            return {"papers": 0, "references": 0, "citations": 0, "local_references": 0}

        targets = [
            (data.get("title") or "", data.get("source_key") or "")
            for _node_id, data in self.citation_graph.graph.nodes(data=True)
            if data.get("in_library")
        ]
        total = len(targets)
        summary = {"papers": 0, "references": 0, "citations": 0, "local_references": 0}
        for index, (title, source_key) in enumerate(targets):
            if index and delay > 0:
                time.sleep(delay)
            arxiv_code = (
                source_key[len("arxiv:"):] if source_key.startswith("arxiv:") else ""
            )
            try:
                counts = self.graph_builder.ingest(
                    arxiv_id=arxiv_code,
                    title=title,
                    source_key=source_key,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("重新抓取 %s 的引用关系失败：%s", title, exc)
                if on_progress is not None:
                    try:
                        on_progress(index + 1, total, title)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            summary["papers"] += 1
            for key in ("references", "citations", "local_references"):
                summary[key] += int(counts.get(key, 0))
            if on_progress is not None:
                try:
                    on_progress(index + 1, total, title)
                except Exception:  # noqa: BLE001
                    pass
        return summary

    def knowledge_count(self) -> int:
        """向量库片段数（带缓存，避免每次都查库）。"""
        if not self._db_available:
            return self._count_cache[1]
        cached_at, cached_value = self._count_cache
        if time.time() - cached_at < 10:
            return cached_value
        value = self.knowledge.count()
        self._count_cache = (time.time(), value)
        return value
