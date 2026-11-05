"""学术文献智能助手 —— 前端（Streamlit）。

前端只负责界面，所有能力都直接调用 `agent/` 里的 ChatSession（它内部使用 `rag/`），
不再经过 HTTP 后端。

启动（在项目根目录执行）：
    streamlit run frontend/app.py
"""

import os
import sys
import time as _time

# 获取当前文件所在项目的根目录路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# frontend 目录也要进 sys.path，方便 `from auth_view import ...`
_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

from typing import List, Optional  # noqa: E402

import streamlit as st  # noqa: E402

from agent.config import (  # noqa: E402
    DEFAULT_LLM_MODEL, # 模型名称
    EMBEDDINGS_CHANGED_HINT, # = "已切换 embedding 模型，请重新上传文献。"
    ENABLE_STORAGE, # = true
    LLM_MODELS,
    LLM_PROVIDER,
    LLM_PROVIDER_LABEL,
    model_supports_tools,
)
from agent.session import ChatSession  # noqa: E402
from admin_view import render_admin  # noqa: E402
from citation_graph_view import render_citation_graph  # noqa: E402
from auth_view import render_login  # noqa: E402
from workspace_ui import render_workspace_style, render_workspace_header, render_chat_welcome, render_page_background, render_console_title
from backend import audit  # noqa: E402
from backend import auth as auth_backend  # noqa: E402
from backend import db as platform_db  # noqa: E402
from backend import users as users_backend  # noqa: E402
from rag import libraries  # noqa: E402
from rag.citation_graph import s2_request_interval  # noqa: E402
from rag.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    RERANK_MODE,
    ollama_reachable,
)
from rag.vector_store import vector_store_hint  # noqa: E402

st.set_page_config(
    page_title="学术文献智能助手",
    page_icon=":orange_heart:",
    layout="wide",
)

# ---------------------------------------------------------------- 登录门禁
# 1) 建表 & 首次启动创建管理员；2) 每次 rerun 回库复核账号状态（停用/改角色立即生效）
platform_db.init_db()
_bootstrap = users_backend.ensure_bootstrap()
if _bootstrap:
    _admin_name, _generated = _bootstrap
    st.session_state["_bootstrap_user"] = _admin_name
    if _generated:
        st.session_state["_bootstrap_password"] = _generated

_me = auth_backend.resolve(st.session_state.get("auth"))
if _me is None:
    st.session_state.pop("auth", None)
    if auth_backend.enabled():
        render_login()
        st.stop()

_me = _me or {
    "username": "local",
    "display_name": "本地调试",
    "role": users_backend.ROLE_ADMIN,
    "library_slug": "",
}
_username = str(_me.get("username") or "")
_role = str(_me.get("role") or users_backend.ROLE_GUEST)
_is_admin = users_backend.can(_role, "view_admin")
_can_write = users_backend.can(_role, "write")
_me_user = users_backend.get_user(_username) if _username != "local" else None


def _index_of(values: List[str], value: Optional[str]) -> int:
    if value and value in values:
        return values.index(value)
    return 0


def _clear_upload_marks() -> None:
    for key in [key for key in st.session_state if key.startswith("uploaded__")]:
        del st.session_state[key]


# ---------------------------------------------------------------- 选择模型
render_console_title()

_role_label = users_backend.ROLE_LABELS.get(_role, _role)
_identity = st.sidebar.container()
_identity.caption(f"👤 {_me.get('display_name') or _username} ｜ {_role_label}")
if auth_backend.enabled() and st.session_state.get("auth"):
    _logout_visual = _identity.container(key="logout_visual")
    if _logout_visual.button("退出登录", key="logout_btn", use_container_width=True):
        auth_backend.logout(_username, _role)
        st.session_state.pop("auth", None)
        st.session_state.pop("session", None)
        st.session_state.pop("bulk_guest_result", None)
        st.session_state.pop("eval_last", None)
        st.session_state.pop("eval_matrix", None)
        st.rerun()

_PAGE_CHAT = "💬 对话"
_PAGE_PAPERS = "📚 文献库"
_PAGE_GRAPH = "🕸 引用图谱"
_PAGE_ADMIN = "⚙ 管理后台"

_page_options = [_PAGE_CHAT, _PAGE_PAPERS]
if _role != users_backend.ROLE_GUEST:
    _page_options.append(_PAGE_GRAPH)
if _is_admin:
    _page_options.append(_PAGE_ADMIN)
if st.session_state.get("main_page") not in _page_options:
    st.session_state["main_page"] = _PAGE_CHAT

with st.sidebar.container(key="workspace_navigation"):
    _page = st.radio(
        "页面",
        options=_page_options,
        format_func=lambda label: label.split(" ", 1)[-1],
        key="main_page",
        help="对话 / 文献库管理 / 引用关系图分开显示，避免同一屏塞太多内容。",
    )
st.sidebar.divider()

embedding_options = list(EMBEDDING_MODELS)

llm_model = st.sidebar.selectbox(
    f"选择大模型（{LLM_PROVIDER_LABEL}）",
    options=LLM_MODELS,
    index=_index_of(LLM_MODELS, DEFAULT_LLM_MODEL),
    key="selected_llm_model",
)
embeddings_model = st.sidebar.selectbox(
    "选择 Embedding 模型",
    options=embedding_options,
    index=_index_of(embedding_options, DEFAULT_EMBEDDING_MODEL),
    key="selected_embeddings_model",
    help="切换后文献需要重新入库。DeepSeek 没有 embedding 接口，向量化仍用 OpenAI 或本地 Ollama。",
)

# ---------------------------------------------------------------- 选择知识库
st.sidebar.divider()
st.sidebar.subheader("知识库（向量库）")

_lib_cache = st.session_state.setdefault("lib_count_cache", {})


def _library_count(library: "libraries.Library") -> int:
    cached = _lib_cache.get(library.slug)
    if cached and _time.time() - cached[0] < 15:
        return cached[1]
    count = libraries.chunk_count(library)
    _lib_cache[library.slug] = (_time.time(), count)
    return count


_libs = libraries.visible_to(_username, _is_admin)
if not _libs:
    _libs = [libraries.default_library()]
_lib_names = [library.name for library in _libs]
_NEW_LIB = "＋ 新建知识库…"
_choices = _lib_names + ([_NEW_LIB] if _is_admin else [])

# 统一在创建控件前把选中值写进 session_state，避免同时传 index 造成的
# "default value + session state" 告警，也让新建/重命名后的切换更稳。
if "pending_library" in st.session_state:
    st.session_state["library_choice"] = st.session_state.pop("pending_library")
if st.session_state.get("library_choice") not in _choices:
    # 成员优先进入管理员分配给自己的库
    _assigned_slug = str(_me.get("library_slug") or "")
    _assigned_name = next(
        (library.name for library in _libs if _assigned_slug and library.slug == _assigned_slug),
        "",
    )
    st.session_state["library_choice"] = _assigned_name or _lib_names[0]

_selected = st.sidebar.selectbox(
    "当前使用的知识库",
    options=_choices,
    key="library_choice",
    help="每套知识库有独立的向量表、引用图谱与对比表数据，切换即用。",
)
if _libs:
    st.sidebar.caption(
        "　".join(
            f"{library.name}·{_library_count(library)} 片段" for library in _libs
        )
    )

if _selected == _NEW_LIB:
    with st.sidebar.form("create_library_form", clear_on_submit=True):
        _new_name = st.text_input(
            "新知识库名称", max_chars=libraries.NAME_MAX_LEN, placeholder="例如：静脉识别"
        )
        _new_note = st.text_input("备注（可选）", placeholder="这个库准备放哪些文献")
        _new_owner = st.text_input("归属用户名（留空 = 共享）", placeholder="例如：alice")
        if st.form_submit_button("创建并切换"):
            try:
                _created = libraries.create_library(
                    _new_name, _new_note, owner=_new_owner.strip()
                )
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"创建失败：{exc}")
            else:
                audit.log(_username, _role, "create_library", target=_created.name)
                st.session_state["pending_library"] = _created.name
                st.rerun()
    active_library = libraries.get_library(st.session_state.get("active_library"))
else:
    active_library = libraries.get_library(_selected)
    if active_library.slug not in {library.slug for library in _libs}:
        active_library = _libs[0]
    st.session_state["active_library"] = active_library.name


def _library_writable(library: "libraries.Library") -> bool:
    """能否往当前库写：管理员都可以；成员只能写自己名下的私有库；访客只读。"""
    if _me_user is not None:
        return users_backend.can_write_library(
            _me_user, library.owner, library.visibility
        )
    return _role == users_backend.ROLE_ADMIN


_library_writable_now = _library_writable(active_library)

with st.sidebar.expander("知识库信息", expanded=False):
    st.caption(f"名称：{active_library.name}")
    st.caption(f"向量目录：{active_library.lancedb}")
    st.caption(
        f"归属：{active_library.owner or '（共享）'}"
        f" ｜ 可见性：{'共享' if active_library.visibility == 'shared' else '私有'}"
    )
    if active_library.note:
        st.caption(f"备注：{active_library.note}")
    if not _library_writable_now:
        st.caption("当前为只读：访客不能修改知识库；成员需要管理员分配一个私有库。")

if _is_admin:
    with st.sidebar.expander("管理知识库（管理员）", expanded=False):
        _rename_to = st.text_input(
            "重命名为", value=active_library.name, key="rename_library_value"
        )
        if st.button("重命名", key="rename_library_btn"):
            try:
                _renamed = libraries.rename_library(active_library.name, _rename_to)
            except Exception as exc:  # noqa: BLE001
                st.error(f"重命名失败：{exc}")
            else:
                audit.log(
                    _username, _role, "rename_library",
                    target=active_library.name, detail=_renamed.name,
                )
                st.session_state["pending_library"] = _renamed.name
                st.rerun()
        st.divider()
        if active_library.is_default:
            st.caption("默认知识库不可删除；只想清空内容请用下方「清空知识库」。")
        else:
            _confirm_delete = st.checkbox(
                "我确认删除该知识库及其全部数据", key="confirm_delete_library"
            )
            if st.button(
                "删除当前知识库",
                key="delete_library_btn",
                disabled=not _confirm_delete,
            ):
                try:
                    libraries.delete_library(active_library.name)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"删除失败：{exc}")
                else:
                    audit.log(
                        _username, _role, "delete_library", target=active_library.name
                    )
                    st.session_state["pending_library"] = libraries.default_library().name
                    st.rerun()

# ---------------------------------------------------------------- 初始化会话
session: Optional[ChatSession] = st.session_state.get("session")
signature = (llm_model, embeddings_model, active_library.slug, _username, _role)

if session is None or st.session_state.get("signature") != signature:
    embeddings_changed = session is not None and session.embeddings_model != embeddings_model
    library_changed = session is not None and session.library.slug != active_library.slug

    with st.spinner("正在初始化 Agent 与知识库…"):
        try:
            session = ChatSession(
                llm_model=llm_model,
                embeddings_model=embeddings_model,
                library=active_library.name,
                owner=_username,
                role=_role,
            )
        except Exception as exc:  # noqa: BLE001 - 直接把原因展示给使用者
            st.error(f"初始化失败：{exc}")
            st.markdown(
                "常见原因：\n"
                f"1. **大模型没配 Key**：当前 provider 是 {LLM_PROVIDER_LABEL}，"
                f"需要在 `.env` 里填 `LLM_PROVIDER={LLM_PROVIDER}` 对应的 API Key（参考 `.env.example`）；\n"
                "2. **Embedding 服务没起**：选的是 `nomic-embed-text` 时，需要本地 Ollama"
                "（`ollama serve`，并执行过 `ollama pull nomic-embed-text`）。\n\n"
                "改完 `.env` 或启动服务后，刷新页面重试。"
            )
            st.stop()

    st.session_state["session"] = session
    st.session_state["signature"] = signature
    _clear_upload_marks()
    if embeddings_changed:
        st.session_state["embeddings_changed"] = True
    if library_changed:
        st.session_state["library_changed"] = True
        # 切换知识库后把上传框也清空：否则框里还挂着的文件会被当成
        # 「待入库」重新灌进新库，导致两个库混进同一批文献。
        st.session_state["uploader_round"] = (
            st.session_state.get("uploader_round", 0) + 1
        )

session = st.session_state["session"]
assert session is not None

# 预热文献清单：顺带把图谱的「在库」标记和向量库对齐，
# 这样侧栏的图谱数字与「文献库」标签页不会打架。
if session.db_available:
    try:
        session.paper_overview()
    except Exception as exc:  # noqa: BLE001 - 预热失败不影响主流程
        st.sidebar.caption(f"文献清单读取失败：{exc}")

# ---------------------------------------------------------------- 侧边栏：知识库状态
st.sidebar.divider()
st.sidebar.subheader("文献入库")

if st.session_state.pop("embeddings_changed", False):
    st.sidebar.info(EMBEDDINGS_CHANGED_HINT)
if st.session_state.pop("library_changed", False):
    st.sidebar.info(
        f"已切换到知识库「{active_library.name}」。"
        "之后的上传会进入这个库，对话已重开（各库文献互不干扰）。"
    )

if not session.db_available:
    st.sidebar.warning(
        f"向量库未就绪：文献入库不可用。{vector_store_hint(active_library.lancedb)}"
    )
elif not ENABLE_STORAGE:
    st.sidebar.caption("会话存储已关闭（ENABLE_ASSISTANT_STORAGE=false），历史对话不会保存。")
elif not session.storage_available:
    st.sidebar.caption("会话历史保存失败，已降级为临时对话（刷新后不保留）。")

if session.embeddings_model == "nomic-embed-text" and not ollama_reachable():
    st.sidebar.warning("本地 Ollama 未启动：文献无法向量化（先运行 `ollama serve`）。")

# ---------------------------------------------------------------- 侧边栏：检索与生成设置
with st.sidebar.expander("检索与生成设置", expanded=False):
    st.caption("改动即时生效，无需重启；分块参数只影响之后新入库的文献。")

    _retrieval_labels = {
        "hybrid（向量 + BM25 + RRF）": "hybrid",
        "dense（只用向量）": "dense",
        "sparse（只用关键词 BM25）": "sparse",
    }
    _retrieval_keys = list(_retrieval_labels)
    _retrieval_default = next(
        (key for key in _retrieval_keys if _retrieval_labels[key] == session.knowledge.retrieval_mode),
        _retrieval_keys[0],
    )
    _retrieval_choice = st.selectbox(
        "检索模式",
        options=_retrieval_keys,
        index=_retrieval_keys.index(_retrieval_default),
        key="opt_retrieval_mode",
        help="hybrid 兼顾语义与专有名词；dense 只看向量；sparse 只看关键词。",
    )

    _rerank_labels = {
        "auto（自动：本地模型优先，否则用大模型）": "auto",
        "llm（让大模型给片段打分）": "llm",
        "cross-encoder（本地精排，需 sentence-transformers）": "cross-encoder",
        "off（不重排）": "off",
    }
    _rerank_keys = list(_rerank_labels)
    _rerank_current = (session.rerank_mode or RERANK_MODE or "auto").strip().lower()
    _rerank_default = next(
        (key for key in _rerank_keys if _rerank_labels[key] == _rerank_current),
        _rerank_keys[0],
    )
    _rerank_choice = st.selectbox(
        "重排方式",
        options=_rerank_keys,
        index=_rerank_keys.index(_rerank_default),
        key="opt_rerank_mode",
        help="精排会显著影响准确率，同时增加延迟/成本。",
    )

    _num_docs = st.slider(
        "送进上下文的片段数",
        min_value=1,
        max_value=8,
        value=int(session.knowledge.num_documents),
        key="opt_num_documents",
        help="越大信息越全，但上下文更长、更容易带进噪声。",
    )
    _recall_k = st.slider(
        "每路召回数",
        min_value=5,
        max_value=50,
        step=5,
        value=int(session.knowledge.recall_per_route),
        key="opt_recall_per_route",
        help="向量路与 BM25 路各自先取多少个候选，再融合。",
    )
    _rrf_k = st.slider(
        "RRF 融合常数 k",
        min_value=10,
        max_value=100,
        step=5,
        value=int(session.knowledge.rrf_k),
        key="opt_rrf_k",
        help="k 越大越平滑；常用 60。",
    )

    _rewrite_labels = {
        "off（不改写）": "off",
        "history（只做多轮补全）": "history",
        "multi_query（只做多问法展开）": "multi_query",
        "both（补全 + 多问法）": "both",
    }
    _rewrite_keys = list(_rewrite_labels)
    _rewrite_current = (session.rewrite_mode or "off").strip().lower()
    _rewrite_default = next(
        (key for key in _rewrite_keys if _rewrite_labels[key] == _rewrite_current),
        _rewrite_keys[0],
    )
    _rewrite_choice = st.selectbox(
        "Query 改写",
        options=_rewrite_keys,
        index=_rewrite_keys.index(_rewrite_default),
        key="opt_rewrite_mode",
    )
    _rewrite_count = st.slider(
        "改写生成问法数",
        min_value=1,
        max_value=4,
        value=int(session.rewrite_num_queries),
        key="opt_rewrite_num_queries",
        help="仅在改写模式包含多问法展开时生效。",
    )

    _refusal_enabled = st.checkbox(
        "启用拒答校准",
        value=bool(session.refusal_enabled),
        key="opt_refusal_enabled",
        help="检索置信度不足时直接说「文献里没有」，减少硬编。",
    )
    _refusal_threshold = st.slider(
        "拒答阈值",
        min_value=0.05,
        max_value=0.80,
        step=0.05,
        value=float(session.refusal_threshold),
        key="opt_refusal_threshold",
        help="阈值越高越容易拒答；中文问英文文献时建议 0.20~0.30。",
    )

    _citation_enabled = st.checkbox(
        "校验引用编号",
        value=bool(session.citation_enabled),
        key="opt_citation_enabled",
        help="检查回答里的 [n] 是否真的对应检索到的片段。",
    )
    _citation_required = st.checkbox(
        "要求必须有引用",
        value=bool(session.citation_required),
        key="opt_citation_required",
        help="开启后会在回答缺少引用时给出警告。",
    )

    _tool_labels = {
        "auto（按问题自动选择）": "auto",
        "on（总是走多步工具链）": "on",
        "off（只走单步检索）": "off",
    }
    _tool_keys = list(_tool_labels)
    _tool_current = (session.tool_mode or "auto").strip().lower()
    _tool_default = next(
        (key for key in _tool_keys if _tool_labels[key] == _tool_current), _tool_keys[0]
    )
    _tool_choice = st.selectbox(
        "多步工具链",
        options=_tool_keys,
        index=_tool_keys.index(_tool_default),
        key="opt_tool_mode",
        help="多步工具链支持「列文献 / 读章节 / 对比 / 引用图谱」这类问题。",
    )

    _graph_enabled = st.checkbox(
        "启用引用图谱",
        value=bool(session.graph_enabled),
        key="opt_graph_enabled",
        help="关闭后不再抓取/使用引用关系，导入会更快。",
    )
    _extraction_enabled = st.checkbox(
        "入库后自动结构化抽取",
        value=bool(session.extraction_enabled),
        key="opt_extraction_enabled",
        help="为「方法对比表」准备数据，会额外消耗 LLM 调用。",
    )

    st.divider()
    st.caption("分块参数（仅影响之后新入库的文献）")
    _chunk_size = st.slider(
        "分块大小（字符）",
        min_value=300,
        max_value=2000,
        step=100,
        value=int(session.chunk_options.get("chunk_size", 800)),
        key="opt_chunk_size",
    )
    _overlap = st.slider(
        "片段重叠（字符）",
        min_value=0,
        max_value=300,
        step=25,
        value=int(session.chunk_options.get("overlap", 100)),
        key="opt_chunk_overlap",
    )
    _min_chars = st.slider(
        "最短片段长度",
        min_value=0,
        max_value=300,
        step=20,
        value=int(session.chunk_options.get("min_chars", 80)),
        key="opt_min_chunk_chars",
    )
    _contextual_header = st.checkbox(
        "给片段加上「标题 + 章节」上下文头",
        value=bool(session.chunk_options.get("contextual_header", True)),
        key="opt_contextual_header",
    )
    _dedupe = st.checkbox(
        "片段去重",
        value=bool(session.chunk_options.get("dedupe", True)),
        key="opt_dedupe_chunks",
    )

session.configure(
    retrieval_mode=_retrieval_labels[_retrieval_choice],
    rerank_mode=_rerank_labels[_rerank_choice],
    num_documents=_num_docs,
    recall_per_route=_recall_k,
    rrf_k=_rrf_k,
    rewrite_mode=_rewrite_labels[_rewrite_choice],
    rewrite_num_queries=_rewrite_count,
    refusal_enabled=_refusal_enabled,
    refusal_threshold=_refusal_threshold,
    citation_enabled=_citation_enabled,
    citation_required=_citation_required,
    tool_mode=_tool_labels[_tool_choice],
    graph_enabled=_graph_enabled,
    extraction_enabled=_extraction_enabled,
    chunk_size=_chunk_size,
    overlap=_overlap,
    min_chars=_min_chars,
    contextual_header=_contextual_header,
    dedupe=_dedupe,
)

# 检索链路的状态条，方便确认「混合检索 / 重排 / 改写」是否真的生效
_reranker = session.knowledge.reranker
_rerank_label = type(_reranker).__name__ if _reranker is not None else "未启用"
st.sidebar.caption(
    f"检索：{session.knowledge.retrieval_mode}"
    f" ｜ 重排：{_rerank_label}"
    f" ｜ 改写：{session.rewrite_mode}"
    f" ｜ 拒答：{('阈值 %.2f' % session.refusal_threshold) if session.refusal_enabled else '关'}"
)
if _reranker is None and _rerank_labels[_rerank_choice] != "off":
    st.sidebar.warning(
        "重排未能启用：本地 cross-encoder 需要 sentence-transformers（且 torch 能正常加载），"
        "可选 `llm` 让大模型打分，或选 `off` 关闭。"
    )


uploader_round = st.session_state.setdefault("uploader_round", 0)
_write_allowed = _can_write and _library_writable_now
uploaded_files = []
if _write_allowed:
    uploaded_files = st.sidebar.file_uploader(
        "批量添加 PDF 文献",
        type="pdf",
        accept_multiple_files=True,
        key=f"pdf_uploader_{uploader_round}",
        help="可一次选择多个 PDF，逐个解析入库。",
    )
else:
    st.sidebar.caption(
        "🔒 只读模式：访客不能导入文献；成员需要管理员分配一个私有知识库后才能写入。"
    )
if uploaded_files:
    def _mark(name: str, size: int) -> str:
        return f"uploaded__{name}__{size}"

    pending = [
        item
        for item in uploaded_files
        if not st.session_state.get(_mark(item.name, len(item.getvalue())))
    ]
    if pending:
        alert = st.sidebar.info(f"正在解析并向量化 {len(pending)} 篇文献…")
        progress = st.sidebar.progress(0.0)
        added_total = 0
        ok_titles: List[str] = []
        empty: List[str] = []
        failed: List[str] = []

        for index, uploaded_file in enumerate(pending, start=1):
            data = uploaded_file.getvalue()
            try:
                result = session.add_pdf(uploaded_file.name, data)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{uploaded_file.name}（{exc}）")
                audit.log(
                    _username, _role, "upload_pdf",
                    target=uploaded_file.name, detail=str(exc), ok=False,
                )
            else:
                st.session_state[_mark(uploaded_file.name, len(data))] = True
                audit.log(
                    _username, _role, "upload_pdf",
                    target=result.title or uploaded_file.name,
                    detail=f"库={active_library.name} / 片段={result.added}",
                )
                if result.added:
                    added_total += result.added
                    ok_titles.append(
                        f"{result.title or uploaded_file.name}（{result.added} 片段）"
                    )
                else:
                    empty.append(uploaded_file.name)
            progress.progress(index / len(pending))

        progress.empty()
        alert.empty()

        if ok_titles:
            _lib_cache.pop(active_library.slug, None)
            st.sidebar.success(
                f"批量入库完成：成功 {len(ok_titles)} 篇，共 {added_total} 个片段。"
            )
            with st.sidebar.expander("查看已入库清单", expanded=False):
                for line in ok_titles:
                    st.caption(line)
        if empty:
            st.sidebar.warning(f"未解析到文本（可能是扫描版）：{'、'.join(empty)}")
        if failed:
            with st.sidebar.expander(f"有 {len(failed)} 篇失败", expanded=True):
                for line in failed:
                    st.caption(line)

if _write_allowed:
    input_arxiv = st.sidebar.text_input(
        "导入 arXiv 论文",
        placeholder="2401.00001 或 https://arxiv.org/abs/2401.00001",
        help="输入 arXiv ID 或论文链接，系统会自动下载 PDF 并按章节解析入库。",
    )
    if st.sidebar.button("导入 arXiv"):
        if not input_arxiv.strip():
            st.sidebar.warning("请先输入 arXiv ID 或论文链接")
        else:
            alert = st.sidebar.info("正在下载并解析 arXiv 论文…")
            try:
                result = session.add_arxiv(input_arxiv.strip())
            except Exception as exc:  # noqa: BLE001
                alert.empty()
                st.sidebar.error(f"导入失败：{exc}")
                audit.log(
                    _username, _role, "import_arxiv",
                    target=input_arxiv.strip(), detail=str(exc), ok=False,
                )
            else:
                alert.empty()
                audit.log(
                    _username, _role, "import_arxiv",
                    target=result.source or input_arxiv.strip(),
                    detail=f"库={active_library.name} / 片段={result.added}",
                )
                if result.added:
                    _lib_cache.pop(active_library.slug, None)
                    st.sidebar.success(f"已入库 {result.added} 个片段：{result.title or result.source}")
                    st.sidebar.caption(
                        f"识别到 {result.sections} 个章节 · {result.year or '年份未知'} · 解析器 {result.parser or '-'}"
                    )
                else:
                    st.sidebar.warning("没有解析到文本，可能是扫描版 PDF。")

st.sidebar.caption(
    f"当前知识库：{active_library.name} ｜ 向量库片段数：{session.knowledge_count()}"
)

_graph_stats = session.graph_stats()
if _graph_stats:
    _papers = _graph_stats.get("papers", 0)
    _edges = _graph_stats.get("edges", 0)
    _in_lib = _graph_stats.get("in_library", 0)
    st.sidebar.caption(
        f"引用图谱：{_papers} 篇文献（在库 {_in_lib} 篇）· {_edges} 条引用边"
    )
    if _edges == 0:
        st.sidebar.caption(
            "（仅登记了本地文献，引用关系未抓取：Semantic Scholar 限流或无网络；"
            "配置 SEMANTIC_SCHOLAR_API_KEY 可提高成功率）"
        )
elif session.citation_graph is None:
    st.sidebar.caption("引用图谱：已关闭或未启用")

if session.citation_graph is not None:
    st.sidebar.caption("文献管理 / 引用关系图 / 刷新引用：见右侧标签页。")

if not model_supports_tools(session.llm_model) and session.tool_mode != "off":
    st.sidebar.warning(
        f"`{session.llm_model}` 不支持函数调用：多步工具链会降级为单步检索，"
        "引用图谱 / 对比表相关工具不会生效。想用这些能力请换 `deepseek-chat` 等支持 tools 的模型。"
    )

# ---------------------------------------------------------------- 侧边栏：历史
st.sidebar.divider()
st.sidebar.subheader("会话历史")
_runs = session.list_run_meta()
_run_ids = [run["run_id"] for run in _runs]

# 会话被删掉后，旧的选中值可能还留在 session_state 里，这里先清理掉
if st.session_state.get("selected_run") not in _run_ids:
    st.session_state.pop("selected_run", None)

if _runs:
    _labels = {run["run_id"]: run["label"] for run in _runs}
    _selected = st.sidebar.selectbox(
        "历史会话",
        options=_run_ids,
        format_func=lambda run_id: _labels.get(run_id, run_id),
        key="selected_run",
    )
    _selected_meta = next((run for run in _runs if run["run_id"] == _selected), {})
    if _selected == session.run_id:
        st.sidebar.caption("· 当前就在这个会话里")
    elif session.run_id not in _run_ids:
        st.sidebar.caption("· 当前是新对话，首次提问后会出现在列表里")

    _col_load, _col_delete = st.sidebar.columns(2)
    if _col_load.button("加载", use_container_width=True, key="load_run_btn"):
        try:
            session.load_run(_selected)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"加载失败：{exc}")
        else:
            st.rerun()

    if _col_delete.button("删除", use_container_width=True, key="delete_run_btn"):
        st.session_state["pending_delete_run"] = _selected

    if st.session_state.get("pending_delete_run") == _selected:
        st.sidebar.warning("删除后无法恢复，确定删除这个会话吗？")
        _col_yes, _col_no = st.sidebar.columns(2)
        if _col_yes.button(
            "确认删除", type="primary", use_container_width=True, key="confirm_delete_btn"
        ):
            try:
                session.delete_run(_selected)
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"删除失败：{exc}")
            else:
                st.session_state.pop("pending_delete_run", None)
                st.rerun()
        if _col_no.button("取消", use_container_width=True, key="cancel_delete_btn"):
            st.session_state.pop("pending_delete_run", None)
            st.rerun()

    with st.sidebar.expander("重命名会话"):
        _new_title = st.text_input(
            "会话名称",
            value=_selected_meta.get("title", ""),
            key=f"rename_title_{_selected}",
            help="留空则恢复为「按第一句提问自动命名」",
        )
        if st.button("保存名称", use_container_width=True, key="rename_run_btn"):
            try:
                session.rename_run(_selected, _new_title)
            except Exception as exc:  # noqa: BLE001
                st.error(f"重命名失败：{exc}")
            else:
                st.rerun()
else:
    st.sidebar.caption("暂无历史记录（需要开启会话存储）。")

if st.sidebar.button("新建对话", use_container_width=True):
    try:
        session.reset()
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"新建对话失败：{exc}")
    else:
        st.rerun()

# ---------------------------------------------------------------- 主区域
render_workspace_style()
render_page_background(_page)
render_workspace_header()

st.caption(
    f"知识库：{active_library.name} ｜ 模型：{session.llm_model} "
    f"｜ Embedding：{session.embeddings_model} ｜ Run：{session.run_id or '-'}"
)


def _graph_dot(snapshot: dict) -> str:
    """把子图快照转成 Graphviz DOT（不依赖任何前端绘图库）。"""

    def _escape(text: str, limit: int = 24) -> str:
        cleaned = (text or "").replace("\\", " ").replace('"', "'").replace("\n", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned[:limit] + ("…" if len(cleaned) > limit else "")

    nodes = snapshot.get("nodes") or []
    index = {node["id"]: f"n{position}" for position, node in enumerate(nodes)}

    lines = [
        "digraph citations {",
        '  rankdir=LR; bgcolor="transparent";',
        '  graph [size="9,4.6", ratio=compress, nodesep=0.22, ranksep=0.5];',
        '  node [shape=box, style="rounded,filled", fontsize=9, width=1.6,',
        '        margin="0.08,0.05"];',
        '  edge [color="#9aa4b2", arrowsize=0.55];',
    ]
    for node in nodes:
        label = _escape(
            node.get("title") or node.get("id"),
            28 if node.get("is_center") else 20,
        )
        if node.get("year"):
            label += f"\\n{node['year']}"
        if node.get("is_center"):
            style = 'fillcolor="#e6daf3", color="#82699f", penwidth=2.0'
        elif node.get("in_library"):
            style = 'fillcolor="#e1eeeb", color="#6d988c"'
        elif node.get("local_ref"):
            style = 'fillcolor="#f1f3f5", color="#adb5bd"'
        else:
            style = 'fillcolor="#ffffff", color="#cccccc"'
        lines.append(f'  {index[node["id"]]} [label="{label}", {style}];')

    for edge in snapshot.get("edges") or []:
        source = index.get(edge.get("source"))
        target = index.get(edge.get("target"))
        if source and target:
            lines.append(f"  {source} -> {target};")

    lines.append("}")
    return "\n".join(lines)


if _page == _PAGE_CHAT:
    if not any(m.get('role') in ('user', 'assistant') for m in session.messages):
        render_chat_welcome()
    for message in session.messages:
        role = message.get("role")
        if role == "system":
            continue
        with st.chat_message(role or "assistant"):
            st.markdown(message.get("content") or "")

    _remaining = (
        users_backend.remaining_quota(_me_user) if _me_user is not None else None
    )
    if _me_user is not None and _remaining is not None:
        st.caption(f"今日剩余提问额度：{_remaining} 次（管理员可在后台调整）")

    prompt = st.chat_input("就你上传的文献提问…")
    if prompt:
        if _remaining == 0:
            st.warning(
                "今日提问额度已用完。请明天再试，或联系管理员在「管理后台 → 用户与权限」里调整配额。"
            )
            st.stop()

        with st.chat_message("user"):
            st.markdown(prompt)

        _started = _time.monotonic()
        _answer_text = ""
        with st.chat_message("assistant"):
            try:
                _streamed = st.write_stream(session.stream_answer(prompt))
                if isinstance(_streamed, str):
                    _answer_text = _streamed
                elif _streamed:
                    _answer_text = "".join(str(part) for part in _streamed)
            except Exception as exc:  # noqa: BLE001
                _elapsed = int((_time.monotonic() - _started) * 1000)
                if _me_user is not None:
                    users_backend.record_question(
                        _username,
                        active_library.name,
                        len(prompt),
                        0,
                        _elapsed,
                        model=session.llm_model,
                        ok=False,
                        error=str(exc),
                    )
                st.error(f"生成回答失败：{exc}")
                st.stop()

            # 可信度面板：引用校验、检索置信度、工具调用轨迹
            report = session.last_report
            confidence = session.last_confidence
            tool_trace = session.last_tool_trace
            _refused = bool(getattr(session, "last_refused", False))

            if _me_user is not None:
                users_backend.record_question(
                    _username,
                    active_library.name,
                    len(prompt),
                    len(_answer_text),
                    int((_time.monotonic() - _started) * 1000),
                    model=session.llm_model,
                    grounded=(report.grounded if report is not None else None),
                    confidence=(confidence.score if confidence is not None else None),
                    tool_steps=len(tool_trace or []),
                    refused=_refused,
                )
                _remaining_now = users_backend.remaining_quota(_me_user)
                if _remaining_now is not None:
                    st.caption(f"今日剩余额度：{_remaining_now} 次")

            if confidence is not None:
                label = f"检索置信度 {confidence.score:.2f}"
                if confidence.score >= session.refusal_threshold:
                    st.caption(f"✅ {label}（{confidence.describe()}）")
                else:
                    st.caption(f"⚠️ {label}（{confidence.describe()}）")
            if _refused:
                st.caption("⛔ 已触发拒答：文献里没有足够依据，未让模型强行作答。")

        if report is not None and report.sources:
            if report.grounded:
                st.caption(f"✅ {report.summary()}")
            else:
                st.caption(f"⚠️ {report.summary()} —— 回答缺少可溯源的引用")
            with st.expander("查看依据文献", expanded=False):
                for source in report.used_sources or report.sources:
                    st.markdown(f"**{source.label}**")
                    st.caption(source.snippet())

        if tool_trace:
            with st.expander("工具调用轨迹", expanded=False):
                for line in tool_trace:
                    st.markdown(f"- {line}")

        if session.last_table:
            st.caption("已生成方法对比表（见回答末尾）")

# ---------------------------------------------------------------- 页面：文献库
if _page == _PAGE_PAPERS:
    st.subheader("文献库")
    if not session.db_available:
        st.warning(
            f"向量库未就绪，暂时无法管理文献。{vector_store_hint(active_library.lancedb)}"
        )

    _papers = session.paper_overview() if session.db_available else []
    if not _papers:
        st.info("还没有入库的文献。用左侧「批量添加 PDF」或「导入 arXiv」先添加几篇。")
    else:
        _total_chunks = sum(row["chunks"] for row in _papers)
        _extracted = sum(1 for row in _papers if row["extracted"])
        _lib_cols = st.columns(4)
        _lib_cols[0].metric("文献数", len(_papers))
        _lib_cols[1].metric("片段数", _total_chunks)
        _lib_cols[2].metric("已抽方法画像", f"{_extracted}/{len(_papers)}")
        _lib_cols[3].metric(
            "图谱已建边", sum(1 for row in _papers if row["graph_node"])
        )

        _table = [
            {
                "标题": row["title"],
                "年份": row["year"],
                "章节": row["sections"],
                "片段": row["chunks"],
                "页数": row["pages"],
                "它引用了": row["references"],
                "被引用": row["citations"],
                "方法画像": "✅" if row["extracted"] else "—",
                "来源": row["source_key"],
            }
            for row in _papers
        ]
        st.dataframe(_table, width="stretch", hide_index=True)

        _paper_keys = [row["source_key"] for row in _papers]
        _paper_labels = {
            row["source_key"]: (
                f"{row['title']}（{row['year']}）" if row["year"] else row["title"]
            )
            for row in _papers
        }
        _target = st.selectbox(
            "选择一篇文献进行操作",
            options=_paper_keys,
            format_func=lambda key: _paper_labels.get(key, key),
            key="paper_action_target",
        )
        _target_row = next(row for row in _papers if row["source_key"] == _target)

        if _library_writable_now:
            _act_refresh, _act_delete, _act_info = st.columns([1, 1, 2])
            if _act_refresh.button(
                "重抓这篇的引用",
                key="paper_refresh_btn",
                help="只向 Semantic Scholar 重新查询这一篇的 references / cited-by。",
            ):
                try:
                    _counts = session.refresh_paper_graph(_target)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"抓取失败：{exc}")
                else:
                    audit.log(
                        _username, _role, "refresh_graph",
                        target=_target_row["title"], detail=str(_counts),
                    )
                    st.success(
                        f"references {_counts.get('references', 0)} 条、"
                        f"citations {_counts.get('citations', 0)} 条"
                        f"（其中 PDF 兜底 {_counts.get('local_references', 0)} 条）"
                    )

            if _act_delete.button("删除这篇", key="paper_delete_btn"):
                st.session_state["pending_delete_paper"] = _target

            _act_info.caption(
                f"来源 {_target_row['source_key']} ｜ "
                f"{_target_row['chunks']} 个片段 ｜ {_target_row['sections']} 个章节"
            )
        else:
            st.caption("🔒 只读模式：当前账号不能修改这个知识库。")

        with st.expander("查看解析详情"):
            st.markdown(
                f"**{_target_row['title']}**"
                + (f"（{_target_row['year']}）" if _target_row["year"] else "")
            )
            if _target_row["authors"]:
                st.caption(_target_row["authors"])
            if _target_row["section_names"]:
                st.caption("识别到的章节：" + "、".join(_target_row["section_names"][:12]))
            else:
                st.caption("没有识别出章节名（可能是扫描件或排版特殊）。")
            st.caption(
                f"覆盖 {_target_row['pages']} 页 ｜ 引用关系："
                f"它引用了 {_target_row['references']} 篇、被 {_target_row['citations']} 篇引用"
            )

        if _library_writable_now and st.session_state.get("pending_delete_paper") == _target:
            st.warning(
                f"将删除《{_target_row['title']}》的 {_target_row['chunks']} 个片段、"
                "方法画像与「在库」标记（引用边保留）。删除后不可恢复。"
            )
            _del_yes, _del_no = st.columns([1, 1])
            if _del_yes.button(
                "确认删除", type="primary", key="confirm_delete_paper_btn"
            ):
                try:
                    _deleted = session.delete_paper(_target)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"删除失败：{exc}")
                    audit.log(
                        _username, _role, "delete_paper",
                        target=_target_row["title"], detail=str(exc), ok=False,
                    )
                else:
                    audit.log(
                        _username, _role, "delete_paper",
                        target=_target_row["title"],
                        detail=f"库={active_library.name} / 片段={_deleted['chunks']}",
                    )
                    st.session_state.pop("pending_delete_paper", None)
                    st.session_state["uploader_round"] = uploader_round + 1
                    st.success(f"已删除 {_deleted['chunks']} 个片段")
                    st.rerun()
            if _del_no.button("取消", key="cancel_delete_paper_btn"):
                st.session_state.pop("pending_delete_paper", None)
                st.rerun()

        if _is_admin:
            st.divider()
            with st.expander("危险操作：清空整个知识库"):
                st.caption(
                    "会删除该库的全部文献、引用图谱与结构化抽取结果，不可恢复。"
                    "只想删一篇请用上面的「删除这篇」。仅管理员可执行。"
                )
                _clear_ok = st.checkbox(
                    "我确认要清空这个知识库", key="confirm_clear_kb"
                )
                if st.button(
                    "清空知识库",
                    type="primary",
                    disabled=not _clear_ok,
                    key="clear_kb_btn",
                ):
                    try:
                        session.clear_knowledge_base()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"清空失败：{exc}")
                        audit.log(
                            _username, _role, "clear_library",
                            target=active_library.name, detail=str(exc), ok=False,
                        )
                    else:
                        audit.log(
                            _username, _role, "clear_library", target=active_library.name
                        )
                        _clear_upload_marks()
                        _lib_cache.pop(active_library.slug, None)
                        st.session_state["uploader_round"] = uploader_round + 1
                        st.rerun()

# ---------------------------------------------------------------- 页面：引用图谱
if _page == _PAGE_GRAPH:
    st.subheader("引用关系图")
    _graph_available = session.citation_graph is not None
    _graph_overview = session.graph_overview() if _graph_available else []

    if not _graph_available:
        st.info(
            "引用图谱未启用：在左侧「检索与生成设置」里勾选「启用引用图谱」后重新提问。"
        )
    elif not _graph_overview:
        st.info(
            "图谱里还没有在库论文。先上传文献，系统会自动抓取引用关系；"
            "抓不到时可用 PDF 的参考文献兜底。"
        )
    else:
        _stats = session.graph_stats()
        _gcols = st.columns(3)
        _gcols[0].metric("节点总数", _stats.get("papers", 0))
        _gcols[1].metric("引用边", _stats.get("edges", 0))
        _gcols[2].metric("在库论文", _stats.get("in_library", 0))

        _center_ids = [node["id"] for node in _graph_overview]
        _center_labels = {node["id"]: node["label"] for node in _graph_overview}
        _head, _hops_col, _dir_col = st.columns([2.6, 1, 1.4])
        _center = _head.selectbox(
            "中心论文",
            options=_center_ids,
            format_func=lambda node_id: _center_labels.get(node_id, node_id),
            key="graph_center",
            help="按「被引次数」倒序，被引多的论文更容易看到「谁引用了它」。",
        )
        _hops = _hops_col.radio(
            "跳数", options=[1, 2], index=0, horizontal=True, key="graph_hops"
        )
        _direction_labels = {
            "both（双向）": "both",
            "out（它引用了谁）": "out",
            "in（谁引用了它）": "in",
        }
        _direction_keys = list(_direction_labels)
        _direction_choice = _dir_col.radio(
            "方向",
            options=_direction_keys,
            index=0,
            horizontal=False,
            key="graph_direction",
            help="只看一个方向时图会小很多，也更聚焦。",
        )
        _direction = _direction_labels[_direction_choice]
        _max_nodes = st.slider(
            "最多显示节点数",
            min_value=5,
            max_value=60,
            value=15,
            step=5,
            key="graph_max_nodes",
            help="节点增多时保留清晰字号，可拖动平移或缩放查看，默认显示 15 个。",
        )

        _snapshot = session.graph_snapshot(
            _center, hops=_hops, max_nodes=_max_nodes, direction=_direction
        )
        _center_node = next(
            (node for node in _graph_overview if node["id"] == _center), {}
        )
        _in_count = int(_center_node.get("citations", 0))
        _out_count = int(_center_node.get("references", 0))

        _info_cols = st.columns(3)
        _info_cols[0].metric("它引用了", _out_count)
        _info_cols[1].metric("被引用", _in_count)
        _info_cols[2].metric("当前画出", f"{len(_snapshot['nodes'])} 节点")

        if _snapshot["nodes"]:
            render_citation_graph(_snapshot)
            _caption = (
                f"子图：{len(_snapshot['nodes'])} 个节点 · {len(_snapshot['edges'])} 条边 ｜ "
                "🟧 中心论文 · 🟩 在库 · ⬜ 库外 · 灰框 = PDF 参考文献（S2 未索引）"
            )
            if _snapshot["truncated"]:
                _caption += f" ｜ 邻域共 {_snapshot['total_nodes']} 个节点，已按度数截断"
            st.caption(_caption)
        else:
            st.warning("这个节点还没有任何引用关系。可以点下面的「重抓这篇的引用」。")

        if _in_count == 0:
            _cited_nodes = [
                node for node in _graph_overview if int(node["citations"]) > 0
            ]
            _hint = (
                "想看效果可以把中心换成"
                + "、".join(
                    f"《{node['title'][:22]}》（被引 {node['citations']}）"
                    for node in _cited_nodes[:2]
                )
                + "。"
                if _cited_nodes
                else "库里暂时没有哪篇有被引记录。"
            )
            st.info(
                "**「被引用」显示 0 不代表功能没实现**\n\n"
                "它表示 Semantic Scholar 上还没有这篇论文的被引记录：新论文（2024 之后）"
                "和未被 S2 索引的期刊/会议经常如此，等 S2 收录了引用它的论文才会出现。"
                "「它引用了谁」不受影响，来自 S2 的 references 与 PDF 参考文献兜底。\n\n"
                + _hint
                + " 也可以点下面的「重抓这篇的引用」再查一次。"
            )
        elif _in_count > _max_nodes:
            st.caption(
                f"该论文共有 {_in_count} 篇引用者，当前只画出度数最高的 {_max_nodes} 个。"
                "把「方向」切到「谁引用了它」可以只看入边。"
            )

        _left, _right = st.columns(2)
        with _left:
            with st.expander(f"它引用了谁（{_out_count}）", expanded=False):
                _refs = session.citation_graph.references(_center)
                if not _refs:
                    st.caption("还没有抓到它的参考文献。可点下方「重抓这篇的引用」。")
                for _node in _refs[:20]:
                    _tag = "✅ 在库" if _node.in_library else (
                        "📄 本地参考文献" if _node.venue == "本地参考文献" else "库外"
                    )
                    st.markdown(
                        f"- 《{(_node.title or _node.paper_id)[:80]}》"
                        + (f"（{_node.year}）" if _node.year else "")
                        + f" · {_tag}"
                    )
                if len(_refs) > 20:
                    st.caption(f"…… 共 {len(_refs)} 篇")
        with _right:
            with st.expander(f"谁引用了它（{_in_count}）", expanded=_in_count > 0):
                _cites = session.citation_graph.citations(_center)
                if not _cites:
                    st.caption(
                        "Semantic Scholar 上没有这篇论文的被引记录（0 条），"
                        "不是查询失败：新论文/未被 S2 收录的刊物常见这种情况。"
                    )
                    _has_cites = [
                        node for node in _graph_overview if int(node["citations"]) > 0
                    ]
                    if _has_cites:
                        st.caption(
                            "想看被引效果，可以把中心换成："
                            + "、".join(
                                f"《{node['title'][:24]}》（{node['citations']} 条）"
                                for node in _has_cites[:2]
                            )
                        )
                for _node in _cites[:20]:
                    _tag = "✅ 在库" if _node.in_library else "库外"
                    st.markdown(
                        f"- 《{(_node.title or _node.paper_id)[:80]}》"
                        + (f"（{_node.year}）" if _node.year else "")
                        + f" · {_tag}"
                    )
                if len(_cites) > 20:
                    st.caption(f"…… 共 {len(_cites)} 篇")

        _graph_center_row = next(
            (
                row
                for row in (session.paper_overview() if session.db_available else [])
                if row.get("graph_node") == _center
            ),
            None,
        )
        _btn_cols = st.columns([1.2, 2])
        if _btn_cols[0].button(
            "重抓这篇的引用",
            key="graph_refresh_one_btn",
            disabled=_graph_center_row is None or not _library_writable_now,
        ):
            try:
                _counts = session.refresh_paper_graph(
                    _graph_center_row["source_key"] if _graph_center_row else ""
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"抓取失败：{exc}")
            else:
                audit.log(
                    _username, _role, "refresh_graph",
                    target=_center_labels.get(_center, _center),
                )
                st.success(
                    f"references {_counts.get('references', 0)} 条、"
                    f"citations {_counts.get('citations', 0)} 条"
                )
        _btn_cols[1].caption(
            "新论文在 Semantic Scholar 上常常还没有引用索引，抓不到属正常；"
            "上传时用 PDF 参考文献兜底能补出「它引用了谁」。"
        )

        if _library_writable_now:
            with st.expander("批量维护（整库重新抓取）"):
                _interval = s2_request_interval()
                _requests_per_paper = 2  # references + citations（已解析过的论文不再重复解析）
                _estimate = max(
                    1, round(len(_graph_overview) * _requests_per_paper * _interval)
                )
                st.caption(
                    f"Semantic Scholar 的额度是 **1 请求/秒（全端点累计）**，"
                    f"代码里已按 {_interval:.1f}s 的间隔自动排队。\n\n"
                    f"当前 {len(_graph_overview)} 篇 × {_requests_per_paper} 次请求 "
                    f"≈ **{_estimate} 秒**；抓取过程中可以正常等待，不会触发 429。"
                )
                if st.button("刷新全部引用关系", key="refresh_all_graph_btn"):
                    _bar = st.progress(0.0, text="正在排队抓取…")
                    _status = st.empty()

                    def _on_progress(done: int, total: int, title: str) -> None:
                        ratio = done / total if total else 1.0
                        _bar.progress(
                            min(ratio, 1.0),
                            text=f"已处理 {done}/{total} 篇（约剩 {max(0, round((total - done) * _requests_per_paper * _interval))} 秒）",
                        )
                        _status.caption(f"最近一篇：{title[:40] or '(无标题)'}")

                    try:
                        _refresh = session.refresh_graph(on_progress=_on_progress)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"刷新失败：{exc}")
                    else:
                        _bar.empty()
                        _status.empty()
                        audit.log(
                            _username, _role, "refresh_graph",
                            target=f"整库:{active_library.name}", detail=str(_refresh),
                        )
                        st.success(
                            f"已刷新 {_refresh.get('papers', 0)} 篇："
                            f"references {_refresh.get('references', 0)} 条、"
                            f"citations {_refresh.get('citations', 0)} 条"
                            f"（PDF 兜底 {_refresh.get('local_references', 0)} 条）"
                        )

# ---------------------------------------------------------------- 页面：管理后台
if _page == _PAGE_ADMIN and _is_admin:
    render_admin(_me)
