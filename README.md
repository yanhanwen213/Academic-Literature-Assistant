# 学术文献智能助手（RAG + Agent）

> 上传论文 PDF / 导入 arXiv → 版面感知解析 → 章节级切分与混合检索 → 大模型带引用作答，并支持引用图谱、方法对比表与检索评测。

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="LangChain" src="https://img.shields.io/badge/Orchestration-LangChain%201.x-1C3C3C" />
  <img alt="LanceDB" src="https://img.shields.io/badge/VectorDB-LanceDB-6E56CF" />
  <img alt="Ollama" src="https://img.shields.io/badge/Embedding-Ollama%20nomic--embed--text-000000" />
  <img alt="PRs" src="https://img.shields.io/badge/PRs-welcome-brightgreen" />
</p>

一个面向科研场景的「文献问答 + 文献理解」应用：既能基于论文正文回答问题并给出**可溯源的 `[n]` 引用**，也能沿着**引用网络**做多跳推理，还能自动抽取论文方法并生成**方法对比表**。项目不依赖任何常驻数据库或 Docker，除本地向量化服务 Ollama 外，一条命令即可启动。

---

## 目录

- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [功能与页面](#功能与页面)
- [数据是怎么处理的](#数据是怎么处理的)
- [检索链路](#检索链路)
- [可信作答：引用溯源 / 拒答校准 / 多步工具链](#可信作答引用溯源--拒答校准--多步工具链)
- [创新点 1：引用图谱 + 多跳检索](#创新点-1引用图谱--多跳检索)
- [创新点 2：方法论结构化抽取 + 对比表](#创新点-2方法论结构化抽取--对比表)
- [检索评测](#检索评测)
- [完整实验流程](#完整实验流程)
- [切换大模型 / 向量库](#切换大模型--向量库)
- [常见问题](#常见问题)
- [安全提示](#安全提示)

---

## 核心特性

- **版面感知解析**：PyMuPDF 按字号 / 加粗 / 坐标识别标题、作者、章节，丢掉页眉页脚与参考文献，参考文献默认不入库。
- **混合检索**：向量（LanceDB） + BM25（中英混排分词）双路召回，RRF 融合，可选 cross-encoder / LLM 重排，Query 改写支持多轮补全与多问法展开。
- **可信作答**：回答中的 `[n]` 会被校验并映射回章节页码；检索置信度不足时**主动拒答**，不硬编。
- **多步工具链**：8 个工具由模型按问题自动调用（检索 / 列文献 / 读章节 / 对比 / 引用图谱 / 方法对比表），带步数上限与降级。
- **引用图谱**：基于 Semantic Scholar 构建 `networkx` 有向图，支持「它引用了谁 / 谁引用了它 / 多跳演进脉络」。
- **方法结构化抽取**：把论文方法抽成结构化 JSON（方法名 / 核心思想 / 数据集 / 指标 / 参数量 / 优点 / 局限），一键生成 Markdown 对比表。
- **多知识库 + 账号权限**：一个用户绑定一个私有知识库，访客只读；后台含账号管理、审计日志、用量统计与评测中心。
- **可复现评测**：Hit Rate / Recall@k / MRR / 关键词命中率 / 检索精确率，支持 dense → hybrid → rerank 消融对比。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 前端 | Streamlit |
| 编排 | LangChain 1.x（Prompt / LCEL / VectorStore / Retriever） |
| 向量库 | LanceDB（嵌入式，本地文件） |
| Embedding | Ollama `nomic-embed-text`（768 维，本地免费离线）或 OpenAI |
| 稀疏检索 | `rank_bm25`（纯 Python，中文字符 bigram） |
| 解析 | PyMuPDF（主）+ pypdf（兜底） |
| 图谱 | networkx + Semantic Scholar API |
| 存储 | SQLite（标准库）：会话 / 账号 / 审计 / 评测 / 抽取 |

> 唯一的外部服务是**本地向量化**：Ollama（`ollama serve` + `ollama pull nomic-embed-text`）。**不需要 Docker、不需要 Postgres、不需要开数据库服务。**

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                          frontend/                           │
│   Streamlit：登录门禁 · 对话 · 文献库 · 引用图谱 · 管理后台    │
└───────────────┬───────────────────────────────┬─────────────┘
                │                               │
        ┌───────▼────────┐              ┌───────▼────────┐
        │   backend/      │              │    agent/       │
        │ 账号 / 权限 /    │              │ LLM · Prompt ·  │
        │ 审计 / 统计 /    │              │ 会话 · RAG 链   │
        │ 评测            │              └───────┬────────┘
        └───────┬────────┘                      │
                │                       ┌───────▼────────┐
        ┌───────▼────────┐              │     rag/        │
        │  platform.db    │              │ 解析 / 切分 /    │
        │  (SQLite)       │              │ 检索 / 缓存 /    │
        └─────────────────┘              │ 图谱 / 抽取      │
                                         └───────┬────────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │ LanceDB · BM25 · Ollama │
                                    └─────────────────────────┘
```

依赖方向单向：`frontend → backend`（账号 / 权限 / 审计）与 `frontend → agent → rag`；`agent`、`rag` 不反向依赖 `backend`。

## 目录结构

```
Academic_Literature_Assistant/
├── frontend/                      # Streamlit 界面层
│   ├── app.py                     # 入口：登录门禁 + 四个页面的路由与交互
│   ├── auth_view.py               # 登录页
│   ├── admin_view.py              # 管理后台六个页签（概览/用户/知识库/评测/审计/健康）
│   ├── citation_graph_view.py     # 可缩放引用关系图（纯 SVG + JS，无第三方绘图库）
│   ├── workspace_ui.py            # 统一视觉组件：主题、页头、页面背景装饰
│   ├── title_orbits.py            # 标题星轨装饰（内联 SVG/CSS）
│   └── assets/                    # 背景图片素材（本地 base64 内联，无外链）
├── backend/                       # 平台层：账号 / 权限 / 审计 / 统计 / 评测（SQLite）
│   ├── config.py                  # APP_* 环境变量：门禁、配额、限流、保留期
│   ├── db.py                      # 平台库连接与建表（users/audit_log/qa_log/eval_runs，WAL）
│   ├── users.py                   # 账号 CRUD、scrypt 密码、角色能力表、配额
│   ├── auth.py                    # 登录校验、失败限流、会话快照
│   ├── audit.py                   # 审计日志（动作字典、筛选、CSV 导出、过期清理）
│   ├── eval.py                    # 评测中心：数据集管理、跑评测、结果落库、导出
│   └── admin.py                   # 概览指标、在线质量、知识库运维、备份/恢复、健康检查
├── agent/                         # Agent 层：LLM + Prompt + 会话
│   ├── config.py                  # provider（Groq/DeepSeek/…）、模型列表、存储开关
│   ├── llm.py                     # LLM 工厂：ChatGroq / ChatOpenAI（OpenAI 兼容）
│   ├── prompts.py                 # ChatPromptTemplate + 参考资料格式化
│   ├── rag_chain.py               # LCEL 链：prompt | llm | StrOutputParser
│   ├── storage.py                 # 会话历史 SQLite（含标题/改名/删除/备份）
│   ├── session.py                 # ChatSession：对话、入库、文献管理、历史
│   └── check_llm.py               # API Key / 模型连通性自检脚本
├── rag/                           # RAG 层：解析 + 向量化 + 检索 + 评测
│   ├── config.py                  # 数据目录、切分与检索参数
│   ├── cleaning.py                # 文本清洗：控制符/连字/伪字符/空白归一
│   ├── parsers.py                 # 版面感知 PDF 解析（PyMuPDF，pypdf 兜底）
│   ├── chunking.py                # 章节级切分 + 元数据（标题/作者/章节/页码）
│   ├── arxiv.py                   # arXiv 元数据拉取与 PDF 下载
│   ├── loaders.py                 # 输入源 → 文档（上传 PDF / arXiv）
│   ├── embedder.py                # embedding 工厂：Ollama / OpenAI
│   ├── bm25_index.py              # BM25 内存索引（中英混排分词）
│   ├── rrf.py                     # Reciprocal Rank Fusion 多路融合
│   ├── rerank.py                  # cross-encoder / LLM 重排
│   ├── query_rewrite.py           # 多轮补全 + 多查询展开
│   ├── citations.py               # 引用溯源：片段编号、[n] 解析与校验
│   ├── refusal.py                 # 拒答校准：相似度 × 关键词覆盖 → 置信度
│   ├── tools.py                   # 工具集（检索/列文献/读章节/对比/图谱/抽取表）
│   ├── tool_chain.py              # ReAct 多步工具链（带步数上限与降级）
│   ├── citation_graph.py          # 创新点1：引用图谱（networkx）+ 多跳检索
│   ├── extraction.py              # 创新点2：方法论结构化抽取 + 对比表
│   ├── lance_store.py             # LanceVectorStore：实现 LangChain VectorStore 接口
│   ├── vector_store.py            # 向量库工厂（换后端只改这里）
│   ├── libraries.py               # 多知识库注册表（名称 → 目录 / 归属 / 可见性）
│   ├── knowledge_base.py          # 知识库门面：入库、混合检索、融合、重排、retriever
│   ├── evaluation.py              # 检索评测：hit rate / recall@k / MRR
│   ├── eval_questions.example.json  # 评测数据集模板
│   └── eval_questions.sample.json   # 与 make_sample_papers 配套的样例评测集
├── scripts/
│   ├── run_experiments.py         # 消融实验矩阵（dense/hybrid/rerank）
│   ├── make_sample_papers.py      # 生成可离线跑通的样例 PDF（含干扰文献）
│   └── check_s2.py                # Semantic Scholar Key / 限流诊断
├── data/                          # 运行后自动生成（已在 .gitignore 中）
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

> 环境：Python 3.10+（3.13 已实测可用）。

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 准备环境变量
copy .env.example .env      # Windows；macOS/Linux 用 cp
#   编辑 .env：填入大模型 API Key（下面「配置说明」有最小示例）

# 3) 启动本地向量化服务（首次）
ollama pull nomic-embed-text
ollama serve                # 通常安装后已在后台运行

# 4) 启动应用
streamlit run frontend/app.py
```

浏览器打开 <http://localhost:8501>。

首次启动会自动创建一个管理员账号（用户名 `admin`）。若 `.env` 未设置 `APP_ADMIN_PASSWORD`，密码会随机生成并显示在登录页 / 启动日志里，登录后请立即修改。登录后侧栏会多出「⚙ 管理后台」。

## 配置说明

全部配置走环境变量，模板见 [`.env.example`](.env.example)。最小可用示例（DeepSeek + 本地向量化）：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的_key
DEFAULT_LLM_MODEL=deepseek-chat
LLM_MODELS=deepseek-chat,deepseek-reasoner

DEFAULT_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_HOST=http://127.0.0.1:11434
```

常用开关：

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` / `deepseek` / `openai` / `custom` | `groq` |
| `DEFAULT_EMBEDDING_MODEL` | 向量模型 | `nomic-embed-text` |
| `RAG_RETRIEVAL_MODE` | `hybrid` / `dense` / `sparse` | `hybrid` |
| `RAG_RERANK_MODE` | `auto` / `llm` / `cross-encoder` / `off` | `auto` |
| `RAG_QUERY_REWRITE` | `off` / `history` / `multi_query` / `both` | `both` |
| `RAG_REFUSAL_MIN_CONFIDENCE` | 拒答阈值 | `0.30` |
| `RAG_TOOL_MODE` | `auto` / `on` / `off` | `auto` |
| `APP_AUTH_ENABLED` | 是否启用登录门禁 | `true` |

> `.env` 已在 `.gitignore` 中，**不会被提交**。

## 功能与页面

侧栏「页面」按角色显示：

| 页面 | 能做什么 | 可见角色 |
| --- | --- | --- |
| 💬 对话 | 提问、流式回答、展开「依据文献 / 工具调用轨迹 / 检索置信度」，显示今日剩余额度 | 全部 |
| 📚 文献库 | 论文表格（年份 / 章节 / 片段 / 引用度数 / 方法画像）、查看解析详情、单篇删除、单篇重抓引用、清空全库（二次确认） | 全部（访客只读） |
| 🕸 引用图谱 | 选一篇看它的 1~2 跳引用网络（橙=中心、绿=在库、灰=PDF 参考文献、白=库外），可单篇或整库刷新 | 管理员 / 成员 |
| ⚙ 管理后台 | 概览、用户与权限、知识库运维、评测中心、审计日志、系统健康 | 仅管理员 |

侧栏「会话历史」按**第一条提问自动命名**，支持重命名和删除（二次确认）；历史按账号隔离，管理员可看全部。危险操作（清空知识库、删除文献、删库）都要求二次确认。

### 账号与角色

三档角色（能力表集中在 `backend/users.py` 的 `_CAPABILITIES`）：

| 角色 | 提问 | 导入/删除文献 | 建库/删库/清空 | 用户管理 | 后台 |
| --- | --- | --- | --- | --- | --- |
| `admin` 管理员 | ✅ | ✅ 所有库 | ✅ | ✅ | ✅ |
| `member` 成员 | ✅（默认 200 次/天） | ✅ 仅自己名下的私有库 | ❌ | ❌ | ❌ |
| `guest` 访客 | ✅（默认 30 次/天） | ❌ 只读 | ❌ | ❌ | ❌ |

隔离策略：**一个用户 → 绑定一个私有知识库**（管理后台分配），共享的「默认知识库」只有管理员可写，避免多人往演示库里互相污染。新建的知识库拥有独立的向量表、引用图与抽取库。

### 多知识库

侧栏「知识库」可切换 / 新建 / 重命名 / 删除（默认库不可删除，只能清空）。注册表存放在 `data/libraries.json`，新建库落到 `data/libraries/<slug>/`。

## 数据是怎么处理的

从「上传一个 PDF」到「能被检索」的完整链路：

| 阶段 | 位置 | 做了什么 |
| --- | --- | --- |
| 解析 | `parsers.py` | 读取每行字号 / 加粗 / 坐标，识别标题、作者、章节；丢掉页眉页脚、页码、arXiv 侧边标记、纯符号行；参考文献默认不入库 |
| 清洗 | `cleaning.py` | 去控制字符、软连字符、零宽字符、BOM；展开连字（`ﬁ`→`fi`）；清掉 `(cid:123)`；NFKC 归一；压缩空白；单节超长自动截断 |
| 分块 | `chunking.py` | 章节内先按段落、再按句子切；短片段合并；相邻块重叠；`RAG_MIN_CHUNK_CHARS` 以下的碎片直接丢弃 |
| 去重 | `chunking.py` + `loaders.py` | 文档级（`source_key` 用 PDF 内容指纹 sha1 或 arXiv ID）+ 段落级 + 片段级 |
| 编码 | `embedder.py` | `nomic-embed-text`（768 维）或 `text-embedding-3-small`（1536 维）；Ollama 报错会包成可执行提示 |
| 入库 | `lance_store.py` | 自研 `LanceVectorStore`（LangChain `VectorStore` 子类）；`upsert_documents` 按 id 先删后插；入库与检索都做 L2 归一化 |
| 检索 | `knowledge_base.py` | `search()` 返回带元数据片段，`retriever()` 返回标准 LangChain retriever |
| 评测 | `evaluation.py` + `backend/eval.py` | Hit Rate / Recall@k / MRR / 关键词命中率；后台可视化、落库、消融与导出 |

可调参数：

```dotenv
RAG_CHUNK_SIZE=1000          # 每个片段的字符数上限
RAG_CHUNK_OVERLAP=150        # 相邻片段重叠字符数
RAG_MIN_CHUNK_CHARS=40       # 小于此长度的片段丢弃（0 = 不过滤）
RAG_DEDUPE_CHUNKS=true       # 段落/片段级去重
RAG_CONTEXTUAL_HEADER=true   # 入库文本是否加「标题 + 章节」头
RAG_INCLUDE_REFERENCES=false # 参考文献是否入库
RAG_NUM_DOCUMENTS=2          # 每次提问检索几个片段
```

## 检索链路

```
用户问题
   │
   ├─(1) Query 改写（query_rewrite.py）
   │     · 多轮补全：把「那它的实验结果呢？」补成独立问题
   │     · 多问法展开：生成 2 个等价问法，提升召回
   │
   ├─(2) 双路召回（knowledge_base.py）
   │     · 稠密：LanceDB 余弦近邻
   │     · 稀疏：BM25 内存索引（中英混排分词）
   │
   ├─(3) RRF 融合（rrf.py）：score = Σ 1/(k + rank)，k 默认 60
   │
   ├─(4) Rerank（rerank.py）：cross-encoder 或 LLM 打分，取 top-N
   │
   └─(5) 拼 prompt → 流式作答（rag_chain.py）
```

每一步都能单独关掉，方便做基线对比：

```dotenv
RAG_RETRIEVAL_MODE=hybrid    # hybrid（BM25+向量+RRF）/ dense（纯向量）/ sparse（纯 BM25）
RAG_RECALL_PER_ROUTE=20      # 每一路召回多少候选
RAG_RRF_K=60                 # RRF 平滑常数
RAG_QUERY_REWRITE=both       # off / history / multi_query / both
RAG_REWRITE_NUM_QUERIES=2    # 额外生成几个问法
RAG_RERANK_MODE=auto         # off / llm / cross-encoder / auto
RAG_RERANK_MODEL=BAAI/bge-reranker-base
RAG_RERANK_TOP_N=3           # 重排后留几条进 prompt
```

几点说明：

- **BM25 是内存索引**，从向量库现有内容重建（几千片段毫秒级），不额外落盘；入库、清空时自动同步。中文用字符 bigram，英文按词，无需分词器依赖。
- **`RAG_RERANK_MODE=auto`** 只在本地已缓存 cross-encoder 时才用它，否则自动退回 LLM 重排，避免首次运行联网拉模型卡住页面。想强制用本地模型设 `cross-encoder`，并先跑一次让它下载完。
- **LLM 重排按批打分**（默认 8 个候选一次请求，输出「编号: 分数」），控制成本与延迟。
- **改写 / 重排失败都会自动降级**为原问题、原顺序，不会成为新的故障点。

## 可信作答：引用溯源 / 拒答校准 / 多步工具链

三个「让回答可信」的机制在 `agent/session.py` 的 `stream_answer` 里串起来：

```
改写 → 检索（多步工具链 或 单步混合检索）
     → 拒答校准（置信度不足直接拒答）
     → 引用溯源（片段编号 [n] → 生成 → 校验编号）
     → 输出 + 「依据文献」清单 + 可信度面板
```

### 引用溯源（`rag/citations.py`）

- 每个检索片段分配 `[n]` 编号（含《标题》· 章节 · 页码），拼进 prompt；
- prompt 要求模型在结论后标 `[n]`；
- 生成后解析回答里的所有 `[n]`（支持 `[1]` `[1,2]` `[1-3]` 全角 `［1］`）并校验：有效编号 → 命中来源；越界编号 → 告警；一条都没引 → 标记「未溯源」；
- 回答末尾自动追加「依据文献」清单，前端可展开看每条来源的片段原文。

### 拒答校准（`rag/refusal.py`）

置信度 = `sqrt(归一化相似度 × 关键词覆盖率)`（几何平均，任一为 0 即不可信）：

- 相似度先按 `[0.40, 0.95]` 线性归一化（`nomic-embed-text` 对无关文本也会给 0.6+ 的余弦）；
- 低于 `RAG_REFUSAL_MIN_CONFIDENCE`（默认 0.30）就拒答，输出统一的「文献里没有 + 三条建议」话术；
- 工具链已查证的问题（如「我上传了哪些文献」由 `list_papers` 直接给出目录）不做拒答判定。

### 多步工具链（`rag/tools.py` + `rag/tool_chain.py`）

八个工具，模型自己决定调哪个、调几次（ReAct 循环，最多 `RAG_TOOL_MAX_STEPS` 步）：

| 工具 | 用途 |
| --- | --- |
| `search_literature(query)` | 混合检索（向量 + BM25 + RRF + rerank） |
| `list_papers()` | 列出已入库文献：标题、年份、片段数、章节名 |
| `read_section(source_key, section)` | 按来源 + 章节读取正文 |
| `compare_papers(topic)` | 按主题分组检索，用于多篇横向对比 |
| `get_references(paper)` | 沿出边：它引用了谁 / 基于哪些工作 |
| `get_citations(paper)` | 沿入边：谁引用了它 / 有哪些后续改进 |
| `trace_lineage(paper, hops)` | BFS 多跳：继承了什么 + 后续演进了什么 |
| `compare_methods(papers)` | 读结构化方法画像，生成 Markdown 对比表 |

`RAG_TOOL_MODE=auto` 时只在问题明显需要多步（「上传了哪些文献」「对比两篇」「第 3 节怎么做的」）才启用，其余走单步 RAG（更快更省）。取值 `auto` / `on` / `off`，三者都能跑，便于对比。

## 创新点 1：引用图谱 + 多跳检索（`rag/citation_graph.py`）

单步 RAG 只能答「这篇论文说了什么」，答不了**「谁改进了它？它和 X 的差异在哪？」**——这类答案不在正文里，而在**引用网络**里。

做法：

1. **抽取引用关系**：优先用 Semantic Scholar API（同时有 references 与 cited-by），按 arXiv ID 或标题定位，抓引用边；
2. **建图**：`networkx.DiGraph`，边 `A → B` 表示「A 引用了 B」，节点存标题 / 年份 / 作者 / 是否在库，JSON 持久化；
3. **三个多跳工具**：`get_references` / `get_citations` / `trace_lineage`（见上表）；
4. **多跳推理**：`trace_lineage` 用 BFS 沿引用边扩展 `hops` 跳，得到「祖先（它引用的…）」与「后代（引用它的…）」，并提示模型梳理技术演进链。

```dotenv
RAG_GRAPH_ENABLED=true
RAG_GRAPH_PATH=data/citation_graph.json
RAG_GRAPH_MAX_HOPS=2
# 可选：填 key 提升 Semantic Scholar 配额
# SEMANTIC_SCHOLAR_API_KEY=...
```

### 侧栏显示「0 条引用边」是什么意思

这通常表示**引用边没有抓取到**，不是功能坏了。图在**入库时**构建（`session._post_ingest` → `CitationGraphBuilder.ingest`），依赖 Semantic Scholar：

- 有 key / 网络通畅：正常抓 references + citations，侧栏显示 `N 篇文献 · M 条引用边`；
- 无 key 且被限流（HTTP 429）：抓不到边，但系统仍会把论文登记为本地节点（`in_library=True`），显示「N 篇文献（在库 K 篇）· 0 条引用边」。

对本地 PDF，S2 只能用**标题**搜索，合成 / 未发表 / 改过标题的论文通常搜不到。想拿到真实引用边，最稳的做法是**按 arXiv ID 导入**。当图里只有本地节点时，多跳工具会明确回复「已登记，但引用关系未抓取到」，不会乱答。

补全步骤：

1. 申请免费 S2 key（<https://www.semanticscholar.org/product/api#api-key-form>），在 `.env` 写 `SEMANTIC_SCHOLAR_API_KEY=...`；
2. 重启应用，**重新导入**该论文（arXiv ID 优先）；
3. 之前登记的本地节点会在抓取成功时自动合并。

## 创新点 2：方法论结构化抽取 + 自动对比表（`rag/extraction.py`）

研究者最费时间的事：读 10 篇论文做一张方法对比表。难点不在「生成」，而在**稳定地生成合法结构**：

1. **JSON schema 约束**：抽取 `{方法名, 核心思想, 数据集, 评测指标, 参数量, 优点, 局限}`；
2. **解析容错**：剥离 ` ```json ` 围栏、截取 `{...}`；缺必填字段视为失败；
3. **失败重试**：`RAG_EXTRACTION_MAX_RETRIES` 次，仍失败不阻塞入库；
4. **结构化存储**：SQLite 一篇论文一行，重复抽取覆盖；
5. **对比表工具** `compare_methods(papers)`：直接读结构化表生成 Markdown 对比表，**交付物是一张表而不是一段话**。

```dotenv
RAG_EXTRACTION_AUTO=true
RAG_EXTRACTION_PATH=data/extraction.db
RAG_EXTRACTION_MAX_RETRIES=2
RAG_EXTRACTION_MAX_CHARS=6000
```

对比表示例：

| 对比维度 | ResNet（2015） | DenseNet（2016） |
| --- | --- | --- |
| 核心思想 | 残差连接 | 密集连接 |
| 数据集 | ImageNet | CIFAR |
| 评测指标 | top-1 | error rate |
| 参数量 | 25M | 8M |
| 优点 | 可训练极深 | 参数更少 |
| 局限 | 算力需求高 | 显存占用大 |

> 没有引入 LangGraph：这条链路本质是 ReAct 循环，原生 tool-calling 更透明。真要做多 Agent 协作再换。

## 检索评测

RAG 的效果八成取决于检索，项目提供可重复的检索评测脚本。

```bash
# 纯向量基线
python -m rag.evaluation --mode dense --no-rerank

# 混合检索 + RRF
python -m rag.evaluation --mode hybrid --no-rerank

# 混合 + cross-encoder 重排（推荐配置）
python -m rag.evaluation --mode hybrid --rerank cross-encoder

# 输出每个问题的命中片段 / 写 JSON 报告
python -m rag.evaluation --dataset questions.json --top-k 5 --dump
python -m rag.evaluation --dataset questions.json --top-k 5 --json report.json
```

问题集格式（`expected_sources` 支持包含匹配，可填 `source_key` / 文件名 / 标题）：

```json
{
  "cases": [
    {
      "question": "这篇论文的核心贡献是什么？",
      "expected_sources": ["arxiv:2401.00001"],
      "expected_keywords": ["retrieval", "embedding"]
    }
  ]
}
```

| 指标 | 分母 | 说明 |
| --- | --- | --- |
| Hit Rate | **正样本**（`expected_sources` 非空） | top-k 内命中任一期望来源的题占比 |
| Recall@k | 正样本 | 期望来源被召回的比例（多来源按覆盖率计） |
| MRR | 正样本 | 首个命中名次的倒数均值 |
| 关键词命中率 | **写了 `expected_keywords` 的题** | 没写关键词的题不参与，避免「空关键词=满分」 |
| 检索精确率 | 正样本 | top-k 片段里来自期望来源的比例 |
| 负样本 | — | `expected_sources` 为空：只用于拒答评估，**不参与**以上指标 |

> 指标全是 0 时，多半是 `expected_sources` 写错了：上传的 PDF 用 `pdf:xxxxxxxxxxxx`，arXiv 用 `arxiv:2401.00001`。用 `--dump` 看实际命中的来源再改。

### 后台「🧪 评测中心」

不用命令行也能跑评测，结果落库、可复现：

- **数据集**：内置 `rag/eval_questions*.json`，也可上传（字段校验 / 模板下载 / 原始 JSON 下载）；
- **数据集体检**：逐条检查 `expected_sources` 能否在所选知识库找到，找不到会列出可用 `source_key`；
- **自动生成数据集**：按文献抽样出题（模板模式零成本 / LLM 模式更自然），可附带负样本；
- **生成并运行**：一键出题 + 跑评测 + 写入历史；
- **配置**：知识库、`dense/hybrid`、`off/cross-encoder/llm`、top-k、题数上限；
- **历史 / 消融 / 导出**：按次查看、一键跑「纯向量 / 混合 / 混合+重排」三组对比、CSV 导出、每条记录带复现命令。

## 完整实验流程

```bash
# 第 0 步：准备文献（四选一）
# A. 库里已有论文（界面里上传过），直接跑
# B. 导入 arXiv
python -m scripts.run_experiments --arxiv 1512.03385 2016.14248 --dataset my.json
# C. 用本地 PDF
python -m scripts.run_experiments --pdf papers/a.pdf papers/b.pdf --dataset my.json
# D. 没有真实论文时，造一组样例（含干扰文献，保证消融有区分度）
python scripts/make_sample_papers.py --with-distractors

# 第 1 步：准备评测集
#   参照 rag/eval_questions.example.json（模板）或 rag/eval_questions.sample.json（配套样例）
#   每题写 question / expected_sources / expected_keywords，务必放几条负样本

# 第 2 步：跑消融矩阵
python -m scripts.run_experiments --dataset rag/eval_questions.sample.json --top-k 3
python -m scripts.run_experiments --dataset my.json --top-k 3 \
    --configs "baseline:dense:off,hybrid:hybrid:off"
python -m scripts.run_experiments --dataset my.json --top-k 3 --refusal-sweep
```

输出两张 Markdown 表 + JSON（`data/experiments/`），可直接贴论文或画图。

### 怎么让消融结果有区分度

小语料（<20 个片段）下 dense 与 hybrid 会几乎一致，因为没有干扰项。要让差异显出来：

1. **加干扰文献**（`--with-distractors` 已内置一批同主题论文 + 一篇大量提 BM25/RRF 的干扰项）；
2. **`--top-k` 调小到 1~3**，召回压力才体现；
3. **评测集放「精确术语」类问题**（RRF、k=60、某参数值），BM25 才有发挥空间；
4. 语料越大（几十篇以上），混合检索与重排的收益越明显。

## 切换大模型 / 向量库

### 换大模型（不改代码）

`LLM_PROVIDER` 决定用哪家接口，`DEFAULT_LLM_MODEL` / `LLM_MODELS` 里只填**模型名**：

| `LLM_PROVIDER` | 需要的环境变量 | 模型名举例 |
| --- | --- | --- |
| `groq`（默认） | `GROQ_API_KEY` | `llama3-70b-8192` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `custom` | `LLM_API_KEY` + `LLM_BASE_URL` | 任意 OpenAI 兼容模型 |

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的_key
DEFAULT_LLM_MODEL=deepseek-chat
LLM_MODELS=deepseek-chat,deepseek-reasoner
```

- 不要在这里填网址（会被当成模型名）；网址只由 `LLM_PROVIDER` 或 `LLM_BASE_URL` 指定。
- DeepSeek 没有 embedding 接口，向量化固定用本地 Ollama 或 OpenAI。
- 改完 `.env` 重启 Streamlit；自检可用 `python -m agent.check_llm deepseek-chat`。

### 换向量库（pgvector 等）

`rag/vector_store.py` 是唯一的切换点：只要返回一个实现了 `langchain_core.vectorstores.VectorStore` 的对象即可（`similarity_search` + `add_documents`），上层的知识库、对话链、评测脚本都不用改。

```python
# rag/vector_store.py
from langchain_postgres import PGVector

def get_vector_store(embeddings_model, embedding):
    return PGVector(
        embeddings=embedding,
        collection_name=get_collection_name(embeddings_model),
        connection="postgresql+psycopg://ai:ai@localhost:5532/ai",
        use_jsonb=True,
    )
```

## 常见问题

- **界面提示「初始化失败：未配置 XXX_API_KEY」**：`.env` 里没填当前 provider 对应的 key（groq 用 `GROQ_API_KEY`，deepseek 用 `DEEPSEEK_API_KEY`）。
- **上传时提示「本地 Ollama 未启动」**：运行 `ollama serve`；首次还要 `ollama pull nomic-embed-text`。可用 `curl http://127.0.0.1:11434/api/tags` 看模型列表。
- **上传成功但回答没引用文献**：`RAG_NUM_DOCUMENTS` 默认检索 2 个片段，可调大；确认 PDF 不是扫描件（无文字层）。
- **导入 arXiv 失败（HTTP 502 / SSL 报错）**：可能是 arXiv 超载，也可能出口网关拦截了 `export.arxiv.org`。代码已做 https→http 双通道 + 5xx 重试；反复失败可在 `.env` 配 `ARXIV_PROXY=http://127.0.0.1:7890`（只代理 arXiv）。
- **抓引用关系很慢 / 日志刷 429**：Semantic Scholar 的 key 额度是「1 请求/秒，全端点累计」。代码里所有 S2 请求共用一道全局限速器（有 key 默认 1.1s/请求，无 key 3.1s），刷新 N 篇约需 `2N × 1.1` 秒，属正常排队。
- **识别出的章节不准**：`parsers.py` 是基于字号与章节名规则的启发式解析，排版特殊的论文可能漏识别；可用 `--dump` 看切片确认。
- **评测结果全是 0**：多半是 `expected_sources` 写错，用 `--dump` 看实际命中的来源再改。
- **没有历史对话**：`ENABLE_ASSISTANT_STORAGE=false` 时关闭；否则历史在 `data/assistant.db`。
- **想彻底重置知识库**：界面「清空知识库」，或删除 `data/lancedb/`（`data/` 均为本地数据，不影响代码）。
- **控制台刷 `OSError: [WinError 1114] ... c10.dll`**：Windows 上 `pyarrow` 自带的旧版 `msvcp140.dll` 会抢先加载，导致之后 `torch` 初始化失败（仅本地 cross-encoder 重排用到 torch）。可把 pyarrow 里那份旧 CRT 改名：

```powershell
Rename-Item "$env:CONDA_PREFIX\Lib\site-packages\pyarrow\msvcp140.dll" "msvcp140.dll.bak"
```

  或把「重排方式」选 `llm` / `off` 避开。
- **启动时一直打印 `Examining the path of transformers...`**：这是 Streamlit 文件监视器在遍历已导入模块，不是报错；可加 `--server.fileWatcherType none` 静默（代价是改代码不自动热重载）。

## 安全提示

- **不要提交 `.env`**：里面是真实的 API Key 与管理员密码。仓库已通过 `.gitignore` 忽略 `.env`、`data/`、`*.db`。
- 如果你曾把 `.env` 或密钥推到公开平台，请**立即在对应平台重置 / 吊销 Key**。
- 公网演示时保持 `APP_AUTH_ENABLED=true`，并修改默认管理员密码。

---

如果这个项目对你有帮助，欢迎 Star ⭐ 与提交 Issue / PR。
