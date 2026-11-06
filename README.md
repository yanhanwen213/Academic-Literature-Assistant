# 学术文献智能助手

一个面向论文阅读的 RAG 应用。上传 PDF 或导入 arXiv 后，系统会解析正文、按章节切分并建立索引；提问时先检索相关片段，再由大模型生成带 `[n]` 引用的回答。除问答外，还提供引用图谱、方法对比表、图表理解和检索评测。

界面用 Streamlit，编排用 LangChain，向量库用 LanceDB，embedding 默认走本地 Ollama。除 Ollama 外不需要额外的数据库或 Docker。

## 主要功能

- **文献导入**：支持上传 PDF 和按 arXiv ID 导入，自动识别标题、作者、章节。
- **版面感知解析**：根据字号、加粗和坐标识别章节，过滤页眉页脚、页码和参考文献。
- **混合检索**：向量检索与 BM25 双路召回，RRF 融合，可选 cross-encoder 或 LLM 重排。
- **引用溯源**：回答中的 `[n]` 会校验并对应到具体章节和页码；检索结果不可靠时直接拒答。
- **多步工具链**：模型可按需调用检索、列文献、读章节、对比论文、查引用图谱、生成方法对比表、解释图表等工具。
- **多模态**：输入框可直接附加图片，图片会和检索到的文献一起交给视觉模型；语音先转写再进入问答流程。
- **引用图谱**：基于 Semantic Scholar 构建引用关系图，支持查看参考文献、被引情况和多跳演进。
- **方法抽取**：把论文方法整理成结构化数据，并生成 Markdown 对比表。
- **多知识库与账号权限**：支持管理员、成员、访客三种角色，提供独立知识库、审计日志和用量统计。
- **检索评测**：提供 Hit Rate、Recall@k、MRR、关键词命中率和检索精确率，可在后台跑消融对比。

## 技术栈

| 模块 | 选型 |
| --- | --- |
| 前端 | Streamlit 1.43+ |
| 编排 | LangChain 1.x |
| 向量库 | LanceDB |
| Embedding | Ollama `nomic-embed-text` 或 OpenAI |
| 稀疏检索 | `rank_bm25` |
| PDF 解析 | PyMuPDF，pypdf 兜底 |
| 多模态 | OpenAI 兼容视觉接口、Whisper 兼容语音转写 |
| MCP | `mcp` Server（可选）、`langchain-mcp-adapters` 客户端（可选） |
| 图谱 | networkx、Semantic Scholar API |
| 存储 | SQLite |

## 快速开始

环境要求：Python 3.10+，Node.js 仅在需要外部 MCP 工具时才用到。

```bash
# 安装依赖
pip install -r requirements.txt

# 准备环境变量
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux

# 编辑 .env，填入大模型 API Key
# 启动本地 embedding 服务
ollama pull nomic-embed-text
ollama serve

# 启动应用
streamlit run frontend/app.py
```

浏览器打开 <http://localhost:8501>。

首次启动会创建管理员账号 `admin`。如果 `.env` 没有设置 `APP_ADMIN_PASSWORD`，密码会随机生成并显示在登录页或启动日志里，登录后请及时修改。

## 配置

配置全部通过环境变量读取，模板在 `.env.example`。最小可用配置示例：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的_key
DEFAULT_LLM_MODEL=deepseek-chat
LLM_MODELS=deepseek-chat,deepseek-reasoner

DEFAULT_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_HOST=http://127.0.0.1:11434
```

常用配置项：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` / `deepseek` / `openai` / `custom` | `groq` |
| `DEFAULT_EMBEDDING_MODEL` | 向量模型 | `nomic-embed-text` |
| `RAG_RETRIEVAL_MODE` | `hybrid` / `dense` / `sparse` | `hybrid` |
| `RAG_RERANK_MODE` | `auto` / `llm` / `cross-encoder` / `off` | `auto` |
| `RAG_QUERY_REWRITE` | `off` / `history` / `multi_query` / `both` | `both` |
| `RAG_REFUSAL_MIN_CONFIDENCE` | 拒答阈值 | `0.30` |
| `RAG_TOOL_MODE` | `auto` / `on` / `off` | `auto` |
| `RAG_STORE_PDF` | 是否保存上传的 PDF，用于图表理解 | `true` |
| `APP_AUTH_ENABLED` | 是否启用登录 | `true` |

`.env` 已在 `.gitignore` 中，不会被提交。

## 目录结构

```
Academic_Literature_Assistant/
├── frontend/                 # Streamlit 页面
│   ├── app.py                # 入口、登录、对话与文献库
│   ├── admin_view.py         # 管理后台
│   ├── auth_view.py          # 登录页
│   ├── workspace_ui.py       # 页面主题与背景
│   ├── citation_graph_view.py
│   ├── title_orbits.py
│   └── assets/               # 本地背景图
├── backend/                  # 账号、权限、审计、评测
├── agent/                    # LLM、Prompt、会话
├── rag/                      # 解析、切分、检索、图谱、多模态
├── scripts/                  # 实验脚本、样例数据、MCP Server
├── data/                     # 运行时数据，已忽略
├── requirements.txt
├── .env.example
└── README.md
```

各层职责：

- `frontend/` 只负责页面和交互，通过 `ChatSession` 调用下层能力。
- `backend/` 处理账号、权限、审计、用量和评测记录。
- `agent/` 负责模型选择、Prompt、多轮对话和检索调用。
- `rag/` 负责 PDF / arXiv 解析、切分、向量化、检索、图谱和抽取。

## 使用说明

### 页面

| 页面 | 用途 | 可见角色 |
| --- | --- | --- |
| 对话 | 提问、流式回答、查看引用、工具调用和检索置信度 | 全部 |
| 文献库 | 查看论文、解析详情、删除文献、清空知识库 | 全部，访客只读 |
| 引用图谱 | 查看 1 到 2 跳引用关系 | 管理员、成员 |
| 管理后台 | 概览、用户、知识库、评测、审计、系统健康 | 管理员 |

### 角色

| 角色 | 提问 | 导入 / 删除文献 | 建库 / 删库 | 用户管理 | 后台 |
| --- | --- | --- | --- | --- | --- |
| 管理员 | 是 | 所有库 | 是 | 是 | 是 |
| 成员 | 是，默认 200 次/天 | 仅自己名下的私有库 | 否 | 否 | 否 |
| 访客 | 是，默认 30 次/天 | 否，只读 | 否 | 否 | 否 |

### 文献处理流程

1. 解析 PDF，识别标题、作者和章节，过滤页眉页脚与参考文献。
2. 清理文本，处理连字、控制字符和空白。
3. 在章节内按段落和句子切分，合并过短片段，相邻片段保留重叠。
4. 使用 embedding 模型编码后写入 LanceDB。
5. 提问时执行混合检索、RRF 融合和可选重排，再把片段拼进 Prompt。

相关参数：

```dotenv
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_MIN_CHUNK_CHARS=40
RAG_DEDUPE_CHUNKS=true
RAG_CONTEXTUAL_HEADER=true
RAG_INCLUDE_REFERENCES=false
RAG_NUM_DOCUMENTS=2
```

### 检索流程

```
用户问题
  → Query 改写（多轮补全、多问法展开）
  → 向量检索 + BM25 检索
  → RRF 融合
  → cross-encoder / LLM 重排
  → 生成回答并校验 [n] 引用
```

每一步都可以单独关闭，方便做基线对比：

```dotenv
RAG_RETRIEVAL_MODE=hybrid
RAG_RECALL_PER_ROUTE=20
RAG_RRF_K=60
RAG_QUERY_REWRITE=both
RAG_REWRITE_NUM_QUERIES=2
RAG_RERANK_MODE=auto
RAG_RERANK_MODEL=BAAI/bge-reranker-base
RAG_RERANK_TOP_N=3
```

`RAG_RERANK_MODE=auto` 只在本地已经缓存 cross-encoder 模型时使用它，否则退回 LLM 重排，避免第一次运行联网下载模型卡住页面。

### 引用与拒答

- 每个检索片段分配 `[n]` 编号，回答要求标注引用。
- 生成后解析并校验编号，越界或未引用会给出提示。
- 置信度由相似度和关键词覆盖率计算，低于阈值时直接说明文献中没有相关信息。
- 工具链已经查证的问题，例如列出已上传文献，不做拒答判断。

### 多模态

- 输入框可以直接附加图片，提交后自动交给视觉模型，不需要额外点击“附加到下一次提问”。
- 上传的 PDF 默认保存一份，用于后续提取图表。
- 文献库中的“图表理解”可以按页查看图片，并让视觉模型解释。
- 语音先通过 Whisper 兼容接口转写，再进入同一条问答流程。

```dotenv
RAG_MULTIMODAL_ENABLED=true
# RAG_VISION_MODEL=gpt-4o
RAG_MULTIMODAL_MAX_IMAGES=3
RAG_STORE_PDF=true

RAG_ASR_MODEL=whisper-1
RAG_ASR_BASE_URL=https://api.openai.com/v1
```

### 引用图谱

引用图谱在导入文献时构建，数据来源是 Semantic Scholar。有 API Key 时会抓取参考文献和被引关系；没有 Key 或触发限流时，只登记本地节点，不生成引用边。

```dotenv
RAG_GRAPH_ENABLED=true
RAG_GRAPH_PATH=data/citation_graph.json
RAG_GRAPH_MAX_HOPS=2
# SEMANTIC_SCHOLAR_API_KEY=...
```

本地 PDF 只能按标题搜索，合成论文或改过标题的论文通常找不到引用关系。需要准确引用边时，建议按 arXiv ID 导入。

### 方法抽取

系统会把论文方法整理成结构化数据，包括方法名、核心思想、数据集、指标、参数量、优点和局限。抽取失败不会影响文献入库，后续可以通过对比工具生成 Markdown 表格。

```dotenv
RAG_EXTRACTION_AUTO=true
RAG_EXTRACTION_PATH=data/extraction.db
RAG_EXTRACTION_MAX_RETRIES=2
RAG_EXTRACTION_MAX_CHARS=6000
```

## 检索评测

命令行评测：

```bash
# 纯向量
python -m rag.evaluation --mode dense --no-rerank

# 混合检索
python -m rag.evaluation --mode hybrid --no-rerank

# 混合检索 + cross-encoder 重排
python -m rag.evaluation --mode hybrid --rerank cross-encoder

# 查看每个问题命中的片段
python -m rag.evaluation --dataset questions.json --top-k 5 --dump
```

评测集格式：

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

`expected_sources` 支持 `source_key`、文件名或标题。上传的 PDF 一般使用 `pdf:xxxxxxxxxxxx`，arXiv 使用 `arxiv:2401.00001`。

指标说明：

| 指标 | 说明 |
| --- | --- |
| Hit Rate | top-k 内命中期望来源的题目比例 |
| Recall@k | 期望来源被召回的比例 |
| MRR | 首个命中名次的倒数均值 |
| 关键词命中率 | 只统计填写了 `expected_keywords` 的题目 |
| 检索精确率 | top-k 片段中来自期望来源的比例 |
| 负样本 | `expected_sources` 为空的题目，只用于拒答评估 |

管理后台也提供评测页面，可以选择数据集、知识库、检索模式、重排方式和 top-k，运行后结果会写入历史记录，并支持导出 CSV。

## 切换模型

`LLM_PROVIDER` 决定使用哪家接口：

| `LLM_PROVIDER` | 环境变量 | 模型名示例 |
| --- | --- | --- |
| `groq` | `GROQ_API_KEY` | `llama3-70b-8192` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `custom` | `LLM_API_KEY`、`LLM_BASE_URL` | 任意 OpenAI 兼容模型 |

模型名只填名称，不要填接口地址。DeepSeek 没有 embedding 接口，向量化仍使用本地 Ollama 或 OpenAI。

修改 `.env` 后重启 Streamlit。可以用下面的命令检查模型是否连通：

```bash
python -m agent.check_llm deepseek-chat
```

## 常见问题

**界面提示没有配置 API Key**

检查 `.env` 中当前 provider 对应的变量名是否正确，例如 DeepSeek 使用 `DEEPSEEK_API_KEY`，Groq 使用 `GROQ_API_KEY`。

**上传时提示 Ollama 未启动**

先运行 `ollama serve`，首次还要执行 `ollama pull nomic-embed-text`。可以用下面的命令确认模型列表：

```bash
curl http://127.0.0.1:11434/api/tags
```

**回答没有引用文献**

可能是 PDF 没有文字层，或者检索片段数量太少。可以调大 `RAG_NUM_DOCUMENTS`，并确认上传的不是扫描件。

**arXiv 导入失败**

可能是 arXiv 接口超时或被网络拦截。代码已经做了 https 和 http 双通道以及重试。如果仍然失败，可以在 `.env` 中配置只对 arXiv 生效的代理：

```dotenv
ARXIV_PROXY=http://127.0.0.1:7890
```

**引用关系抓取很慢，日志里出现 429**

Semantic Scholar 的匿名额度较低。所有请求共用全局限速，有 Key 时约 1.1 秒一次，无 Key 时约 3.1 秒一次。刷新多篇论文需要排队等待。

**评测结果全是 0**

通常是 `expected_sources` 写错。用 `--dump` 查看实际命中的来源，再修改评测集。

**没有历史对话**

检查 `ENABLE_ASSISTANT_STORAGE` 是否为 `false`。历史数据默认在 `data/assistant.db`。

**想重置知识库**

可以在界面里清空知识库，或者删除 `data/lancedb/`。`data/` 是运行时数据，不影响源码。

**Windows 下出现 `c10.dll` 相关错误**

这是 pyarrow 自带的旧版运行库抢先加载导致的，只影响本地 cross-encoder 重排。可以把 pyarrow 里的旧文件改名：

```powershell
Rename-Item "$env:CONDA_PREFIX\Lib\site-packages\pyarrow\msvcp140.dll" "msvcp140.dll.bak"
```

或者把重排方式改成 `llm` 或 `off`。

## 注意事项

- 不要提交 `.env`，里面包含真实的 API Key 和管理员密码。
- 如果曾经把密钥推到公开仓库，请立即在对应平台重置。
- 公网演示时保持 `APP_AUTH_ENABLED=true`，并修改默认管理员密码。
- `data/` 目录不会上传到 Git，包含向量库、会话、账号和评测记录。
