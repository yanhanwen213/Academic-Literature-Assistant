"""Agent 层的可配置项，统一从环境变量读取。

大模型通过 `LLM_PROVIDER` 选择「用哪家的接口」，通过 `LLM_MODELS` /
`DEFAULT_LLM_MODEL` 选择「具体哪个模型名」。注意：模型配置里填的是**模型名**，
接口网址由 provider 决定（Groq / DeepSeek / OpenAI 各自固定，custom 由
`LLM_BASE_URL` 指定）。
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# 无论从哪个目录启动，都固定读取项目根目录的 .env（utf-8-sig 兼容带 BOM 的文件）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", encoding="utf-8-sig")

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# --------------------------------------------------------------------- 大模型
# kind:
#   groq             -> LangChain 的 ChatGroq 客户端
#   openai_compatible -> 任何兼容 OpenAI Chat Completions 的服务（DeepSeek 等）
LLM_PROFILES: Dict[str, Dict[str, Any]] = {
    "groq": {
        "label": "Groq",
        "kind": "groq",
        "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        "api_key_env": "GROQ_API_KEY",
        "base_url_env": None,
        "default_base_url": None,
    },
    "deepseek": {
        "label": "DeepSeek",
        "kind": "openai_compatible",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
    },
    "openai": {
        "label": "OpenAI",
        "kind": "openai_compatible",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": None,
    },
    "custom": {
        "label": "自定义 OpenAI 兼容接口",
        "kind": "openai_compatible",
        "models": ["deepseek-chat"],
        "api_key_env": "LLM_API_KEY",
        "base_url_env": "LLM_BASE_URL",
        "default_base_url": None,
    },
}

LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
if LLM_PROVIDER not in LLM_PROFILES:
    logger.warning(
        "未知的 LLM_PROVIDER=%s，已回退为 groq（可选：%s）",
        LLM_PROVIDER,
        ", ".join(LLM_PROFILES),
    )
    LLM_PROVIDER = "groq"

LLM_PROFILE: Dict[str, Any] = LLM_PROFILES[LLM_PROVIDER]
LLM_PROVIDER_LABEL: str = LLM_PROFILE["label"]

# 前端下拉框里的模型名列表；切换 provider 后默认列表会随之变化
LLM_MODELS = _env_list("LLM_MODELS", ",".join(LLM_PROFILE["models"]))
if not LLM_MODELS:
    LLM_MODELS = list(LLM_PROFILE["models"])

# DEFAULT_LLM_MODEL 必须落在可选列表里，否则回退到第一个
_configured_default_model = (os.getenv("DEFAULT_LLM_MODEL") or "").strip()
DEFAULT_LLM_MODEL = (
    _configured_default_model
    if _configured_default_model in LLM_MODELS
    else LLM_MODELS[0]
)

# 是否把每轮对话持久化到 data/assistant.db（关闭后无法查看 / 恢复历史会话）
ENABLE_STORAGE = _env_bool("ENABLE_ASSISTANT_STORAGE", True)

# 有些模型不支持「函数调用 / tools」，此时多步工具链会静默降级为单步检索。
# 前端据此给出明确提示，避免误以为是引用图谱坏了。
_TOOL_UNSUPPORTED_HINTS = ("deepseek-reasoner",)


def model_supports_tools(model: str) -> bool:
    name = (model or "").strip().lower()
    return not any(hint in name for hint in _TOOL_UNSUPPORTED_HINTS)

# 打开后输出更详细的调试日志
DEBUG_MODE = _env_bool("AGENT_DEBUG", False)

# 带给 LLM 的历史消息条数
NUM_HISTORY_MESSAGES = int(os.getenv("NUM_HISTORY_MESSAGES", "4"))

# 视觉模型可以单独指定 Key / 接口地址（例如文本用 DeepSeek，图片用 OpenAI 兼容视觉服务）
VISION_API_KEY = (os.getenv("RAG_VISION_API_KEY") or "").strip()
VISION_BASE_URL = (os.getenv("RAG_VISION_BASE_URL") or "").strip()

WELCOME_MESSAGE = "先上传 PDF 文献或导入 arXiv 论文，然后就可以开始提问了。"

# 前端提示语：切换 embedding 模型后需要重新入库
EMBEDDINGS_CHANGED_HINT = "已切换 embedding 模型，请重新上传文献。"
