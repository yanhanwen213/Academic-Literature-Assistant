"""LLM 工厂：按 `LLM_PROVIDER` 构造 LangChain 的聊天模型。

填的是模型名（如 `deepseek-chat` / `llama3-70b-8192`），接口地址由 provider 决定；
`LLM_PROVIDER=custom` 时用 `LLM_BASE_URL` 指向任意 OpenAI 兼容服务。
"""

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agent.config import LLM_PROFILE, LLM_PROVIDER, LLM_PROVIDER_LABEL

# 这些占位符说明用户复制了模板但没填自己的 key
_PLACEHOLDER_MARKERS = (
    "在这里粘贴",
    "your_",
    "your-",
    "sk-xxx",
    "sk-填",
    "填这里",
    "changeme",
    "todo",
)


def resolve_api_key() -> Optional[str]:
    env_name = LLM_PROFILE["api_key_env"]
    return (os.getenv(env_name) or "").strip() or None


def resolve_base_url() -> Optional[str]:
    env_name = LLM_PROFILE.get("base_url_env")
    configured = (os.getenv(env_name) or "").strip() if env_name else ""
    return configured or LLM_PROFILE.get("default_base_url")


def build_llm(model: Optional[str], temperature: float = 0.2) -> BaseChatModel:
    """构造聊天模型；缺少密钥 / 地址时抛 ValueError，便于前端看到明确提示。"""
    if not model:
        raise ValueError("未指定模型名")

    api_key = resolve_api_key()
    if not api_key:
        raise ValueError(
            f"未配置 {LLM_PROFILE['api_key_env']}，请在 .env 中填写后再使用 {LLM_PROVIDER_LABEL} 模型"
        )
    if any(marker in api_key.lower() for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(
            f"{LLM_PROFILE['api_key_env']} 还是示例值，请把 .env 里的这一行换成你自己的 "
            f"{LLM_PROVIDER_LABEL} API Key"
        )

    if LLM_PROFILE["kind"] == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=api_key, temperature=temperature)

    base_url = resolve_base_url()
    if LLM_PROVIDER == "custom" and not base_url:
        raise ValueError("LLM_PROVIDER=custom 时必须配置 LLM_BASE_URL")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
