"""LLM 连通性自检：验证 API Key 与模型是否可用。

用法（项目根目录）：
    python -m agent.check_llm
    python -m agent.check_llm deepseek-reasoner
"""


from agent.config import (
    DEFAULT_LLM_MODEL,
    LLM_PROVIDER_LABEL,
)
from agent.llm import build_llm


def main() -> int:
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LLM_MODEL
    try:
        llm = build_llm(model)
        response = llm.invoke("用一句话解释什么是 RAG（检索增强生成）。")
    except Exception as exc:  # noqa: BLE001 - 直接把原因展示给使用者
        print(f"调用失败：{exc}")
        return 1

    content = getattr(response, "content", None) or str(response)
    print(f"[{LLM_PROVIDER_LABEL}] {model} 可用：\n{content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
