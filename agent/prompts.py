"""Prompt 模板与上下文格式化。

RAG 的「拼 prompt」在这里完成：`RAG_PROMPT` 接收三个变量——
`context`（带编号的文献片段）、`chat_history`（历史对话）、`question`（本次提问）。

上下文里的每个片段都带 `[n]` 编号，要求模型在结论后标 `[n]`，
这样回答可以被 `rag/citations.py` 精确校验并映射回章节页码。
"""

from typing import Iterable, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """你是一名严谨的学术文献助手，负责根据用户上传的论文回答问题。

参考资料（按相关度排序，每条的 [编号] 就是引用标识）：
{context}

回答要求：
1. 只依据参考资料作答，关键结论后用 [编号] 标注出处，例如「命中率提升到 85%[2]」；
2. 引用编号必须来自上面资料里真实存在的编号，不要编造；
3. 参考资料不足以回答时，明确说明「提供的文献中没有提到」，不要凭印象补充；
4. 使用与用户提问相同的语言回答；
5. 结构清晰，必要时用要点或分步骤说明。"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ]
)


def format_documents(documents: Iterable[Document]) -> str:
    """把检索结果拼成带 [n] 编号的参考资料文本。"""
    blocks: List[str] = []
    for index, document in enumerate(documents, start=1):
        meta: dict = document.metadata or {}
        title = meta.get("title") or meta.get("source") or "未知文献"
        header = f"[{index}] 《{title}》"
        if meta.get("section"):
            header += f" · {meta['section']}"
        if meta.get("page"):
            header += f" · p.{meta['page']}"
        blocks.append(f"{header}\n{(document.page_content or '').strip()}")

    if not blocks:
        return "（本次没有检索到与问题相关的文献片段）"
    return "\n\n".join(blocks)


def format_sources(documents: Iterable[Document]) -> str:
    """把检索结果拼成简短来源清单（用于最终回答的「依据」区）。"""
    lines: List[str] = []
    for index, document in enumerate(documents, start=1):
        meta: dict = document.metadata or {}
        title = meta.get("title") or meta.get("source") or "未知文献"
        parts = [f"- [{index}] 《{title}》"]
        if meta.get("section"):
            parts.append(str(meta["section"]))
        if meta.get("page"):
            parts.append(f"p.{meta['page']}")
        lines.append(" · ".join(parts))
    return "\n".join(lines)