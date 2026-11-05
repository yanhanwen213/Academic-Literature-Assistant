"""RAG 链：把「Prompt 模板 → 聊天模型 → 文本解析」串成一条 LCEL 流水线。

用官方写法：

    chain = prompt | llm | StrOutputParser()
    for chunk in chain.stream({"context": ..., "question": ..., "chat_history": ...}):
        ...

检索在调用方（ChatSession）里显式执行，好处是能顺手拿到命中的片段做引用展示；
如果哪天想改成「链内检索」，把 `RunnablePassthrough.assign(context=retriever | format_documents)`
接到 `RAG_PROMPT` 前面即可，其它代码不用动。
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from agent.prompts import RAG_PROMPT


def build_rag_chain(llm: BaseChatModel) -> Runnable:
    """输入 {context, question, chat_history}，输出回答文本。"""
    return RAG_PROMPT | llm | StrOutputParser()


def build_chain_with_retriever(llm: BaseChatModel, retriever: Any) -> Runnable:
    """可选写法：检索也放进链里（输入 {question, chat_history}）。"""
    from langchain_core.runnables import RunnablePassthrough

    from agent.prompts import format_documents

    return (
        RunnablePassthrough.assign(
            context=(lambda data: data["question"]) | retriever | format_documents
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
