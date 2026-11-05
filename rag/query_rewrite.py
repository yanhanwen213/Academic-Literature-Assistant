"""Query 改写：多轮补全 + 多查询变体。

解决两个真实问题：

1. 追问失效 —— 用户问「那它的实验部分呢？」，单路检索拿这句话去做向量检索，
   和文献里的正文几乎不匹配。必须结合对话历史补全成完整问题；
2. 单次改写有随机性 —— 一次生成 2~3 个等价问法，各自检索再融合，召回更全。

失败时必须降级为原问题：改写是「增强」，不能成为新的故障点。
"""

import logging
import re
from typing import List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# - SystemMessagePromptTemplate -> '你是一个{role}，只回答一句话。'
# - HumanMessagePromptTemplate  -> '{question}'
_HISTORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你负责把用户的追问改写成可以独立检索的完整问题。\n"
            "规则：\n"
            "1. 只输出改写后的问题本身，不要解释、不要加引号；\n"
            "2. 补全省略的主语和指代（它 / 该论文 / 这个方法），保留原问题的语言；\n"
            "3. 如果原问题已经完整独立，原样输出；\n"
            "4. 不要编造原文里没有的术语。",
        ),
        (
            "human",
            "对话历史：\n{history}\n\n用户追问：{question}\n\n改写后的问题：",
        ),
    ]
)

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你负责为一个学术文献检索系统生成多个等价问法。\n"
            "要求：\n"
            "1. 每行一个，不要编号、不要解释；\n"
            "2. 保留关键的专有名词、缩写、方法名（它们决定了检索精度）；\n"
            "3. 可以用同义词、更具体的表述、英文/中文互换来增加召回；\n"
            "4. 与问题保持相同语言，最多 {count} 行。",
        ),
        ("human", "{question}"),
    ]
)


def _format_history(history: Sequence[BaseMessage], limit: int = 6) -> str:
    lines: List[str] = []
    for message in list(history)[-limit:]:
        if isinstance(message, HumanMessage):
            lines.append(f"用户：{message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"助手：{message.content}")
    return "\n".join(lines) if lines else "（无历史，这是对话的第一句）"

# 清洗模型输出的一行：去掉行首的列表符号/编号（如 "1." "- " "* " "2、"）和首尾引号、空白。
# 模型未必遵守"只输出问题本身"，常会写成 markdown 列表或加引号，这里统一剥掉。
def _clean_line(text: str) -> str:
    line = (text or "").strip()
    line = re.sub(r"^[\-\*\d\.\)、\s]+", "", line)
    line = line.strip().strip('"').strip("'").strip()
    return line


class QueryRewriter:
    """把用户问题改写成适合检索的形式。"""

    def __init__(
        self,
        llm: BaseChatModel,
        enable_history: bool = True,
        enable_multi_query: bool = True,
        num_queries: int = 2,
    ) -> None:
        self.llm = llm
        self.enable_history = enable_history
        self.enable_multi_query = enable_multi_query
        self.num_queries = max(1, num_queries)

    def contextualize(self, question: str, history: Sequence[BaseMessage]) -> str:
        """结合历史把追问补全成独立问题；失败时返回原问题。"""
        if not self.enable_history or not history:
            return question
        try:
            # 先定义一个流程
            chain = _HISTORY_PROMPT | self.llm | StrOutputParser()  # LCEL（LangChain Expression Language）的管道写法——用 | 把「模板 → 模型 → 解析」三步串成一条流水线。它和 Shell 的 cat a | grep b 是同一个思想：上一步的输出，就是下一步的输入。
            # 执行该流程
            rewritten = chain.invoke(
                {"history": _format_history(history), "question": question}
            )
        except Exception as exc:  # noqa: BLE001 - 改写失败不影响主流程
            logger.warning("问题补全失败，使用原问题检索：%s", exc)
            return question

        cleaned = _clean_line(rewritten)
        if not cleaned or len(cleaned) > 400:
            return question
        logger.debug("问题改写：%r → %r", question, cleaned)
        return cleaned

    def expand(self, question: str) -> List[str]:
        """生成多个检索问法（始终包含原问题）。"""
        queries = [question]
        if not self.enable_multi_query or self.num_queries <= 1:
            return queries

        try:
            chain = _MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
            response = chain.invoke({"question": question, "count": self.num_queries})
        except Exception as exc:  # noqa: BLE001
            logger.warning("多查询生成失败，只用原问题检索：%s", exc)
            return queries

        for raw in (response or "").splitlines():
            candidate = _clean_line(raw)
            if not candidate or len(candidate) > 300:
                continue
            if candidate.lower() == question.strip().lower():
                continue
            if candidate not in queries:
                queries.append(candidate)
            if len(queries) >= self.num_queries + 1:
                break

        logger.debug("检索问法：%s", queries)
        return queries

    def rewrite(self, question: str, history: Optional[Sequence[BaseMessage]] = None) -> List[str]:
        """完整流程：先补全，再多查询展开。"""
        standalone = self.contextualize(question, history or [])
        return self.expand(standalone)