"""多步工具链：让模型自己决定「先查什么、再查什么」，而不是一次检索定生死。

为什么不直接用 LangGraph：这条链路本质是「ReAct 循环」（思考 → 调工具 → 观察 →
再思考），用原生 tool-calling 写 60 行就能说清楚每一步，比引入 StateGraph 更好讲也
更好调。真要做多 Agent 协作再换 LangGraph 不迟。

循环结束后，把工具收集到的证据交给一次流式生成，既保证答案有出处，也保留了
打字机效果（工具阶段本来就没有可展示的文本，不需要流式）。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

from rag.config import TOOL_MAX_STEPS
from rag.tools import LiteratureTools, ToolCall

logger = logging.getLogger(__name__)

_AGENT_PROMPT = """你是一个学术文献检索助手。你可以调用工具来查证问题，也可以直接回答。

可用工具：{tool_names}

规则：
1. 需要文献证据时先调用工具，不要凭空猜测；
2. 一个问题通常 1~2 次工具调用就够，最多 {max_steps} 步；
3. 工具选择建议：
   - 「上传了哪些论文」→ {list_tool}
   - 「哪一节怎么做的」→ {read_tool}
   - 「对比多篇论文的差异」→ {compare_tool}
   - 「它引用了谁 / 谁引用了它 / 技术演进脉络」→ 引用图谱工具
     （get_references / get_citations / trace_lineage）
   - 「做一张方法对比表」→ compare_methods
   - 其余具体问题 → {search_tool}
4. 证据足够时就停止调用工具，直接给出回答（回答会由后续步骤统一生成）。"""

# 根据写的「字段」自动生成 __init__、__repr__、__eq__ 等样板代码。
@dataclass
class ToolChainResult:
    """工具链的执行结果。"""

    documents: List[Document] = field(default_factory=list)
    calls: List[ToolCall] = field(default_factory=list)
    evidence: str = ""
    steps: int = 0
    stopped_reason: str = "done"
    table: str = ""

    # 判断是否调用了工具函数
    # property把一个「方法」伪装成「属性」——访问时不用加括号，用起来像读字段，调用方法跟调用属性一样 ToolChainResult.used_tools        # 不用加 ()  ← 语法更干净
    @property
    def used_tools(self) -> bool:
        return bool(self.calls)


def _extract_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """兼容不同模型返回工具调用的格式。"""
    # 安全地取出 message.tool_calls，并保证结果「永远是列表」——没有就返回 []，不会报错。「取出」tool_calls 这个列表
    calls = getattr(message, "tool_calls", None) or []
    normalized: List[Dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            normalized.append(
                {
                    "name": call.get("name") or "",
                    "args": call.get("args") or {},
                    "id": call.get("id") or "",
                }
            )
    return [call for call in normalized if call["name"]]


class ToolChainRunner:
    """ReAct 风格的多步工具链。"""

    def __init__(
        self,
        llm: BaseChatModel,
        knowledge,
        max_steps: int = TOOL_MAX_STEPS,
        citation_graph=None,
        extraction_store=None,
    ) -> None:
        self.llm = llm
        self.knowledge = knowledge
        self.max_steps = max(1, max_steps)
        self.citation_graph = citation_graph
        self.extraction_store = extraction_store
        self.tools = LiteratureTools(
            knowledge,
            citation_graph=citation_graph,
            extraction_store=extraction_store,
        )

    # 给 _AGENT_PROMPT 里的 {占位符} 填上真实值，生成一段完整的提示词字符串。
    def _system_prompt(self) -> str:
        return _AGENT_PROMPT.format(
            tool_names=", ".join(tool.name for tool in self.tools.as_langchain_tools()),
            max_steps=self.max_steps,
            list_tool="list_papers",
            read_tool="read_section",
            compare_tool="compare_papers",
            search_tool="search_literature",
        )

    def run(
        self,
        question: str,
        history: Optional[Sequence[BaseMessage]] = None,
    ) -> ToolChainResult:
        """执行工具链，返回收集到的证据与调用记录。"""
        self.tools = LiteratureTools(
            self.knowledge,
            citation_graph=self.citation_graph,
            extraction_store=self.extraction_store,
        )
        result = ToolChainResult() # 判断是否调用了工具函数，为bool类型

        try:
            # 是在框架层的BaseChatModel 这个类的一个方法 bind_tools()，bind_tools 不做决定，它只做一件事：把「有哪些工具」告诉模型。并不会直接运行
            bound = self.llm.bind_tools(self.tools.as_langchain_tools())
        except Exception as exc:  # noqa: BLE001 - 模型不支持工具调用
            logger.warning("当前模型不支持工具调用，回退单步检索：%s", exc)
            documents = self.knowledge.search(question)
            result.documents = documents
            result.evidence = "\n\n".join(
                (document.page_content or "") for document in documents
            )
            result.stopped_reason = "model_without_tools"
            return result

        # 拼接正确的SystemMessage
        messages: List[BaseMessage] = [SystemMessage(content=self._system_prompt())]
        for message in list(history or [])[-6:]:
            messages.append(message)
        messages.append(HumanMessage(content=question))

        evidence_blocks: List[str] = []

        for step in range(1, self.max_steps + 1):
            try:
                # 思考(Reasoning)：把消息发给模型 → 返回 AIMessage（可能带 tool_calls，表示"想调哪个工具"）
                response = bound.invoke(messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("工具链第 %d 步模型调用失败：%s", step, exc)
                result.stopped_reason = "llm_error"
                break

            result.steps = step
            messages.append(response)
            calls = _extract_tool_calls(response)  # 从 tool_calls 提取【工具名 + 参数】列表（此时还没执行）

            if not calls:
                content = getattr(response, "content", "") or ""
                if content:
                    evidence_blocks.append(str(content))
                result.stopped_reason = "done"
                break

            for call in calls:
                before = len(self.tools.collector)  # 执行前的片段条数（快照，用于算本次新增）
                output = self.tools.dispatch(call["name"], call["args"])  # 行动(Acting)：真正执行工具，返回结果文本
                record = self.tools.record(step, call["name"], call["args"], output)  # 造一条调用记录（ToolCall 实例）
                record.documents = len(self.tools.collector) - before  # 本次新增片段条数 = 执行后 - 执行前
                evidence_blocks.append(f"【{call['name']}】\n{output}")
                messages.append(
                    # 观察(Observation)：把工具结果回填给模型，下一轮它才能看到
                    ToolMessage(content=output, tool_call_id=call["id"] or call["name"])
                )
        else:
            result.stopped_reason = "max_steps"

        result.calls = self.tools.calls
        result.documents = self.tools.collector.documents
        result.evidence = "\n\n".join(evidence_blocks)
        result.table = self.tools.last_table
        logger.info(
            "工具链结束：%d 步，%d 次调用，收集 %d 条片段（%s）",
            result.steps,
            len(result.calls),
            len(result.documents),
            result.stopped_reason,
        )
        return result

    def stream_final(
        self,
        chain: Runnable,
        question: str,
        context: str,
        history: Optional[Sequence[BaseMessage]] = None,
        extra_evidence: str = "",
    ) -> Iterator[str]:
        """用收集到的证据做一次流式生成。"""
        if extra_evidence:
            context = f"{context}\n\n---\n工具查证记录（可能包含多篇文献）：\n{extra_evidence}"
        payload = {
            "context": context,
            "question": question,
            "chat_history": list(history or []),
        }
        for chunk in chain.stream(payload):
            if chunk:
                yield chunk