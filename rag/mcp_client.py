"""可选的 MCP 客户端：把外部 MCP Server 的工具并入 Agent 工具链。

项目本身已经用 Function Calling（`bind_tools`）实现了工具调用，满足赛道的
「MCP 协议或 Function Calling」要求；这里再补一层 **MCP 协议客户端**，让
Agent 可以复用外部生态里的 MCP Server（文件系统、数据库、检索服务等）。

设计为「装了就用、没装就静默降级」：

- 需要 `langchain-mcp-adapters`（以及被连接 Server 自己的依赖）；
- `MCP_SERVERS` 用 JSON 配置多个 Server，格式与 `MultiServerMCPClient` 一致；
- 任何异常都只记日志，不阻塞主流程 —— 没有 MCP 也能正常问答。
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from rag.config import MCP_SERVERS

logger = logging.getLogger(__name__)

_CACHE: Dict[str, List[Any]] = {}

# 外部 MCP Server 启动（npx / node）有时会很慢甚至卡住；给一个较短的超时，
# 避免登录后「正在初始化 Agent 与知识库…」被 MCP 连接拖住。
_MCP_LOAD_TIMEOUT = float(os.getenv("MCP_LOAD_TIMEOUT", "3") or "3")


def _run_coroutine(coro):
    """在同步代码里执行协程；若当前线程已有事件循环则放到子线程跑。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _run_coroutine_with_timeout(coro, timeout: float):
    """在独立线程里执行协程，超时后放弃等待，避免阻塞会话初始化。"""
    result: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"MCP Server 连接超过 {timeout:.1f}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def parse_servers(raw: Optional[str] = None) -> Dict[str, dict]:
    """解析 MCP_SERVERS JSON；格式不对时返回空字典并记录日志。"""
    payload = (raw if raw is not None else MCP_SERVERS or "").strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except ValueError as exc:
        logger.warning("MCP_SERVERS 不是合法 JSON，已忽略：%s", exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("MCP_SERVERS 必须是 {name: config} 形式的对象，已忽略")
        return {}
    return {str(name): config for name, config in parsed.items() if isinstance(config, dict)}


def load_mcp_tools(raw: Optional[str] = None) -> List[Any]:
    """从配置的 MCP Server 加载 LangChain 工具列表（失败返回空列表）。"""
    config = parse_servers(raw)
    if not config:
        return []

    key = json.dumps(config, sort_keys=True, ensure_ascii=False)
    if key in _CACHE:
        return _CACHE[key]

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "未安装 langchain-mcp-adapters，跳过外部 MCP 工具（%s）。"
            "需要时执行：pip install langchain-mcp-adapters",
            exc,
        )
        _CACHE[key] = []
        return []

    try:
        client = MultiServerMCPClient(config)
        tools = _run_coroutine_with_timeout(client.get_tools(), _MCP_LOAD_TIMEOUT)
    except TimeoutError as exc:
        logger.warning(
            "连接外部 MCP Server 超时（%s），已降级为内置工具；"
            "如需启用请检查 MCP_SERVERS 或调大 MCP_LOAD_TIMEOUT。",
            exc,
        )
        _CACHE[key] = []
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("连接外部 MCP Server 失败，已降级为内置工具：%s", exc)
        _CACHE[key] = []
        return []

    tools = list(tools or [])
    _CACHE[key] = tools
    logger.info("已从 MCP Server 加载 %d 个外部工具：%s", len(tools), ", ".join(
        getattr(tool, "name", "?") for tool in tools
    ))
    return tools
