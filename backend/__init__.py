"""平台后端：账号 / 权限 / 审计 / 运营统计。

前端（`frontend/`）只调用这里的函数，不直接写 SQL；`agent/` 与 `rag/`
仍然只负责对话与检索，三者互不越界。
"""

from backend import admin, audit, auth, db, eval, users  # noqa: F401

__all__ = ["admin", "audit", "auth", "db", "eval", "users"]
