"""审计日志：谁在什么时候做了什么。

只记"改变系统状态"的动作（登录、导入、删除、清空、建库、备份、改权限…），
不记普通提问（提问在 qa_log 里，用于配额与成本统计）。
"""

import csv
import io
import logging
import time
from typing import Dict, List, Optional

from backend import db
from backend.config import AUDIT_KEEP_DAYS

logger = logging.getLogger(__name__)

# 动作字典：前端筛选下拉框直接用，避免各处手写字符串出现拼写差异
ACTIONS: Dict[str, str] = {
    "login": "登录",
    "login_failed": "登录失败",
    "logout": "登出",
    "upload_pdf": "上传 PDF",
    "import_arxiv": "导入 arXiv",
    "delete_paper": "删除文献",
    "clear_library": "清空知识库",
    "create_library": "新建知识库",
    "rename_library": "重命名知识库",
    "delete_library": "删除知识库",
    "assign_library": "分配知识库",
    "refresh_graph": "抓取引用关系",
    "user_create": "新建用户",
    "user_reset": "重置密码",
    "user_role": "修改角色",
    "user_active": "启用/停用",
    "user_quota": "修改配额",
    "user_delete": "删除用户",
    "backup": "创建备份",
    "restore": "恢复备份",
    "eval_run": "运行评测",
    "eval_upload": "上传评测集",
    "eval_delete": "删除评测记录",
    "eval_clear": "清空评测记录",
}


def log(
    username: str,
    role: str,
    action: str,
    target: str = "",
    detail: str = "",
    ok: bool = True,
) -> None:
    try:
        db.init_db()
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO audit_log (ts, username, role, action, target, detail, ok)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    username or "-",
                    role or "-",
                    action,
                    (target or "")[:200],
                    (detail or "")[:500],
                    1 if ok else 0,
                ),
            )
            connection.commit()
    except Exception as exc:  # noqa: BLE001 - 审计失败不能影响主流程
        logger.warning("写审计日志失败：%s", exc)


def recent(
    limit: int = 200,
    username: Optional[str] = None,
    action: Optional[str] = None,
) -> List[Dict[str, object]]:
    clauses = []
    params: List[object] = []
    if username:
        clauses.append("username = ?")
        params.append(username)
    if action:
        clauses.append("action = ?")
        params.append(action)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))
    with db.connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
    return [
        {
            "时间": row["ts"],
            "用户": row["username"],
            "角色": row["role"],
            "动作": ACTIONS.get(row["action"], row["action"]),
            "对象": row["target"],
            "详情": row["detail"],
            "结果": "成功" if row["ok"] else "失败",
        }
        for row in rows
    ]


def to_csv(rows: List[Dict[str, object]]) -> str:
    buffer = io.StringIO()
    fields = ["时间", "用户", "角色", "动作", "对象", "详情", "结果"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})
    return buffer.getvalue()


def purge_old(days: int = AUDIT_KEEP_DAYS) -> int:
    """清理超过保留期的日志（0 表示不清理）。"""
    if days <= 0:
        return 0
    cutoff = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - days * 86400))
    with db.connect() as connection:
        cursor = connection.execute("DELETE FROM audit_log WHERE ts < ?", (cutoff,))
        connection.commit()
        return int(cursor.rowcount or 0)
