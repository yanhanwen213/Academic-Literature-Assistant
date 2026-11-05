"""会话历史存储：SQLite（Python 标准库，零依赖）。

一个 run 一行，对话消息以 JSON 存进 `messages` 字段。
之所以不用第三方 ORM/存储：这里的数据结构就一张表，标准库足够，
也便于在答辩时把每一行 SQL 讲清楚。
"""

import json
import logging
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from rag.config import SQLITE_PATH

logger = logging.getLogger(__name__)

_TABLE = "chat_runs"
_BACKUP_KEEP = 10
_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    run_id     TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    messages   TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    owner      TEXT NOT NULL DEFAULT ''
)
"""

# 自动命名的长度上限（中文字符按 1 计）
_TITLE_MAX = 24


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def summarize_title(text: str) -> str:
    """把第一句话压缩成一个短标题。

    只做确定性截断（不调大模型），这样命名不花钱、不联网、也不会因为
    API 超时把会话搞成「未命名」。
    """
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    cleaned = cleaned.lstrip("#>-*· ").strip()
    if not cleaned:
        return ""

    sentence = re.search(r"[。！？!?;；]", cleaned)
    if sentence and sentence.start() >= 6:
        cleaned = cleaned[: sentence.start()]
    cleaned = cleaned.strip(" ，,。.：:；;、\"'“”")

    if len(cleaned) > _TITLE_MAX:
        cleaned = cleaned[:_TITLE_MAX].rstrip() + "…"
    return cleaned


def derive_title(messages: List[Dict[str, str]]) -> str:
    """用第一条用户提问生成默认标题。"""
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = (message.get("content") or "").strip()
        if content:
            return summarize_title(content)
    return ""


def backup_file(path: str, keep: int = _BACKUP_KEEP) -> Optional[Path]:
    """把数据库/图谱这类小文件复制到同级 backups/ 目录，只留最近 keep 份。

    删除类操作前调用：即使误删，也能从 data/backups/ 里捞回来。
    """
    source = Path(path)
    if not source.exists():
        return None
    backup_dir = source.parent / "backups"
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / f"{source.stem}-{time.strftime('%Y%m%d-%H%M%S')}{source.suffix}"
        shutil.copy2(source, target)
    except OSError as exc:
        logger.warning("备份 %s 失败：%s", source.name, exc)
        return None

    backups = sorted(
        backup_dir.glob(f"{source.stem}-*{source.suffix}"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old in backups[keep:]:
        try:
            old.unlink()
        except OSError:
            pass
    return target


class SqliteRunStore:
    """极简会话仓库：新建 run、保存消息、改名、删除、按更新时间倒序列出。"""

    def __init__(self, db_path: str = SQLITE_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(_DDL)
            # 老库没有 title 列：就地补上，避免用户升级后历史丢失
            columns = {
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({_TABLE})").fetchall()
            }
            if "title" not in columns:
                connection.execute(
                    f"ALTER TABLE {_TABLE} ADD COLUMN title TEXT NOT NULL DEFAULT ''"
                )
                logger.info("会话表升级：已补充 title 列")
            if "owner" not in columns:
                connection.execute(
                    f"ALTER TABLE {_TABLE} ADD COLUMN owner TEXT NOT NULL DEFAULT ''"
                )
                logger.info("会话表升级：已补充 owner 列")
            connection.commit()

    # ------------------------------------------------------------------ 写
    def create_run(
        self,
        run_id: str,
        messages: List[Dict[str, str]],
        title: str = "",
        owner: str = "",
    ) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO {_TABLE} "
                "(run_id, created_at, updated_at, messages, title, owner) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    now,
                    now,
                    json.dumps(messages, ensure_ascii=False),
                    title or derive_title(messages),
                    owner or "",
                ),
            )
            connection.commit()

    def save(
        self, run_id: str, messages: List[Dict[str, str]], owner: str = ""
    ) -> None:
        # 用 upsert 而不是 UPDATE：即使这一行被清理掉（比如用户在另一个标签页
        # 删了这个会话），当前会话的消息也不会丢，会被重新写回去。
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO {_TABLE} (run_id, created_at, updated_at, messages, title, owner) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "updated_at = excluded.updated_at, "
                "messages = excluded.messages, "
                f"title = CASE WHEN COALESCE({_TABLE}.title, '') = '' "
                "THEN excluded.title ELSE "
                f"{_TABLE}.title END, "
                f"owner = CASE WHEN COALESCE({_TABLE}.owner, '') = '' "
                "THEN excluded.owner ELSE "
                f"{_TABLE}.owner END",
                (
                    run_id,
                    _now(),
                    _now(),
                    json.dumps(messages, ensure_ascii=False),
                    derive_title(messages),
                    owner or "",
                ),
            )
            connection.commit()

    def owner_of(self, run_id: str) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT owner FROM {_TABLE} WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return row["owner"] or ""

    def rename(self, run_id: str, title: str) -> None:
        """手动改名；传空标题时退回按首条提问自动命名。"""
        with self._connect() as connection:
            if title.strip():
                connection.execute(
                    f"UPDATE {_TABLE} SET title = ? WHERE run_id = ?",
                    (title.strip()[:60], run_id),
                )
            else:
                row = connection.execute(
                    f"SELECT messages FROM {_TABLE} WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is None:
                    return
                try:
                    messages = json.loads(row["messages"])
                except (TypeError, ValueError):
                    messages = []
                connection.execute(
                    f"UPDATE {_TABLE} SET title = ? WHERE run_id = ?",
                    (derive_title(messages), run_id),
                )
            connection.commit()

    # ------------------------------------------------------------------ 读
    def load(self, run_id: str) -> Optional[List[Dict[str, str]]]:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT messages FROM {_TABLE} WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["messages"])
        except (TypeError, ValueError):
            logger.warning("会话 %s 的消息不是合法 JSON", run_id)
            return None

    def list_runs(self, limit: int = 50, owner: Optional[str] = None) -> List[str]:
        query = f"SELECT run_id FROM {_TABLE}"
        params: List[object] = []
        if owner is not None:
            query += " WHERE owner = ?"
            params.append(owner)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [row["run_id"] for row in rows]

    def list_run_meta(
        self, limit: int = 50, owner: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """会话列表：id、标题、时间、消息条数（前端下拉框展示用）。

        `owner=None` 表示不过滤（管理员视角）；传用户名时只看自己的历史。
        """
        query = (
            f"SELECT run_id, title, created_at, updated_at, messages, owner FROM {_TABLE}"
        )
        params: List[object] = []
        if owner is not None:
            query += " WHERE owner = ?"
            params.append(owner)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        metas: List[Dict[str, str]] = []
        for row in rows:
            title = (row["title"] or "").strip()
            count = 0
            has_user = False
            try:
                messages = json.loads(row["messages"] or "[]")
                count = len(
                    [m for m in messages if m.get("role") in ("user", "assistant")]
                )
                has_user = any(m.get("role") == "user" for m in messages)
                if not title:
                    # 老数据没标题：用第一条提问补一个，不写回库也不报错
                    title = derive_title(messages)
            except (TypeError, ValueError):
                count = 0
            display = title or ("未命名对话" if has_user else "空对话")
            updated = (row["updated_at"] or "")[5:16]
            metas.append(
                {
                    "run_id": row["run_id"],
                    "title": display,
                    "created_at": row["created_at"] or "",
                    "updated_at": row["updated_at"] or "",
                    "label": f"{display} · {updated}".strip(),
                    "message_count": count,
                }
            )
        return metas

    def delete(self, run_id: str) -> None:
        backup_file(self.db_path)
        with self._connect() as connection:
            connection.execute(f"DELETE FROM {_TABLE} WHERE run_id = ?", (run_id,))
            connection.commit()
