"""平台数据库连接与建表（SQLite，标准库，零依赖）。

与会话库 `data/assistant.db` 分开：这里只放「账号 / 审计 / 用量」，
这样重置聊天记录不会顺手把用户删掉，备份也可以单独做。
"""

import logging
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from backend.config import PLATFORM_DB_PATH

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT    NOT NULL UNIQUE,
    display_name TEXT    NOT NULL DEFAULT '',
    pwd_hash     TEXT    NOT NULL,
    salt         TEXT    NOT NULL,
    role         TEXT    NOT NULL DEFAULT 'member',
    quota_daily  INTEGER NOT NULL DEFAULT 200,
    library_slug TEXT    NOT NULL DEFAULT '',
    active       INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    NOT NULL,
    last_login   TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS audit_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    role     TEXT NOT NULL DEFAULT '',
    action   TEXT NOT NULL,
    target   TEXT NOT NULL DEFAULT '',
    detail   TEXT NOT NULL DEFAULT '',
    ok       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS qa_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    username      TEXT NOT NULL DEFAULT '',
    library       TEXT NOT NULL DEFAULT '',
    question_chars INTEGER NOT NULL DEFAULT 0,
    answer_chars  INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    model         TEXT NOT NULL DEFAULT '',
    ok            INTEGER NOT NULL DEFAULT 1,
    error         TEXT NOT NULL DEFAULT '',
    grounded      INTEGER NOT NULL DEFAULT -1,
    confidence    REAL NOT NULL DEFAULT -1,
    tool_steps    INTEGER NOT NULL DEFAULT 0,
    refused       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    username     TEXT NOT NULL DEFAULT '',
    dataset      TEXT NOT NULL DEFAULT '',
    library      TEXT NOT NULL DEFAULT '',
    mode         TEXT NOT NULL DEFAULT '',
    rerank       TEXT NOT NULL DEFAULT '',
    top_k        INTEGER NOT NULL DEFAULT 0,
    cases        INTEGER NOT NULL DEFAULT 0,
    hit_rate     REAL NOT NULL DEFAULT 0,
    recall       REAL NOT NULL DEFAULT 0,
    mrr          REAL NOT NULL DEFAULT 0,
    keyword_hit  REAL NOT NULL DEFAULT 0,
    precision    REAL NOT NULL DEFAULT 0,
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT NOT NULL DEFAULT '',
    detail_json  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_qa_ts ON qa_log(ts);
CREATE INDEX IF NOT EXISTS idx_qa_user ON qa_log(username, ts);
CREATE INDEX IF NOT EXISTS idx_eval_ts ON eval_runs(ts);
"""

# 旧库升级：CREATE TABLE IF NOT EXISTS 不会给已存在的表补列，这里按需 ALTER
_MIGRATIONS = {
    "qa_log": {
        "grounded": "INTEGER NOT NULL DEFAULT -1",
        "confidence": "REAL NOT NULL DEFAULT -1",
        "tool_steps": "INTEGER NOT NULL DEFAULT 0",
        "refused": "INTEGER NOT NULL DEFAULT 0",
    },
}


def _migrate(connection: sqlite3.Connection) -> None:
    for table, columns in _MIGRATIONS.items():
        existing = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, ddl in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                logger.info("平台库迁移：%s 新增列 %s", table, name)

_initialized = False


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """短连接 + WAL + busy_timeout：多用户同时导入时不会 `database is locked`。"""
    path = Path(PLATFORM_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    """幂等建表；进程内只做一次。"""
    global _initialized
    if _initialized:
        return
    with connect() as connection:
        connection.executescript(_SCHEMA)
        _migrate(connection)
        connection.commit()
    _initialized = True
    logger.info("平台库已就绪：%s", PLATFORM_DB_PATH)


def backup() -> Optional[Path]:
    """把平台库复制到 backups/，返回备份文件路径。"""
    source = Path(PLATFORM_DB_PATH)
    if not source.exists():
        return None
    target_dir = source.parent / "backups"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"platform-{time.strftime('%Y%m%d-%H%M%S')}.db"
    try:
        shutil.copy2(source, target)
    except OSError as exc:  # noqa: BLE001
        logger.warning("备份平台库失败：%s", exc)
        return None
    return target
