"""管理后台的数据层：概览指标、知识库运维、备份、健康检查。

前端只负责渲染，这里把「查什么、怎么算」集中起来，方便单独测试，
也避免把 SQL 散落在界面代码里。
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from backend import db, users
from backend.config import BACKUP_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 概览
def _dir_size_mb(path: Path, cap_files: int = 4000) -> float:
    if not path.exists():
        return 0.0
    total = 0
    count = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
            count += 1
            if count >= cap_files:
                return round(total / 1024 / 1024, 2)
    return round(total / 1024 / 1024, 2)


def _graph_metrics(path: str) -> Dict[str, int]:
    file = Path(path)
    if not file.exists():
        return {"papers": 0, "edges": 0, "in_library": 0, "size_kb": 0}
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"papers": 0, "edges": 0, "in_library": 0, "size_kb": 0}
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or payload.get("links") or []
    in_library = sum(1 for node in nodes if node.get("in_library"))
    return {
        "papers": len(nodes),
        "edges": len(edges),
        "in_library": in_library,
        "size_kb": round(file.stat().st_size / 1024),
    }


def library_metrics() -> List[Dict[str, object]]:
    from rag import libraries as libs

    rows: List[Dict[str, object]] = []
    for library in libs.list_libraries():
        graph = _graph_metrics(library.graph)
        rows.append(
            {
                "名称": library.name,
                "slug": library.slug,
                "归属": library.owner or "（共享）",
                "可见性": "共享" if library.visibility == "shared" else "私有",
                "片段数": libs.chunk_count(library),
                "论文数": graph["in_library"],
                "图谱节点": graph["papers"],
                "引用边": graph["edges"],
                "体积(MB)": _dir_size_mb(Path(library.lancedb)),
            }
        )
    return rows


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _qa_today() -> Dict[str, float]:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(AVG(latency_ms), 0) AS latency, "
            "COALESCE(SUM(question_chars + answer_chars), 0) AS chars, "
            "COALESCE(SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END), 0) AS failed "
            "FROM qa_log WHERE ts LIKE ?",
            (f"{_today()}%",),
        ).fetchone()
    return {
        "total": int(row["total"] or 0),
        "latency_ms": float(row["latency"] or 0),
        "chars": int(row["chars"] or 0),
        "failed": int(row["failed"] or 0),
    }


def trend(days: int = 7) -> List[Tuple[str, int]]:
    """最近 N 天每天的提问数（按本地日期）。"""
    start = time.time() - (days - 1) * 86400
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT substr(ts, 1, 10) AS day, COUNT(*) AS total FROM qa_log "
            "WHERE ts >= ? GROUP BY day",
            (time.strftime("%Y-%m-%d 00:00:00", time.localtime(start)),),
        ).fetchall()
    found = {row["day"]: int(row["total"]) for row in rows}
    series = []
    for offset in range(days):
        day = time.strftime("%m-%d", time.localtime(start + offset * 86400))
        key = time.strftime("%Y-%m-%d", time.localtime(start + offset * 86400))
        series.append((day, found.get(key, 0)))
    return series


def quality(days: int = 7) -> Dict[str, object]:
    """在线运行质量：引用可溯源率、平均置信度、工具步数、拒答率。

    只有升级后产生的新记录才带这些字段（旧记录 grounded/confidence 为 -1，
    统计时排除），所以样本数可能小于总提问数。
    """
    start = time.time() - (days - 1) * 86400
    cutoff = time.strftime("%Y-%m-%d 00:00:00", time.localtime(start))
    with db.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN grounded >= 0 THEN 1 ELSE 0 END), 0) AS graded, "
            "COALESCE(SUM(CASE WHEN grounded = 1 THEN 1 ELSE 0 END), 0) AS grounded, "
            "COALESCE(AVG(CASE WHEN confidence >= 0 THEN confidence END), 0) AS conf, "
            "COALESCE(AVG(CASE WHEN grounded >= 0 THEN tool_steps END), 0) AS steps, "
            "COALESCE(AVG(CASE WHEN grounded >= 0 THEN refused END), 0) AS refused "
            "FROM qa_log WHERE ts >= ?",
            (cutoff,),
        ).fetchone()
    graded = int(row["graded"] or 0)
    return {
        "days": days,
        "total": int(row["total"] or 0),
        "graded": graded,
        "grounded_rate": (int(row["grounded"] or 0) / graded) if graded else None,
        "avg_confidence": float(row["conf"] or 0) if graded else None,
        "avg_tool_steps": float(row["steps"] or 0),
        "refusal_rate": float(row["refused"] or 0),
    }


def overview() -> Dict[str, object]:
    counts = users.count_users()
    today = _qa_today()
    token_estimate = int(today["chars"] / 4)
    price = float(os.getenv("APP_TOKEN_PRICE_PER_1K", "0") or 0)
    return {
        "users_total": counts["total"],
        "users_active": counts["active"],
        "users_admins": counts["admins"],
        "qa_today": today["total"],
        "qa_failed": today["failed"],
        "avg_latency_ms": int(today["latency_ms"]),
        "tokens_today": token_estimate,
        "cost_today": round(token_estimate / 1000 * price, 4) if price else None,
        "trend": trend(7),
        "quality": quality(7),
        "libraries": library_metrics(),
    }


def recent_questions(limit: int = 100) -> List[Dict[str, object]]:
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT ts, username, library, question_chars, answer_chars, latency_ms,"
            " model, ok, error, grounded, confidence, tool_steps, refused"
            " FROM qa_log ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()

    def _grounded(value: int) -> str:
        if int(value) < 0:
            return "—"
        return "✅ 可溯源" if int(value) == 1 else "⚠️ 无引用"

    return [
        {
            "时间": row["ts"],
            "用户": row["username"],
            "知识库": row["library"],
            "问题字数": row["question_chars"],
            "回答字数": row["answer_chars"],
            "延迟(ms)": row["latency_ms"],
            "约 tokens": int((row["question_chars"] + row["answer_chars"]) / 4),
            "模型": row["model"],
            "引用": _grounded(row["grounded"]),
            "置信度": "—"
            if float(row["confidence"]) < 0
            else round(float(row["confidence"]), 2),
            "工具步数": row["tool_steps"],
            "拒答": "是" if row["refused"] else "否",
            "结果": "成功" if row["ok"] else f"失败：{row['error'][:40]}",
        }
        for row in rows
    ]


# ---------------------------------------------------------------- 健康检查
def health() -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        items.append({"检查项": name, "状态": "✅" if ok else "❌", "说明": detail, "ok": ok})

    # 平台库
    try:
        db.init_db()
        counts = users.count_users()
        add("平台库（SQLite）", True, f"{counts['total']} 个账号")
    except Exception as exc:  # noqa: BLE001
        add("平台库（SQLite）", False, str(exc)[:80])

    # 会话库
    try:
        from rag.config import SQLITE_PATH

        path = Path(SQLITE_PATH)
        add("会话库（SQLite）", path.exists(), str(path))
    except Exception as exc:  # noqa: BLE001
        add("会话库（SQLite）", False, str(exc)[:80])

    # 向量库
    try:
        from rag import libraries as libs

        default = libs.default_library()
        count = libs.chunk_count(default)
        add("向量库（LanceDB）", True, f"{default.name}：{count} 片段")
    except Exception as exc:  # noqa: BLE001
        add("向量库（LanceDB）", False, str(exc)[:80])

    # 目录可写
    try:
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add("数据目录可写", True, str(data_dir))
    except Exception as exc:  # noqa: BLE001
        add("数据目录可写", False, str(exc)[:80])

    # 大模型
    try:
        from agent.config import LLM_PROFILE, LLM_PROVIDER_LABEL

        key_env = str(LLM_PROFILE.get("api_key_env") or "")
        has_key = bool(os.getenv(key_env, "").strip()) if key_env else True
        add(
            f"大模型（{LLM_PROVIDER_LABEL}）",
            has_key,
            f"{key_env or '无需 Key'}：" + ("已配置" if has_key else "未配置"),
        )
    except Exception as exc:  # noqa: BLE001
        add("大模型", False, str(exc)[:80])

    # 本地 Ollama（可选）
    try:
        from rag.config import ollama_reachable

        add("本地 Ollama", ollama_reachable(), "embedding / 本地模型可选")
    except Exception as exc:  # noqa: BLE001
        add("本地 Ollama", False, str(exc)[:80])

    # Semantic Scholar
    try:
        from rag.citation_graph import s2_request_interval, s2_stats

        stats = s2_stats()
        add(
            "Semantic Scholar",
            True,
            f"间隔 {s2_request_interval():.1f}s ｜ 请求 {stats['requests']} 次 ｜ "
            f"429 {stats['throttled']} 次",
        )
    except Exception as exc:  # noqa: BLE001
        add("Semantic Scholar", False, str(exc)[:80])

    return items


# ---------------------------------------------------------------- 备份与恢复
_TARGETS = {
    "platform": "platform",
    "assistant": "assistant",
    "libraries": "libraries",
}


def _targets() -> Dict[str, Path]:
    from rag import libraries as libs
    from rag.config import SQLITE_PATH

    return {
        "platform": Path(db.PLATFORM_DB_PATH),
        "assistant": Path(SQLITE_PATH),
        "libraries": Path(libs.REGISTRY_PATH),
    }


def list_backups() -> List[Dict[str, object]]:
    if not BACKUP_DIR.exists():
        return []
    rows = []
    for file in sorted(BACKUP_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if file.suffix.lower() not in {".db", ".json"} or not file.is_file():
            continue
        rows.append(
            {
                "文件": file.name,
                "大小(KB)": round(file.stat().st_size / 1024, 1),
                "时间": time.strftime("%Y-%m-%d %H:%M", time.localtime(file.stat().st_mtime)),
                "path": str(file),
            }
        )
    return rows[:60]


def create_backup() -> List[str]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    created: List[str] = []
    for name, source in _targets().items():
        if not source.exists():
            continue
        target = BACKUP_DIR / f"{name}-{stamp}{source.suffix}"
        try:
            shutil.copy2(source, target)
            created.append(target.name)
        except OSError as exc:  # noqa: BLE001
            logger.warning("备份 %s 失败：%s", source, exc)
    return created


def restore_from_bytes(filename: str, data: bytes) -> str:
    """从上传的备份文件恢复；只允许 platform/assistant/libraries 三类。"""
    prefix = (filename or "").split("-")[0].lower()
    if prefix not in _TARGETS:
        raise ValueError("只支持恢复 platform-*.db / assistant-*.db / libraries-*.json")
    target = _targets()[prefix]

    # 覆盖前先把当前文件再备一份，避免恢复错版本无法回退
    if target.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            target,
            BACKUP_DIR / f"{prefix}-prerestore-{time.strftime('%Y%m%d-%H%M%S')}{target.suffix}",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(target)


def restore_file(filename: str) -> str:
    path = BACKUP_DIR / filename
    if not path.exists():
        raise ValueError("备份文件不存在")
    return restore_from_bytes(filename, path.read_bytes())


# ---------------------------------------------------------------- 知识库运维
def refresh_library_graph(
    library_name: str,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, int]:
    """对某个知识库里的全部在库论文重新抓引用关系（不依赖 LLM）。"""
    from rag import libraries as libs
    from rag.citation_graph import CitationGraph, CitationGraphBuilder

    library = libs.get_library(library_name)
    graph = CitationGraph(path=library.graph)
    builder = CitationGraphBuilder(graph)
    targets = [
        (str(data.get("title") or ""), str(data.get("source_key") or ""))
        for _node, data in graph.graph.nodes(data=True)
        if data.get("in_library")
    ]

    summary = {"papers": 0, "references": 0, "citations": 0, "local_references": 0}
    total = len(targets)
    for index, (title, source_key) in enumerate(targets):
        arxiv_code = source_key[len("arxiv:"):] if source_key.startswith("arxiv:") else ""
        try:
            counts = builder.ingest(
                arxiv_id=arxiv_code, title=title, source_key=source_key
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("重抓 %s 失败：%s", title, exc)
            counts = {}
        summary["papers"] += 1
        for key in ("references", "citations", "local_references"):
            summary[key] += int((counts or {}).get(key, 0) or 0)
        if on_progress is not None:
            try:
                on_progress(index + 1, total, title)
            except Exception:  # noqa: BLE001
                pass
    return summary
