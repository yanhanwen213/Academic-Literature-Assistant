"""知识库（向量库）注册表：支持多套命名知识库并存、切换。

为什么需要它：LanceDB 是一个目录型向量库，一个目录只对应一套向量表。
把「知识库」提升为可命名的概念后，用户可以为不同课题各建一套
（例如「静脉识别」「RAG 评测」），互不污染，切换即用。

布局约定：

    data/libraries.json              注册表（名称 → 路径）
    data/lancedb/                    默认知识库的向量目录（兼容历史数据）
    data/citation_graph.json         默认知识库的引用图
    data/extraction.db               默认知识库的抽取结果
    data/libraries/<slug>/lancedb/   新建知识库的向量目录
    data/libraries/<slug>/citation_graph.json
    data/libraries/<slug>/extraction.db

每个知识库拥有独立的向量表、引用图与结构化抽取库，切换知识库时
「片段数 / 引用图谱 / 对比表」都会跟着切，不会串。
"""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rag.config import EXTRACTION_PATH, GRAPH_PATH, LANCEDB_DIR

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


REGISTRY_PATH = _env_path("RAG_LIBRARIES_REGISTRY", PROJECT_ROOT / "data" / "libraries.json")
LIBRARIES_DIR = _env_path("RAG_LIBRARIES_DIR", PROJECT_ROOT / "data" / "libraries")

DEFAULT_LIBRARY_NAME = "默认知识库"
NAME_MAX_LEN = 40

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Library:
    """一套知识库的元数据与路径。"""

    name: str
    slug: str
    lancedb: str
    graph: str
    extraction: str
    created: float = 0.0
    note: str = ""
    is_default: bool = False
    # 归属与可见性：owner 为用户名（空 = 未分配）；visibility 取 shared / private
    owner: str = ""
    visibility: str = "shared"

    @property
    def root(self) -> Path:
        return Path(self.lancedb).parent


def _abs(raw: str) -> str:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _rel(raw: str) -> str:
    """注册表里存相对路径，换机器/换目录也能用。"""
    path = Path(raw)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:  # noqa: BLE001
        return str(path)


def _slug_for(name: str) -> str:
    base = _SLUG_RE.sub("-", name.strip().lower()).strip("-")[:24] or "lib"
    digest = hashlib.md5(name.strip().encode("utf-8")).hexdigest()[:6]
    return f"{base}-{digest}"


def _default_entry() -> Library:
    return Library(
        name=DEFAULT_LIBRARY_NAME,
        slug="default",
        lancedb=_rel(LANCEDB_DIR),
        graph=_rel(GRAPH_PATH),
        extraction=_rel(EXTRACTION_PATH),
        created=0.0,
        note="历史数据目录",
        is_default=True,
        owner="",
        visibility="shared",
    )


def _to_dict(library: Library) -> Dict[str, str]:
    payload = {
        "name": library.name,
        "slug": library.slug,
        "lancedb": library.lancedb,
        "graph": library.graph,
        "extraction": library.extraction,
        "created": library.created,
        "note": library.note,
    }
    if library.is_default:
        payload["is_default"] = True
    if library.owner:
        payload["owner"] = library.owner
    if library.visibility and library.visibility != "shared":
        payload["visibility"] = library.visibility
    return payload


def _from_dict(item: Dict) -> Optional[Library]:
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    is_default = bool(item.get("is_default"))
    if is_default:
        # 默认知识库的路径始终以当前配置为准（LANCEDB_DIR 等可能随 .env 变化），
        # 避免用过期的绝对路径把历史数据「藏」起来。
        return _default_entry()
    lancedb = _abs(item.get("lancedb") or "")
    graph = _abs(item.get("graph") or Path(lancedb).parent / "citation_graph.json")
    extraction = _abs(item.get("extraction") or Path(lancedb).parent / "extraction.db")
    return Library(
        name=name,
        slug=str(item.get("slug") or _slug_for(name)),
        lancedb=lancedb,
        graph=graph,
        extraction=extraction,
        created=float(item.get("created") or 0.0),
        note=str(item.get("note") or ""),
        is_default=False,
        owner=str(item.get("owner") or ""),
        visibility="private" if str(item.get("visibility") or "shared") == "private" else "shared",
    )


def _read_registry() -> List[Library]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("知识库注册表读取失败，忽略：%s", exc)
        return []
    items = payload.get("libraries") if isinstance(payload, dict) else payload
    libraries: List[Library] = []
    for item in items or []:
        if isinstance(item, dict):
            library = _from_dict(item)
            if library is not None:
                libraries.append(library)
    return libraries


def _write_registry(libraries: List[Library]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"libraries": [_to_dict(library) for library in libraries]}
    REGISTRY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _normalized(libraries: List[Library]) -> List[Library]:
    """保证默认知识库始终存在，且排在第一位。"""
    if not any(library.is_default for library in libraries):
        libraries = [entry for entry in libraries if entry.slug != "default"]
        libraries.insert(0, _default_entry())
    libraries.sort(key=lambda item: (not item.is_default, item.created))
    return libraries


def list_libraries(reload: bool = True) -> List[Library]:
    """全部知识库（默认知识库永远在第一位）。"""
    libraries = _normalized(_read_registry() if reload else _read_registry())
    if not REGISTRY_PATH.exists():
        try:
            _write_registry(libraries)
        except OSError as exc:  # noqa: BLE001
            logger.warning("初始化知识库注册表失败：%s", exc)
    return libraries


def get_library(name: Optional[str]) -> Library:
    """按名称取知识库；找不到时退回默认知识库。"""
    libraries = list_libraries()
    for library in libraries:
        if name and library.name == name:
            return library
    for library in libraries:
        if library.is_default:
            return library
    return _default_entry()


def default_library() -> Library:
    return get_library(None)


def visible_to(owner: str, is_admin: bool) -> List[Library]:
    """按角色过滤可见知识库。

    管理员看全部；其他人看「共享库 + 自己名下的库」。这样访客只能看到
    演示库，成员既能看到演示库也能看到自己被分配的私有库。
    """
    libraries = list_libraries()
    if is_admin:
        return libraries
    return [
        library
        for library in libraries
        if library.visibility == "shared" or not library.owner or library.owner == owner
    ]


def _validate_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("知识库名称不能为空")
    if len(cleaned) > NAME_MAX_LEN:
        raise ValueError(f"知识库名称最多 {NAME_MAX_LEN} 个字符")
    if any(char in cleaned for char in "\\/:*?\"<>|"):
        raise ValueError("知识库名称不能包含 \\ / : * ? \" < > | 这些字符")
    return cleaned


def create_library(
    name: str, note: str = "", owner: str = "", visibility: str = ""
) -> Library:
    """新建一套知识库（同名已存在时抛错）。

    `owner` 为空时视为共享库；否则默认私有，只有管理员和本人可见可写。
    """
    cleaned = _validate_name(name)
    libraries = list_libraries()
    if any(library.name == cleaned for library in libraries):
        raise ValueError(f"知识库「{cleaned}」已存在")

    slug = _slug_for(cleaned)
    suffix = 2
    while any(library.slug == slug for library in libraries):
        slug = f"{_slug_for(cleaned)}-{suffix}"
        suffix += 1

    root = LIBRARIES_DIR / slug
    library = Library(
        name=cleaned,
        slug=slug,
        lancedb=_rel(root / "lancedb"),
        graph=_rel(root / "citation_graph.json"),
        extraction=_rel(root / "extraction.db"),
        created=time.time(),
        note=(note or "").strip(),
        owner=(owner or "").strip(),
        visibility=(
            visibility
            or ("private" if (owner or "").strip() else "shared")
        ),
    )
    Path(library.lancedb).mkdir(parents=True, exist_ok=True)
    libraries.append(library)
    _write_registry(_normalized(libraries))
    logger.info("已创建知识库 %s（%s）", cleaned, library.lancedb)
    return library


def set_library_access(name: str, owner: str, visibility: str = "") -> Library:
    """调整知识库归属与可见性（管理后台用）。"""
    libraries = list_libraries()
    target = next((item for item in libraries if item.name == name), None)
    if target is None:
        raise ValueError(f"知识库「{name}」不存在")
    if target.is_default and visibility == "private":
        raise ValueError("默认知识库必须保持共享，否则访客无法演示")
    target.owner = (owner or "").strip()
    target.visibility = visibility or ("private" if target.owner else "shared")
    _write_registry(_normalized(libraries))
    return target


def rename_library(old_name: str, new_name: str) -> Library:
    """只改显示名，不搬数据目录。"""
    cleaned = _validate_name(new_name)
    libraries = list_libraries()
    target = next((item for item in libraries if item.name == old_name), None)
    if target is None:
        raise ValueError(f"知识库「{old_name}」不存在")
    if any(item.name == cleaned and item is not target for item in libraries):
        raise ValueError(f"知识库「{cleaned}」已存在")
    target.name = cleaned
    _write_registry(_normalized(libraries))
    return target


def delete_library(name: str) -> None:
    """删除知识库（默认知识库不可删除），连同其目录一起移除。"""
    libraries = list_libraries()
    target = next((item for item in libraries if item.name == name), None)
    if target is None:
        raise ValueError(f"知识库「{name}」不存在")
    if target.is_default:
        raise ValueError("默认知识库不能删除，请用「清空知识库」只清空内容")

    libraries = [item for item in libraries if item is not target]
    _write_registry(_normalized(libraries))

    root = Path(target.lancedb).parent
    try:
        if root.exists() and LIBRARIES_DIR in root.parents:
            shutil.rmtree(root, ignore_errors=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning("删除知识库目录失败（不影响注册表）：%s", exc)


def chunk_count(library: Library) -> int:
    """该知识库所有向量表的行数之和（不需要 embedding，可安全用于侧栏展示）。"""
    try:
        import lancedb
    except ImportError:  # pragma: no cover
        return 0

    uri = Path(library.lancedb)
    if not uri.exists():
        return 0
    try:
        db = lancedb.connect(str(uri))
        names = list(getattr(db.list_tables(), "tables", None) or db.table_names())
        total = 0
        for table_name in names:
            try:
                total += int(db.open_table(table_name).count_rows())
            except Exception:  # noqa: BLE001
                continue
        return total
    except Exception as exc:  # noqa: BLE001 - 侧栏展示，失败就当 0
        logger.debug("读取知识库 %s 片段数失败：%s", library.name, exc)
        return 0
