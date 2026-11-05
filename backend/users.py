"""账号体系：注册、密码、角色、配额。

密码只用标准库 `hashlib.scrypt` + 每用户随机盐，不引入 bcrypt/passlib；
校验用 `hmac.compare_digest`，避免时序侧信道。
"""

import hashlib
import hmac
import logging
import re
import secrets
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from backend import db
from backend.config import GUEST_QUOTA, MEMBER_QUOTA

logger = logging.getLogger(__name__)

ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
ROLE_GUEST = "guest"
ROLES = (ROLE_ADMIN, ROLE_MEMBER, ROLE_GUEST)
ROLE_LABELS = {
    ROLE_ADMIN: "管理员",
    ROLE_MEMBER: "成员",
    ROLE_GUEST: "访客（只读）",
}

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,26}$")
_PWD_ALPHABET = string.ascii_letters + string.digits

# 角色能力表：集中在一处，前端/后端都查这张表，避免权限散落
_CAPABILITIES: Dict[str, set] = {
    ROLE_ADMIN: {"ask", "write", "manage_library", "manage_users", "view_admin"},
    ROLE_MEMBER: {"ask", "write"},
    ROLE_GUEST: {"ask"},
}


@dataclass
class User:
    id: int
    username: str
    display_name: str
    role: str
    quota_daily: int
    library_slug: str
    active: bool
    created_at: str
    last_login: str

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)

    def to_session(self) -> Dict[str, object]:
        """写进 `st.session_state` 的最小快照（权限判断仍以数据库为准）。"""
        return {
            "username": self.username,
            "display_name": self.display_name or self.username,
            "role": self.role,
        }


def can(role: str, action: str) -> bool:
    return action in _CAPABILITIES.get(role, set())


def can_write_library(user: User, owner: str, visibility: str) -> bool:
    """能否往某个知识库里导入/删除文献。

    管理员可写所有库；成员只能写自己的库；共享库（默认库）只有管理员可写，
    避免多人往同一个演示库里灌数据互相污染。
    """
    if user.role == ROLE_ADMIN:
        return True
    if user.role != ROLE_MEMBER:
        return False
    return owner == user.username and visibility != "shared"


# ---------------------------------------------------------------- 密码
def _hash(password: str, salt_hex: str) -> str:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt_hex),
        n=2 ** 14,
        r=8,
        p=1,
        dklen=32,
    ).hex()


def hash_password(password: str) -> Tuple[str, str]:
    salt_hex = secrets.token_hex(16)
    return _hash(password, salt_hex), salt_hex


def verify_password(password: str, pwd_hash: str, salt_hex: str) -> bool:
    if not password or not pwd_hash or not salt_hex:
        return False
    try:
        return hmac.compare_digest(_hash(password, salt_hex), pwd_hash)
    except (ValueError, TypeError):
        return False


def generate_password(length: int = 12) -> str:
    return "".join(secrets.choice(_PWD_ALPHABET) for _ in range(length))


# ---------------------------------------------------------------- 查询
def _row_to_user(row) -> User:
    return User(
        id=int(row["id"]),
        username=row["username"],
        display_name=row["display_name"] or "",
        role=row["role"] or ROLE_MEMBER,
        quota_daily=int(row["quota_daily"] or 0),
        library_slug=row["library_slug"] or "",
        active=bool(row["active"]),
        created_at=row["created_at"] or "",
        last_login=row["last_login"] or "",
    )


def get_user(username: str) -> Optional[User]:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return _row_to_user(row) if row else None


def list_users() -> List[User]:
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM users ORDER BY (role = 'admin') DESC, created_at"
        ).fetchall()
    return [_row_to_user(row) for row in rows]


def count_users() -> Dict[str, int]:
    with db.connect() as connection:
        total = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active = connection.execute(
            "SELECT COUNT(*) FROM users WHERE active = 1"
        ).fetchone()[0]
        admins = connection.execute(
            "SELECT COUNT(*) FROM users WHERE role = ? AND active = 1", (ROLE_ADMIN,)
        ).fetchone()[0]
    return {"total": int(total), "active": int(active), "admins": int(admins)}


# ---------------------------------------------------------------- 写
def ensure_bootstrap() -> Optional[Tuple[str, str]]:
    """没有任何账号时创建管理员。

    返回 (用户名, 自动生成的明文密码) —— 密码只在这一刻可见，调用方负责
    显示一次并提示尽快修改；配置了 `APP_ADMIN_PASSWORD` 时返回明文密码为 None。
    """
    db.init_db()
    if count_users()["total"] > 0:
        return None

    from backend.config import ADMIN_PASSWORD, ADMIN_USER

    password = ADMIN_PASSWORD or generate_password()
    user, plain = create_user(
        ADMIN_USER,
        password=password,
        display_name="系统管理员",
        role=ROLE_ADMIN,
        quota_daily=0,
    )
    logger.warning(
        "已创建初始管理员 %s，密码：%s（请登录后立即修改）",
        user.username,
        plain,
    )
    return (user.username, None if ADMIN_PASSWORD else plain)


def create_user(
    username: str,
    password: Optional[str] = None,
    display_name: str = "",
    role: str = ROLE_MEMBER,
    quota_daily: Optional[int] = None,
    library_slug: str = "",
) -> Tuple[User, str]:
    """新建账号，返回 (用户, 明文密码)。密码为空时自动生成。"""
    cleaned = (username or "").strip()
    if not _USERNAME_RE.match(cleaned):
        raise ValueError("用户名需为 3~32 位字母/数字/下划线/点/短横线")
    if role not in ROLES:
        raise ValueError(f"未知角色：{role}")
    if get_user(cleaned) is not None:
        raise ValueError(f"用户名「{cleaned}」已存在")

    plain = password or generate_password()
    if len(plain) < 6:
        raise ValueError("密码至少 6 位")
    pwd_hash, salt_hex = hash_password(plain)
    quota = (
        quota_daily
        if quota_daily is not None
        else (0 if role == ROLE_ADMIN else GUEST_QUOTA if role == ROLE_GUEST else MEMBER_QUOTA)
    )
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO users "
            "(username, display_name, pwd_hash, salt, role, quota_daily, library_slug,"
            " active, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                cleaned,
                (display_name or cleaned).strip()[:32],
                pwd_hash,
                salt_hex,
                role,
                int(quota),
                library_slug or "",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()
    user = get_user(cleaned)
    assert user is not None
    return user, plain


def create_guests(
    prefix: str = "guest",
    count: int = 5,
    password: Optional[str] = None,
    quota_daily: Optional[int] = None,
) -> List[Tuple[User, str]]:
    """批量创建访客账号（答辩 / 试用场景），返回 [(用户, 明文密码)]。

    用户名按「前缀 + 两位序号」生成（guest01、guest02…），自动跳过已存在的名字；
    角色固定为 guest（只读 + 每日配额），避免这个入口被用来批量造管理员。
    密码只在这一刻返回，库里只存 scrypt 哈希。
    """
    cleaned = (prefix or "guest").strip()
    if not _PREFIX_RE.match(cleaned):
        raise ValueError("前缀需为 1~26 位字母/数字/下划线/点/短横线")
    total = int(count)
    if not 1 <= total <= 50:
        raise ValueError("一次最多创建 50 个账号")
    if password and len(password) < 6:
        raise ValueError("统一密码至少 6 位")

    existing = {user.username for user in list_users()}
    names: List[str] = []
    index = 1
    while len(names) < total:
        candidate = f"{cleaned}{index:02d}"
        index += 1
        if candidate in existing:
            continue
        names.append(candidate)

    quota = GUEST_QUOTA if quota_daily is None else max(0, int(quota_daily))
    prepared: List[Tuple[str, str, str, str]] = []
    for name in names:
        plain = password or generate_password()
        pwd_hash, salt_hex = hash_password(plain)
        prepared.append((name, pwd_hash, salt_hex, plain))

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with db.connect() as connection:
        connection.executemany(
            "INSERT INTO users "
            "(username, display_name, pwd_hash, salt, role, quota_daily, library_slug,"
            " active, created_at) VALUES (?, ?, ?, ?, ?, ?, '', 1, ?)",
            [
                (name, name, pwd_hash, salt_hex, ROLE_GUEST, quota, now)
                for name, pwd_hash, salt_hex, _plain in prepared
            ],
        )
        connection.commit()

    created: List[Tuple[User, str]] = []
    for name, _hash, _salt, plain in prepared:
        user = get_user(name)
        if user is not None:
            created.append((user, plain))
    return created


def set_password(username: str, password: Optional[str] = None) -> str:
    plain = password or generate_password()
    if len(plain) < 6:
        raise ValueError("密码至少 6 位")
    pwd_hash, salt_hex = hash_password(plain)
    with db.connect() as connection:
        connection.execute(
            "UPDATE users SET pwd_hash = ?, salt = ? WHERE username = ?",
            (pwd_hash, salt_hex, username),
        )
        connection.commit()
    return plain


def set_role(username: str, role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"未知角色：{role}")
    user = get_user(username)
    if user is None:
        raise ValueError("用户不存在")
    if user.role == ROLE_ADMIN and role != ROLE_ADMIN and count_users()["admins"] <= 1:
        raise ValueError("至少要保留一个启用的管理员")
    with db.connect() as connection:
        connection.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
        connection.commit()


def set_active(username: str, active: bool) -> None:
    user = get_user(username)
    if user is None:
        raise ValueError("用户不存在")
    if not active and user.role == ROLE_ADMIN and count_users()["admins"] <= 1:
        raise ValueError("至少要保留一个启用的管理员")
    with db.connect() as connection:
        connection.execute(
            "UPDATE users SET active = ? WHERE username = ?", (1 if active else 0, username)
        )
        connection.commit()


def set_quota(username: str, quota_daily: int) -> None:
    with db.connect() as connection:
        connection.execute(
            "UPDATE users SET quota_daily = ? WHERE username = ?",
            (max(0, int(quota_daily)), username),
        )
        connection.commit()


def assign_library(username: str, library_slug: str) -> None:
    with db.connect() as connection:
        connection.execute(
            "UPDATE users SET library_slug = ? WHERE username = ?",
            (library_slug or "", username),
        )
        connection.commit()


def delete_user(username: str, actor: str) -> None:
    user = get_user(username)
    if user is None:
        raise ValueError("用户不存在")
    if username == actor:
        raise ValueError("不能删除当前登录的账号")
    if user.role == ROLE_ADMIN and count_users()["admins"] <= 1:
        raise ValueError("至少要保留一个启用的管理员")
    with db.connect() as connection:
        connection.execute("DELETE FROM users WHERE username = ?", (username,))
        connection.commit()


def touch_login(username: str) -> None:
    with db.connect() as connection:
        connection.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (time.strftime("%Y-%m-%d %H:%M:%S"), username),
        )
        connection.commit()


def check_password(user: User, password: str) -> bool:
    pwd_hash, salt_hex = _credentials(user.username)
    return verify_password(password, pwd_hash, salt_hex)


def _credentials(username: str) -> Tuple[str, str]:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT pwd_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return "", ""
    return row["pwd_hash"], row["salt"]


# ---------------------------------------------------------------- 配额与用量
def today_count(username: str) -> int:
    today = time.strftime("%Y-%m-%d")
    with db.connect() as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM qa_log WHERE username = ? AND ts LIKE ?",
                (username, f"{today}%"),
            ).fetchone()[0]
        )


def remaining_quota(user: User) -> Optional[int]:
    """剩余提问次数；None 表示不限。"""
    if not user.quota_daily:
        return None
    return max(0, user.quota_daily - today_count(user.username))


def record_question(
    username: str,
    library: str,
    question_chars: int,
    answer_chars: int,
    latency_ms: int,
    model: str = "",
    ok: bool = True,
    error: str = "",
    grounded: Optional[bool] = None,
    confidence: Optional[float] = None,
    tool_steps: int = 0,
    refused: bool = False,
) -> None:
    grounded_value = -1 if grounded is None else (1 if grounded else 0)
    confidence_value = -1.0 if confidence is None else float(confidence)
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO qa_log "
            "(ts, username, library, question_chars, answer_chars, latency_ms, model, ok, error,"
            " grounded, confidence, tool_steps, refused)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                username,
                library,
                int(question_chars),
                int(answer_chars),
                int(latency_ms),
                model,
                1 if ok else 0,
                (error or "")[:300],
                grounded_value,
                confidence_value,
                max(0, int(tool_steps)),
                1 if refused else 0,
            ),
        )
        connection.commit()
