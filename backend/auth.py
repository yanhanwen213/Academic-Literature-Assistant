"""登录会话：口令校验、失败限流、会话快照。

不直接依赖 Streamlit：`login()` 返回状态和用户对象，由前端写进
`st.session_state`；`resolve()` 每次 rerun 都回数据库核对一遍，
这样管理员停用账号 / 改角色会立即生效，不用等用户重新登录。
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from backend import audit, users
from backend.config import AUTH_ENABLED, LOGIN_LOCK_SECONDS, LOGIN_MAX_FAILS

logger = logging.getLogger(__name__)

# username -> [失败次数, 锁定截止时间]
_fails: Dict[str, List[float]] = {}
_lock = threading.Lock()


def enabled() -> bool:
    return AUTH_ENABLED


def _locked_for(username: str) -> float:
    state = _fails.get(username)
    if not state:
        return 0.0
    remaining = state[1] - time.time()
    return max(0.0, remaining)


def _record_failure(username: str) -> None:
    with _lock:
        state = _fails.setdefault(username, [0, 0.0])
        state[0] += 1
        if state[0] >= LOGIN_MAX_FAILS:
            state[1] = time.time() + LOGIN_LOCK_SECONDS
            state[0] = 0


def _clear_failures(username: str) -> None:
    with _lock:
        _fails.pop(username, None)


def locked_seconds(username: str) -> float:
    """剩余锁定时间（秒），供登录页提示。"""
    return _locked_for((username or "").strip())


def login(username: str, password: str) -> Tuple[str, Optional[users.User]]:
    """返回 (状态, 用户)。状态：ok / empty / locked / bad / inactive。"""
    name = (username or "").strip()
    if not name or not password:
        return "empty", None

    remaining = _locked_for(name)
    if remaining > 0:
        return "locked", None

    user = users.get_user(name)
    if user is None:
        _record_failure(name)
        return "bad", None
    if not user.active:
        return "inactive", None
    if not users.check_password(user, password):
        _record_failure(name)
        audit.log(name, user.role, "login_failed", detail="密码错误", ok=False)
        return "bad", None

    _clear_failures(name)
    users.touch_login(name)
    audit.log(name, user.role, "login", detail="登录成功")
    return "ok", user


def snapshot(user: users.User) -> Dict[str, object]:
    return {
        "username": user.username,
        "display_name": user.display_name or user.username,
        "role": user.role,
        "library_slug": user.library_slug,
    }


def resolve(auth_state: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """把 session_state 里的登录快照与数据库对齐；失效时返回 None。"""
    if not AUTH_ENABLED:
        return {
            "username": "local",
            "display_name": "本地调试",
            "role": users.ROLE_ADMIN,
            "library_slug": "",
        }
    if not auth_state:
        return None
    username = str(auth_state.get("username") or "").strip()
    if not username:
        return None
    user = users.get_user(username)
    if user is None or not user.active:
        return None
    return snapshot(user)


def logout(username: str, role: str) -> None:
    audit.log(username, role, "logout")
