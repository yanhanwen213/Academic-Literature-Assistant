"""平台侧配置：账号、权限、审计相关的开关与路径。

只读环境变量，不负责加载 `.env`（项目入口已经加载过），保证后端模块
可以被脚本/测试直接 import 而不产生副作用。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# 平台数据库：账号 / 审计 / 用量，与会话库（assistant.db）分开存放
PLATFORM_DB_PATH = str(_env_path("APP_PLATFORM_DB", PROJECT_ROOT / "data" / "platform.db"))

# 是否启用登录门禁。置 0 时前端会以本地管理员身份直接进入（仅限本机调试）
AUTH_ENABLED = _env_bool("APP_AUTH_ENABLED", True)

# 首次启动自动创建的管理员
ADMIN_USER = (os.getenv("APP_ADMIN_USER") or "admin").strip() or "admin"
ADMIN_PASSWORD = (os.getenv("APP_ADMIN_PASSWORD") or "").strip()

# 新用户的默认每日提问配额（0 = 不限）
MEMBER_QUOTA = int(os.getenv("APP_MEMBER_QUOTA", "200") or 200)
GUEST_QUOTA = int(os.getenv("APP_GUEST_QUOTA", "30") or 30)

# 登录失败限流
LOGIN_MAX_FAILS = int(os.getenv("APP_LOGIN_MAX_FAILS", "5") or 5)
LOGIN_LOCK_SECONDS = int(os.getenv("APP_LOGIN_LOCK_SECONDS", "60") or 60)

# 审计日志保留天数（0 = 不清理）
AUDIT_KEEP_DAYS = int(os.getenv("APP_AUDIT_KEEP_DAYS", "180") or 180)

# 备份目录（与会话库备份共用 data/backups）
BACKUP_DIR = _env_path("APP_BACKUP_DIR", PROJECT_ROOT / "data" / "backups")
