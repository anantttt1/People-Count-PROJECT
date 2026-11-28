"""Application configuration loaded from environment variables (and an optional .env file).

Only server-side code reads these values. Nothing in here is ever sent to the browser
except the explicitly public fields returned by ``/api/config``.
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when the environment configuration is invalid or unsafe."""


def _load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional; env vars still work
        log.warning(".env found but python-dotenv is not installed; ignoring .env")
        return
    load_dotenv(env_file, override=False)


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_origins: tuple[str, ...] = ("http://localhost:8000", "http://127.0.0.1:8000")
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")
    trust_proxy: bool = False
    ssl_certfile: str = ""
    ssl_keyfile: str = ""

    access_token: str = ""
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32), repr=False)
    cookie_samesite: Literal["strict", "lax", "none"] = "strict"
    session_ttl_hours: int = 12

    storage_backend: Literal["sqlite", "supabase"] = "sqlite"
    sqlite_path: Path = REPO_ROOT / "data" / "smartvision.db"
    supabase_url: str = ""
    supabase_service_role_key: str = field(default="", repr=False)

    yolo_model: str = "yolo26n.onnx"
    yolo_fallback_model: str = "yolo11n.onnx"
    yolo_model_sha256: str = ""
    allow_model_download: bool = True
    yolo_device: str = ""
    yolo_imgsz: int = 640
    allow_hog_fallback: bool = True

    max_concurrent_sessions: int = 4
    max_frame_bytes: int = 2_000_000
    max_frame_dimension: int = 1920
    snapshot_interval_seconds: int = 30

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.access_token)

    @property
    def cookie_secure(self) -> bool:
        # Secure cookies need HTTPS; production must run behind HTTPS.
        return self.is_production or self.cookie_samesite == "none"


def load_settings() -> Settings:
    """Build Settings from the environment, validating anything security-relevant."""
    _load_dotenv()

    environment = _str("ENVIRONMENT", "development").lower()
    if environment not in {"development", "production"}:
        raise ConfigError("ENVIRONMENT must be 'development' or 'production'")

    origins = tuple(
        o.strip().rstrip("/") for o in _str("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()
    )
    for origin in origins:
        if origin == "*":
            raise ConfigError("ALLOWED_ORIGINS must list explicit origins; '*' is not allowed")
        if not origin.startswith(("http://", "https://")):
            raise ConfigError(f"Invalid origin in ALLOWED_ORIGINS: {origin!r}")

    hosts = tuple(h.strip().lower() for h in _str("ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if h.strip())
    if not hosts:
        raise ConfigError("ALLOWED_HOSTS must list at least one host name")
    if environment == "production" and "*" in hosts:
        raise ConfigError("ALLOWED_HOSTS='*' is not allowed in production")

    ssl_certfile, ssl_keyfile = _str("SSL_CERTFILE"), _str("SSL_KEYFILE")
    if bool(ssl_certfile) != bool(ssl_keyfile):
        raise ConfigError("Set both SSL_CERTFILE and SSL_KEYFILE (or neither)")

    access_token = _str("ACCESS_TOKEN")
    secret_key = _str("SECRET_KEY")
    if environment == "production":
        if len(access_token) < 16:
            raise ConfigError("ACCESS_TOKEN (16+ characters) is required when ENVIRONMENT=production")
        if len(secret_key) < 32:
            raise ConfigError("SECRET_KEY (32+ characters) is required when ENVIRONMENT=production")
    if not secret_key:
        # Development convenience: random per process. Sessions reset on restart.
        secret_key = secrets.token_hex(32)

    samesite = _str("COOKIE_SAMESITE", "strict").lower()
    if samesite not in {"strict", "lax", "none"}:
        raise ConfigError("COOKIE_SAMESITE must be strict, lax or none")

    storage_backend = _str("STORAGE_BACKEND", "sqlite").lower()
    if storage_backend not in {"sqlite", "supabase"}:
        raise ConfigError("STORAGE_BACKEND must be 'sqlite' or 'supabase'")
    supabase_url = _str("SUPABASE_URL").rstrip("/")
    supabase_key = _str("SUPABASE_SERVICE_ROLE_KEY")
    if storage_backend == "supabase":
        if not supabase_url or not supabase_key:
            raise ConfigError("STORAGE_BACKEND=supabase requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        local = supabase_url.startswith(("http://localhost", "http://127.0.0.1"))
        if not supabase_url.startswith("https://") and not local:
            raise ConfigError("SUPABASE_URL must use https://")

    sqlite_path = Path(_str("SQLITE_PATH", "data/smartvision.db"))
    if not sqlite_path.is_absolute():
        sqlite_path = REPO_ROOT / sqlite_path

    yolo_sha = _str("YOLO_MODEL_SHA256").lower()
    if yolo_sha and (len(yolo_sha) != 64 or any(c not in "0123456789abcdef" for c in yolo_sha)):
        raise ConfigError("YOLO_MODEL_SHA256 must be a 64-character hex SHA-256 digest")

    yolo_imgsz = _int("YOLO_IMGSZ", 640, 160, 1920)
    if yolo_imgsz % 32:
        raise ConfigError("YOLO_IMGSZ must be a multiple of 32")

    return Settings(
        environment=environment,  # type: ignore[arg-type]
        log_level=_str("LOG_LEVEL", "INFO").upper(),
        host=_str("HOST", "127.0.0.1"),
        port=_int("PORT", 8000, 1, 65535),
        allowed_origins=origins,
        allowed_hosts=hosts,
        trust_proxy=_bool(os.getenv("TRUST_PROXY"), False),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        access_token=access_token,
        secret_key=secret_key,
        cookie_samesite=samesite,  # type: ignore[arg-type]
        session_ttl_hours=_int("SESSION_TTL_HOURS", 12, 1, 24 * 30),
        storage_backend=storage_backend,  # type: ignore[arg-type]
        sqlite_path=sqlite_path,
        supabase_url=supabase_url,
        supabase_service_role_key=supabase_key,
        yolo_model=_str("YOLO_MODEL", "yolo26n.onnx"),
        yolo_fallback_model=_str("YOLO_FALLBACK_MODEL", "yolo11n.onnx"),
        yolo_model_sha256=yolo_sha,
        allow_model_download=_bool(os.getenv("ALLOW_MODEL_DOWNLOAD"), True),
        yolo_device=_str("YOLO_DEVICE"),
        yolo_imgsz=yolo_imgsz,
        allow_hog_fallback=_bool(os.getenv("ALLOW_HOG_FALLBACK"), True),
        max_concurrent_sessions=_int("MAX_CONCURRENT_SESSIONS", 4, 1, 64),
        max_frame_bytes=_int("MAX_FRAME_BYTES", 2_000_000, 50_000, 10_000_000),
        max_frame_dimension=_int("MAX_FRAME_DIMENSION", 1920, 320, 4096),
        snapshot_interval_seconds=_int("SNAPSHOT_INTERVAL_SECONDS", 30, 5, 3600),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
