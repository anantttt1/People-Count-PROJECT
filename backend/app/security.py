"""Access control, CSRF/origin checks, login rate limiting and security headers.

Auth model (simple on purpose - this is a single-site tool, not a multi-user product):
* If ``ACCESS_TOKEN`` is empty (local development) everything is open, but the server binds
  to 127.0.0.1 by default and only allowed Host/Origin headers are accepted.
* If ``ACCESS_TOKEN`` is set, the dashboard asks for it once; the server answers with an
  HttpOnly, SameSite cookie holding an HMAC-signed expiry. The token itself is never stored
  in the browser. Every API call and the WebSocket check that cookie.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit

from .config import Settings

COOKIE_NAME = "sv_session"
_COOKIE_VERSION = "v1"


# ----------------------------------------------------------------------------- cookies
def _sign(secret: str, payload: str) -> str:
    mac = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def make_session_cookie(settings: Settings, now: float | None = None) -> tuple[str, int]:
    """Return (cookie value, max_age seconds)."""
    max_age = settings.session_ttl_hours * 3600
    expires = int((now or time.time()) + max_age)
    payload = f"{_COOKIE_VERSION}.{expires}.{secrets.token_urlsafe(12)}"
    return f"{payload}.{_sign(settings.secret_key, payload)}", max_age


def verify_session_cookie(settings: Settings, value: str | None, now: float | None = None) -> bool:
    if not value or len(value) > 256:
        return False
    try:
        version, expires, nonce, sig = value.split(".")
    except ValueError:
        return False
    if version != _COOKIE_VERSION:
        return False
    if not hmac.compare_digest(sig, _sign(settings.secret_key, f"{version}.{expires}.{nonce}")):
        return False
    try:
        return int(expires) > (now or time.time())
    except ValueError:
        return False


def is_authorized(settings: Settings, cookie_value: str | None) -> bool:
    return (not settings.auth_enabled) or verify_session_cookie(settings, cookie_value)


def check_token(settings: Settings, token: str) -> bool:
    if not settings.auth_enabled:
        return True
    return hmac.compare_digest(token.encode(), settings.access_token.encode())


# ----------------------------------------------------------------------------- rate limit
class LoginRateLimiter:
    """At most ``max_attempts`` failed logins per client per ``window`` seconds."""

    def __init__(self, max_attempts: int = 5, window: float = 300.0) -> None:
        self.max_attempts = max_attempts
        self.window = window
        self._fails: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, key: str, now: float) -> deque[float]:
        q = self._fails[key]
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def retry_after(self, key: str, now: float | None = None) -> int:
        now = now or time.monotonic()
        with self._lock:
            q = self._trim(key, now)
            if len(q) < self.max_attempts:
                return 0
            return max(1, int(self.window - (now - q[0])) + 1)

    def failed(self, key: str, now: float | None = None) -> None:
        now = now or time.monotonic()
        with self._lock:
            self._trim(key, now).append(now)
            if len(self._fails) > 10_000:  # bound memory under a spray of client keys
                for k in [k for k, v in self._fails.items() if not v][:5_000]:
                    del self._fails[k]

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


# ----------------------------------------------------------------------------- origin / host
def origin_of(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}".lower()


def origin_allowed(settings: Settings, origin: str | None, host: str | None, scheme: str = "http") -> bool:
    """True if a request's Origin is this server itself or in ALLOWED_ORIGINS.

    A missing Origin is accepted: browsers always send it on WebSocket handshakes and on
    cross-site POST/PUT/DELETE, so a missing header means a same-origin GET or a non-browser
    client (which still needs the session cookie when auth is on).
    """
    if not origin:
        return True
    if origin == "null":
        return False
    origin = origin.rstrip("/").lower()
    if host:
        same = {f"http://{host}".lower(), f"https://{host}".lower()}
        if origin in same:
            return True
    return origin in {o.lower() for o in settings.allowed_origins}


def host_allowed(settings: Settings, host_header: str | None) -> bool:
    """DNS-rebinding protection: only answer for configured host names."""
    if "*" in settings.allowed_hosts:
        return True
    if not host_header:
        return False
    host = host_header.lower()
    if host.startswith("["):  # IPv6 literal, e.g. [::1]:8000
        name = host.split("]")[0] + "]"
    else:
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return name in settings.allowed_hosts


# ----------------------------------------------------------------------------- headers
def security_headers(settings: Settings) -> dict[str, str]:
    csp = "; ".join([
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data: blob:",
        "media-src 'self' blob: mediastream:",
        "font-src 'self'",
        "connect-src 'self'",          # same-origin fetch + ws/wss (CSP Level 3)
        "worker-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])
    headers = {
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(self), microphone=(), geolocation=(), payment=(), usb=()",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    if settings.is_production:
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return headers


def client_key(peer_host: str | None, forwarded_for: str | None, trust_proxy: bool) -> str:
    """Identify a client for rate limiting. X-Forwarded-For is trusted only behind a proxy."""
    if trust_proxy and forwarded_for:
        return forwarded_for.split(",")[0].strip()[:64]
    return (peer_host or "unknown")[:64]
