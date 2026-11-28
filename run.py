"""Start Smart Vision AI:  python run.py

Reads HOST / PORT / SSL_* / TRUST_PROXY from the environment (or .env). See .env.example.
"""
from __future__ import annotations

import sys

import uvicorn

from backend.app.config import ConfigError, load_settings


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    scheme = "https" if settings.ssl_certfile else "http"
    shown = "localhost" if settings.host in ("127.0.0.1", "0.0.0.0") else settings.host
    print(f"Smart Vision AI -> {scheme}://{shown}:{settings.port}  (Ctrl+C to stop)")
    if settings.host == "0.0.0.0" and not settings.ssl_certfile and not settings.trust_proxy:
        print("Note: phones and other computers need HTTPS for camera access. See docs/DEPLOYMENT.md.")

    uvicorn.run(
        "backend.app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        ssl_certfile=settings.ssl_certfile or None,
        ssl_keyfile=settings.ssl_keyfile or None,
        proxy_headers=settings.trust_proxy,
        forwarded_allow_ips="*" if settings.trust_proxy else None,
        # Reject oversized WebSocket messages at the protocol level (frame limit + headroom).
        ws_max_size=settings.max_frame_bytes + 64 * 1024,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        server_header=False,
        log_config=None,  # our own logging setup (backend/app/logging_setup.py)
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
