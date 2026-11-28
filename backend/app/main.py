"""FastAPI application: REST API, live WebSocket and the static frontend.

Start with ``python run.py`` (see README). ``create_app`` takes optional overrides so tests
can inject settings, a store and a detector loader.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import FRONTEND_DIR, Settings, get_settings
from .logging_setup import configure_logging
from .schemas import DetectionSettings
from .security import (
    COOKIE_NAME,
    LoginRateLimiter,
    host_allowed,
    is_authorized,
    origin_allowed,
    origin_of,
    security_headers,
)
from .services.analytics import AnalyticsError
from .services.api_service import ApiService
from .services.persistence import PersistenceQueue
from .services.ws_handler import LiveRegistry, StreamConnection, StreamContext
from .storage import build_store
from .storage.base import Store, StoreError
from .vision.detector import DetectorInfo, load_detector

log = logging.getLogger(__name__)

# API paths reachable without a session (everything else under /api needs one when auth is on)
PUBLIC_API = {"/api/health", "/api/config", "/api/auth/status", "/api/auth/login", "/api/auth/logout"}
PAGES = {"/": "index.html", "/dashboard": "dashboard.html", "/analytics": "analytics.html", "/privacy": "privacy.html"}


@dataclass
class AppState:
    settings: Settings
    store: Store
    queue: PersistenceQueue
    service: ApiService
    registry: LiveRegistry
    limiter: LoginRateLimiter = field(default_factory=LoginRateLimiter)
    detector: Any = None
    detector_info: DetectorInfo = field(default_factory=DetectorInfo)
    storage_error: str | None = None
    defaults: DetectionSettings = field(default_factory=DetectionSettings)
    started_at: float = field(default_factory=time.time)


def _default_loader(settings: Settings):
    return load_detector(
        settings.yolo_model,
        settings.yolo_fallback_model,
        settings.yolo_device,
        settings.yolo_imgsz,
        settings.allow_hog_fallback,
        model_sha256=settings.yolo_model_sha256,
        allow_download=settings.allow_model_download,
    )


def create_app(
    settings: Settings | None = None,
    store: Store | None = None,
    detector_loader: Callable[[Settings], tuple[Any, DetectorInfo]] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    store = store or build_store(settings)
    loader = detector_loader or _default_loader
    queue = PersistenceQueue(store)
    state = AppState(settings, store, queue, ApiService(store), LiveRegistry(settings.max_concurrent_sessions))

    async def load_model() -> None:
        try:
            det, info = await asyncio.to_thread(loader, settings)
            state.detector, state.detector_info = det, info
        except Exception as exc:  # noqa: BLE001 - surfaced in /api/health and the UI
            log.error("No detector available: %s", exc)
            state.detector_info = DetectorInfo(status="failed", message=str(exc))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        log.info("Smart Vision AI %s starting (%s, storage=%s, auth=%s)", __version__, settings.environment,
                 store.name, "on" if settings.auth_enabled else "off")
        try:
            await asyncio.to_thread(store.init)
            n = await asyncio.to_thread(store.mark_interrupted_sessions)
            if n:
                log.warning("%d session(s) were still running at the last shutdown; marked interrupted", n)
            state.defaults = await asyncio.to_thread(state.service.get_defaults)
        except StoreError as exc:
            if store.name == "sqlite":
                raise
            # Supabase unreachable: keep counting live, queue writes and retry in the background.
            state.storage_error = str(exc)
            log.error("Storage unavailable at startup (live counting still works): %s", exc)
        await queue.start()
        model_task = asyncio.create_task(load_model())
        app.state.app_state = state
        try:
            yield
        finally:
            model_task.cancel()
            await queue.stop(timeout=10)
            store.close()
            log.info("Stopped")

    app = FastAPI(
        title="Smart Vision AI",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if not settings.is_production else None,
    )
    app.state.app_state = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    headers = security_headers(settings)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if not host_allowed(settings, request.headers.get("host")):
            return JSONResponse({"detail": "Host not allowed"}, status_code=400)
        if path.startswith("/api/"):
            if request.method in ("POST", "PUT", "DELETE"):
                origin = request.headers.get("origin")
                if origin is None and request.headers.get("referer"):
                    origin = origin_of(request.headers["referer"])
                if not origin_allowed(settings, origin, request.headers.get("host")):
                    return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)
            if path not in PUBLIC_API and not is_authorized(settings, request.cookies.get(COOKIE_NAME)):
                return JSONResponse({"detail": "Login required"}, status_code=401)
        response = await call_next(request)
        for k, v in headers.items():
            response.headers.setdefault(k, v)
        if path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(AnalyticsError)
    async def analytics_error(_: Request, exc: AnalyticsError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(StoreError)
    async def store_error(_: Request, exc: StoreError):
        log.error("Storage error: %s", exc)
        return JSONResponse({"detail": "Database unavailable. Live counting continues; data is queued."},
                            status_code=503)

    from .api.routes import build_router  # noqa: E402 - avoids a circular import

    app.include_router(build_router(state))

    # ------------------------------------------------------------------ WebSocket
    def get_detector():
        return state.detector, state.detector_info

    ctx = StreamContext(settings, queue, state.registry, get_detector, lambda: state.defaults)

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket):
        h = websocket.headers
        if not host_allowed(settings, h.get("host")) or not origin_allowed(settings, h.get("origin"), h.get("host")):
            log.warning("WebSocket rejected: origin %r host %r", h.get("origin"), h.get("host"))
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if not is_authorized(settings, websocket.cookies.get(COOKIE_NAME)):
            await websocket.send_text(json.dumps({"type": "error", "code": "auth_required", "message": "Login required"}))
            await websocket.close(code=4401)
            return

        async def send(msg: dict) -> None:
            await websocket.send_text(json.dumps(msg, separators=(",", ":")))

        conn = StreamConnection(ctx, send)
        await conn.hello()
        try:
            while not conn.closed:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await conn.handle_bytes(message["bytes"])
                elif message.get("text") is not None:
                    await conn.handle_text(message["text"])
                if conn.too_many_errors:
                    await websocket.close(code=1008)
                    break
        except WebSocketDisconnect:
            pass
        except RuntimeError as exc:  # receive after close
            log.debug("WebSocket closed: %s", exc)
        finally:
            await conn.close()

    # ------------------------------------------------------------------ frontend
    def page_handler(path):
        # No parameters on purpose: nothing from the request can influence which file is served.
        async def page():
            if not path.is_file():
                log.error("Frontend file missing: %s", path)
                return JSONResponse({"detail": "Page not found"}, status_code=404)
            return FileResponse(path, headers={"Cache-Control": "no-cache"})
        return page

    for route, filename in PAGES.items():
        app.add_api_route(route, page_handler(FRONTEND_DIR / filename), methods=["GET"], include_in_schema=False)

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    return app
