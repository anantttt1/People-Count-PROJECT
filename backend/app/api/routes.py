"""REST API. Every handler is a thin wrapper: FastAPI validates the input, a service does the work.

Handlers are plain ``def`` functions, so FastAPI runs them in its thread pool and the
synchronous store calls never block the event loop (which also carries the live WebSocket).

Access control and the cross-site check are applied by the middleware in ``main.py``:
everything below ``/api`` except ``PUBLIC_API`` needs a valid session cookie when
``ACCESS_TOKEN`` is set.
"""
from __future__ import annotations

import time
import uuid
from datetime import date
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse

from .. import __version__
from ..schemas import DetectionSettings, LoginRequest
from ..security import COOKIE_NAME, check_token, client_key, is_authorized, make_session_cookie

if TYPE_CHECKING:  # pragma: no cover
    from ..main import AppState

TZ = Query(None, max_length=64, description="IANA time zone of the viewer, e.g. Asia/Kolkata")


def _csv(body: str, filename: str) -> PlainTextResponse:
    # BOM so Excel opens UTF-8 session names correctly
    return PlainTextResponse(
        "﻿" + body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def build_router(state: "AppState") -> APIRouter:
    r = APIRouter(prefix="/api")
    settings = state.settings
    service = state.service

    # ------------------------------------------------------------------ public
    @r.get("/health")
    def health(request: Request) -> dict:
        info = state.detector_info
        storage_ok = state.storage_error is None and state.queue.status["ok"]
        body = {
            "status": "ok" if info.status == "ready" and storage_ok else "degraded",
            "version": __version__,
            "detector": info.status,
            "storage": "ok" if storage_ok else "error",
        }
        # Details (model names, error messages, queue size) only for signed-in users.
        if is_authorized(settings, request.cookies.get(COOKIE_NAME)):
            body["details"] = {
                "detector": info.public(),
                "storage": {
                    "backend": state.store.name,
                    "startup_error": state.storage_error,
                    "queue": state.queue.status,
                },
                "live_sessions": len(state.registry.sessions),
                "uptime_seconds": round(time.time() - state.started_at),
            }
        return body

    @r.get("/config")
    def public_config() -> dict:
        return {
            "version": __version__,
            "auth_enabled": settings.auth_enabled,
            "storage_backend": state.store.name,
            "limits": {
                "max_frame_bytes": settings.max_frame_bytes,
                "max_frame_dimension": settings.max_frame_dimension,
                "max_concurrent_sessions": settings.max_concurrent_sessions,
            },
        }

    @r.get("/auth/status")
    def auth_status(request: Request) -> dict:
        return {
            "auth_enabled": settings.auth_enabled,
            "authenticated": is_authorized(settings, request.cookies.get(COOKIE_NAME)),
        }

    @r.post("/auth/login")
    def login(body: LoginRequest, request: Request, response: Response) -> dict:
        if not settings.auth_enabled:
            return {"authenticated": True, "auth_enabled": False}
        key = client_key(request.client.host if request.client else None,
                         request.headers.get("x-forwarded-for"), settings.trust_proxy)
        wait = state.limiter.retry_after(key)
        if wait:
            raise HTTPException(429, "Too many failed attempts. Try again later.",
                                headers={"Retry-After": str(wait)})
        if not check_token(settings, body.token):
            state.limiter.failed(key)
            raise HTTPException(401, "Wrong access token")
        state.limiter.succeeded(key)
        value, max_age = make_session_cookie(settings)
        response.set_cookie(
            COOKIE_NAME, value, max_age=max_age, path="/", httponly=True,
            secure=settings.cookie_secure, samesite=settings.cookie_samesite,
        )
        return {"authenticated": True, "auth_enabled": True}

    @r.post("/auth/logout")
    def logout(response: Response) -> dict:
        response.delete_cookie(COOKIE_NAME, path="/", httponly=True,
                               secure=settings.cookie_secure, samesite=settings.cookie_samesite)
        return {"authenticated": False}

    # ------------------------------------------------------------------ live + settings
    @r.get("/live")
    def live() -> dict:
        return {"sessions": state.registry.public()}

    @r.get("/settings")
    def get_settings() -> DetectionSettings:
        return state.defaults

    @r.put("/settings")
    def put_settings(body: DetectionSettings) -> DetectionSettings:
        saved = service.save_defaults(body)
        state.defaults = saved  # new sessions start with these
        return saved

    # ------------------------------------------------------------------ history
    @r.get("/sessions")
    def sessions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0, le=1_000_000)) -> dict:
        rows = service.list_sessions(limit, offset)
        live_ids = set(state.registry.sessions)
        for row in rows:
            row["live"] = row["id"] in live_ids
        return {"sessions": rows, "limit": limit, "offset": offset}

    @r.get("/sessions/{session_id}")
    def session_detail(session_id: uuid.UUID) -> dict:
        row = service.get_session(str(session_id))
        if row is None:
            raise HTTPException(404, "Session not found")
        row["live"] = row["id"] in state.registry.sessions
        return row

    @r.delete("/sessions/{session_id}")
    def delete_session(session_id: uuid.UUID) -> dict:
        sid = str(session_id)
        if sid in state.registry.sessions:
            raise HTTPException(409, "Stop this session before deleting it")
        if not service.delete_session(sid):
            raise HTTPException(404, "Session not found")
        return {"deleted": sid}

    @r.get("/events/recent")
    def recent_events(limit: int = Query(30, ge=1, le=200)) -> dict:
        return {"events": service.recent_events(limit)}

    # ------------------------------------------------------------------ analytics
    @r.get("/analytics/summary")
    def summary(tz: str | None = TZ) -> dict:
        return service.summary(tz, len(state.registry.sessions))

    @r.get("/analytics/hourly")
    def hourly(tz: str | None = TZ, day: date | None = Query(None, alias="date")) -> dict:
        return service.hourly(tz, day)

    @r.get("/analytics/daily")
    def daily(tz: str | None = TZ, days: int = Query(14, ge=1, le=366)) -> dict:
        return service.daily(tz, days)

    @r.get("/analytics/weekly")
    def weekly(tz: str | None = TZ, weeks: int = Query(8, ge=1, le=104)) -> dict:
        return service.weekly(tz, weeks)

    @r.get("/analytics/peak-hours")
    def peak_hours(tz: str | None = TZ, days: int = Query(30, ge=1, le=366)) -> dict:
        return service.peak_hours(tz, days)

    # ------------------------------------------------------------------ CSV export
    @r.get("/export/events.csv")
    def export_events(start: date, end: date, tz: str | None = TZ) -> PlainTextResponse:
        return _csv(service.events_csv(tz, start, end), f"smartvision-events-{start}-to-{end}.csv")

    @r.get("/export/sessions.csv")
    def export_sessions(tz: str | None = TZ) -> PlainTextResponse:
        return _csv(service.sessions_csv(tz), f"smartvision-sessions-{date.today()}.csv")

    @r.get("/export/daily.csv")
    def export_daily(tz: str | None = TZ, days: int = Query(30, ge=1, le=366)) -> PlainTextResponse:
        return _csv(service.daily_csv(tz, days), f"smartvision-daily-{days}d.csv")

    return r
