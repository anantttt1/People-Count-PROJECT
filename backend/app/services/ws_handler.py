"""WebSocket protocol for live counting - independent of the web framework.

One ``StreamConnection`` per browser tab. The FastAPI route only moves bytes/text in and
out; everything else (validation, sessions, persistence) lives here so it can be tested
without a server.

Flow control is stop-and-wait: the browser sends a frame, waits for the ``result`` and only
then sends the next one. Messages are handled one at a time, so a slow CPU lowers the FPS
instead of building a queue of stale frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from ..schemas import ConfigMessage, ControlMessage, DetectionSettings, StartMessage, client_message_adapter
from ..vision.frames import FrameError, decode_jpeg
from .persistence import EVENTS, SESSION_START, SESSION_UPDATE, SNAPSHOTS, PersistenceQueue
from .stream_session import StreamSession, iso

log = logging.getLogger(__name__)

SUMMARY_EVERY_SECONDS = 10.0
MAX_CONSECUTIVE_ERRORS = 50
MAX_TEXT_BYTES = 16_384


@dataclass
class LiveRegistry:
    """Active sessions across all connections + the concurrency limit."""

    max_sessions: int
    sessions: dict[str, StreamSession] = field(default_factory=dict)

    def can_start(self) -> bool:
        return len(self.sessions) < self.max_sessions

    def add(self, s: StreamSession) -> None:
        self.sessions[s.id] = s

    def remove(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def public(self) -> list[dict]:
        return [
            {"id": s.id, "name": s.name, "started_at": iso(s.started_at), "counts": s.counts(),
             "fps": round(s.fps, 1), "paused": getattr(s, "paused", False)}
            for s in list(self.sessions.values())
        ]


@dataclass
class StreamContext:
    """What a connection needs from the application."""

    settings: Any                                   # config.Settings
    queue: PersistenceQueue
    registry: LiveRegistry
    get_detector: Callable[[], tuple[Any, Any]]     # -> (detector | None, DetectorInfo)
    get_defaults: Callable[[], DetectionSettings]


class StreamConnection:
    def __init__(self, ctx: StreamContext, send: Callable[[dict], Awaitable[None]], clock=time.time) -> None:
        self.ctx = ctx
        self._send = send
        self.clock = clock
        self.session: StreamSession | None = None
        self.paused = False
        self.errors = 0
        self.closed = False
        self._last_summary = 0.0

    # ------------------------------------------------------------------ helpers
    async def send(self, msg: dict) -> None:
        if self.closed:
            return
        try:
            await self._send(msg)
        except Exception:  # noqa: BLE001 - client went away mid-send; the read loop will end
            self.closed = True

    async def error(self, code: str, message: str) -> None:
        self.errors += 1
        await self.send({"type": "error", "code": code, "message": message})

    @property
    def too_many_errors(self) -> bool:
        return self.errors >= MAX_CONSECUTIVE_ERRORS

    async def hello(self) -> None:
        detector, info = self.ctx.get_detector()
        await self.send({
            "type": "ready",
            "detector": info.public(),
            "limits": {
                "max_frame_bytes": self.ctx.settings.max_frame_bytes,
                "max_frame_dimension": self.ctx.settings.max_frame_dimension,
            },
            "defaults": self.ctx.get_defaults().model_dump(),
        })

    # ------------------------------------------------------------------ text messages
    async def handle_text(self, text: str) -> None:
        if len(text) > MAX_TEXT_BYTES:
            await self.error("message_too_large", "Control message is too large.")
            return
        try:
            msg = client_message_adapter.validate_python(json.loads(text))
        except json.JSONDecodeError:
            await self.error("bad_json", "Message is not valid JSON.")
            return
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ()))
            await self.error("invalid_message", f"{where}: {first.get('msg', 'invalid value')}")
            return

        if isinstance(msg, StartMessage):
            await self._start(msg.name, msg.settings)
        elif isinstance(msg, ConfigMessage):
            await self._config(msg.settings)
        elif isinstance(msg, ControlMessage):
            if msg.type == "ping":
                await self.send({"type": "pong", "ts": self.clock()})
            elif msg.type == "stop":
                summary = await self._end_session()
                await self.send({"type": "stopped", **summary})
            elif msg.type == "reset":
                await self._reset()
            elif msg.type == "pause":
                self.paused = True
                if self.session:
                    self.session.pause()
                await self.send({"type": "paused"})
            elif msg.type == "resume":
                self.paused = False
                if self.session:
                    self.session.resume()
                await self.send({"type": "resumed"})
        self.errors = 0

    async def _start(self, name: str, settings: DetectionSettings | None) -> None:
        if self.session is not None:
            await self.error("already_running", "A session is already running on this connection.")
            return
        detector, info = self.ctx.get_detector()
        if detector is None:
            code = "detector_loading" if info.status == "loading" else "detector_failed"
            await self.error(code, info.message or "The AI model is still loading. Try again in a few seconds.")
            return
        if not self.ctx.registry.can_start():
            await self.error("too_many_sessions",
                             f"The server is already running {self.ctx.registry.max_sessions} sessions.")
            return
        s = StreamSession(
            detector,
            settings or self.ctx.get_defaults(),
            name,
            snapshot_interval=self.ctx.settings.snapshot_interval_seconds,
            clock=self.clock,
            detector_name=f"{info.name}:{info.model}",
        )
        self.session = s
        self.paused = False
        self._last_summary = self.clock()
        self.ctx.registry.add(s)
        self.ctx.queue.put(SESSION_START, s.session_row())
        log.info("Session %s started (%s)", s.id, name)
        await self.send({
            "type": "session",
            "session": {"id": s.id, "name": s.name, "started_at": iso(s.started_at), "settings": s.settings.model_dump()},
            "counts": s.counts(),
        })

    async def _config(self, settings: DetectionSettings) -> None:
        if self.session is None:
            await self.error("no_session", "Start a session before changing its settings.")
            return
        self.session.apply_settings(settings)
        # store the new settings with the session so history shows what was used
        self.ctx.queue.put(SESSION_UPDATE, (self.session.id, {"settings": settings.model_dump()}))
        await self.send({"type": "config_applied", "settings": settings.model_dump()})

    async def _reset(self) -> None:
        if self.session is None:
            await self.error("no_session", "No session to reset.")
            return
        name, settings = self.session.name, self.session.settings
        await self._end_session()
        await self._start(name, settings)

    async def _end_session(self) -> dict:
        s = self.session
        if s is None:
            return {"session_id": None, "counts": None}
        self.session = None
        self.ctx.registry.remove(s.id)
        snap = s.snapshot_if_due(force=True)
        if snap:
            self.ctx.queue.put(SNAPSHOTS, [snap])
        self.ctx.queue.put(SESSION_UPDATE, (s.id, s.summary_fields(ended=True)))
        log.info("Session %s ended: %s", s.id, s.counts())
        return {"session_id": s.id, "counts": s.counts()}

    # ------------------------------------------------------------------ frames
    async def handle_bytes(self, data: bytes) -> None:
        s = self.session
        if s is None:
            await self.error("no_session", "Send a start message before sending frames.")
            return
        if self.paused:
            await self.error("paused", "Session is paused; frame ignored.")
            return
        cfg = self.ctx.settings
        try:
            frame = await asyncio.to_thread(decode_jpeg, data, cfg.max_frame_bytes, cfg.max_frame_dimension)
        except FrameError as exc:
            await self.error("bad_frame", str(exc))
            return
        try:
            message, rows = await asyncio.to_thread(s.process_frame, frame)
        except Exception as exc:  # noqa: BLE001 - keep the connection alive; report and log
            log.exception("Frame processing failed in session %s", s.id)
            await self.error("processing_failed", f"Detection failed on this frame ({type(exc).__name__}).")
            return
        finally:
            del frame  # the image is never kept
        if self.session is not s:  # stopped/reset while this frame was being processed
            return
        if rows:
            self.ctx.queue.put(EVENTS, rows)
        snap = s.snapshot_if_due()
        if snap:
            self.ctx.queue.put(SNAPSHOTS, [snap])
        now = self.clock()
        if rows or now - self._last_summary >= SUMMARY_EVERY_SECONDS:
            self._last_summary = now
            self.ctx.queue.put(SESSION_UPDATE, (s.id, s.summary_fields()))
        self.errors = 0
        await self.send(message)

    async def close(self) -> None:
        if self.session is not None:
            await self._end_session()
        self.closed = True
