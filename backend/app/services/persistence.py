"""Asynchronous write queue between live sessions and the database.

Live counting must never stall because the database is slow or down. Sessions push
operations here; a background task writes them in order, in batches, from a worker thread.

* Transient failures (network, 5xx, locked DB) are retried with exponential backoff while
  new operations keep queueing. The health endpoint reports the backlog.
* Permanent failures (bad data, constraint errors) drop only the failing group and log it,
  so one bad row can't block everything behind it.
* Every row carries a client-generated id and inserts ignore duplicates, so a retry after
  an ambiguous failure never double-counts.
* The queue is bounded; if the database stays down long enough to fill it, the oldest
  operations are dropped and counted in ``status['dropped']``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from ..storage.base import Store, StoreError

log = logging.getLogger(__name__)

# Operation kinds
SESSION_START = "session_start"
SESSION_UPDATE = "session_update"
EVENTS = "events"
SNAPSHOTS = "snapshots"
_BATCHABLE = {EVENTS, SNAPSHOTS}


class PersistenceQueue:
    def __init__(self, store: Store, flush_interval: float = 1.0, max_pending: int = 50_000) -> None:
        self.store = store
        self.flush_interval = flush_interval
        self.max_pending = max_pending
        self._ops: deque[tuple[str, Any]] = deque()
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._closing = False
        self.last_error: str | None = None
        self.last_success: float | None = None
        self.dropped = 0
        self.failed = 0

    # -- producer side -----------------------------------------------------------------
    def put(self, kind: str, payload: Any) -> None:
        self._ops.append((kind, payload))
        while len(self._ops) > self.max_pending:
            self._ops.popleft()
            self.dropped += 1
        if self._wake is not None:
            self._wake.set()

    @property
    def pending(self) -> int:
        return len(self._ops)

    @property
    def status(self) -> dict:
        return {
            "ok": self.last_error is None,
            "pending": self.pending,
            "last_error": self.last_error,
            "dropped": self.dropped,
            "failed": self.failed,
        }

    # -- lifecycle -------------------------------------------------------------------------
    async def start(self) -> None:
        self._wake = asyncio.Event()
        self._closing = False
        self._task = asyncio.create_task(self._run(), name="persistence-queue")

    async def stop(self, timeout: float = 10.0) -> None:
        self._closing = True
        if self._wake:
            self._wake.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout)
            except asyncio.TimeoutError:
                self._task.cancel()
                log.error("Persistence queue stopped with %d unsaved operations", self.pending)

    async def flush(self, timeout: float = 10.0) -> bool:
        """Wait until the queue is empty (used by tests and graceful shutdown)."""
        deadline = time.monotonic() + timeout
        while self._ops and time.monotonic() < deadline:
            if self._wake:
                self._wake.set()
            await asyncio.sleep(0.02)
        return not self._ops

    # -- worker -----------------------------------------------------------------------------
    def _next_group(self) -> tuple[str, Any, int]:
        kind, payload = self._ops[0]
        if kind not in _BATCHABLE:
            return kind, payload, 1
        rows = list(payload)
        n = 1
        for k, p in list(self._ops)[1:]:
            if k != kind or len(rows) >= 500:
                break
            rows.extend(p)
            n += 1
        return kind, rows, n

    def _apply(self, kind: str, payload: Any) -> None:
        if kind == SESSION_START:
            self.store.insert_session(payload)
        elif kind == SESSION_UPDATE:
            session_id, fields = payload
            self.store.update_session(session_id, fields)
        elif kind == EVENTS:
            self.store.insert_events(payload)
        elif kind == SNAPSHOTS:
            self.store.insert_snapshots(payload)
        else:  # pragma: no cover
            raise StoreError(f"unknown operation {kind}", retryable=False)

    async def _run(self) -> None:
        backoff = 1.0
        assert self._wake is not None
        while True:
            if not self._ops:
                if self._closing:
                    return
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), self.flush_interval)
                except asyncio.TimeoutError:
                    pass
                continue
            kind, payload, n = self._next_group()
            try:
                await asyncio.to_thread(self._apply, kind, payload)
            except StoreError as exc:
                if not exc.retryable:
                    log.error("Dropping %s after permanent storage error: %s", kind, exc)
                    self.failed += n
                    for _ in range(n):
                        self._ops.popleft()
                    continue
                self.last_error = str(exc)
                log.warning("Storage write failed (%s), retrying in %.0fs; %d pending", exc, backoff, self.pending)
                if self._closing and backoff > 4:
                    log.error("Giving up on %d unsaved operations at shutdown", self.pending)
                    return
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            except Exception as exc:  # noqa: BLE001 - unexpected bug: drop group, keep the worker alive
                log.exception("Unexpected error writing %s", kind)
                self.failed += n
                self.last_error = f"{type(exc).__name__}: {exc}"
                for _ in range(n):
                    self._ops.popleft()
                continue
            for _ in range(n):
                self._ops.popleft()
            self.last_error = None
            self.last_success = time.time()
            backoff = 1.0
