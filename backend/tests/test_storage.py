"""Storage layer: SQLite store, Supabase store (against a local mock PostgREST server) and the
async persistence queue."""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from backend.app.services.persistence import EVENTS, SESSION_START, SESSION_UPDATE, SNAPSHOTS, PersistenceQueue
from backend.app.storage.base import StoreError
from backend.app.storage.sqlite_store import SQLiteStore
from backend.app.storage.supabase_store import SupabaseStore


def _session(sid=None, started="2026-01-05T10:00:00.000+00:00", name="Door"):
    return {
        "id": sid or str(uuid.uuid4()), "name": name, "started_at": started, "status": "running",
        "detector": "yolo", "settings": {"confidence": 0.4}, "total_entries": 0, "total_exits": 0,
        "max_occupancy": 0, "peak_live_count": 0,
    }


def _event(sid, kind="entry", at="2026-01-05T10:05:00.000+00:00", occ=1):
    return {"id": str(uuid.uuid4()), "session_id": sid, "event_type": kind, "occurred_at": at,
            "occupancy_after": occ, "live_count": 1}


@pytest.fixture()
def sqlite_store(tmp_path):
    store = SQLiteStore(tmp_path / "t.db")
    store.init()
    yield store
    store.close()


# --------------------------------------------------------------------------- SQLite
def test_sqlite_roundtrip(sqlite_store):
    s = _session()
    sqlite_store.insert_session(s)
    sqlite_store.insert_session(s)  # duplicate insert is ignored (idempotent retry)
    ev = [_event(s["id"]), _event(s["id"], "exit", "2026-01-05T10:06:00.000+00:00", 0)]
    sqlite_store.insert_events(ev)
    sqlite_store.insert_events(ev)  # retried batch is not double counted
    sqlite_store.insert_snapshots([{
        "id": str(uuid.uuid4()), "session_id": s["id"], "captured_at": "2026-01-05T10:00:30.000+00:00",
        "avg_live_count": 1.5, "max_live_count": 2, "occupancy": 1, "fps": 12.0,
    }])
    sqlite_store.update_session(s["id"], {"total_entries": 1, "total_exits": 1, "status": "completed",
                                          "heatmap": {"cols": 2, "rows": 1, "max": 3, "cells": [[1, 3]]}})

    got = sqlite_store.get_session(s["id"])
    assert got["total_entries"] == 1 and got["status"] == "completed"
    assert got["settings"] == {"confidence": 0.4}
    assert got["heatmap"]["cells"] == [[1, 3]]
    listed = sqlite_store.list_sessions(10, 0)
    assert len(listed) == 1 and "heatmap" not in listed[0]

    events = sqlite_store.get_events("2026-01-05T00:00:00+00:00", "2026-01-06T00:00:00+00:00")
    assert [e["event_type"] for e in events] == ["entry", "exit"]
    only_exits = sqlite_store.get_events("2026-01-05T00:00:00+00:00", "2026-01-06T00:00:00+00:00", ("exit",))
    assert len(only_exits) == 1
    recent = sqlite_store.recent_events(5)
    assert recent[0]["event_type"] == "exit" and recent[0]["session_name"] == "Door"
    assert len(sqlite_store.get_snapshots(s["id"])) == 1


def test_sqlite_rejects_unknown_fields_and_bad_event(sqlite_store):
    s = _session()
    sqlite_store.insert_session(s)
    with pytest.raises(StoreError) as exc:
        sqlite_store.update_session(s["id"], {"id": "hijack"})
    assert exc.value.retryable is False
    with pytest.raises(StoreError) as exc:
        sqlite_store.insert_events([_event(s["id"], kind="face_seen")])
    assert exc.value.retryable is False


def test_sqlite_settings_and_interrupted(sqlite_store):
    assert sqlite_store.get_settings() is None
    sqlite_store.save_settings({"confidence": 0.5}, "2026-01-01T00:00:00Z")
    sqlite_store.save_settings({"confidence": 0.6}, "2026-01-02T00:00:00Z")
    assert sqlite_store.get_settings() == {"confidence": 0.6}
    sqlite_store.insert_session(_session())
    assert sqlite_store.mark_interrupted_sessions() == 1
    assert sqlite_store.list_sessions(5, 0)[0]["status"] == "interrupted"


def test_sqlite_delete_session_cascades(sqlite_store):
    s = _session()
    sqlite_store.insert_session(s)
    sqlite_store.insert_events([_event(s["id"])])
    assert sqlite_store.delete_session(s["id"]) is True
    assert sqlite_store.get_session(s["id"]) is None
    assert sqlite_store.recent_events(5) == []
    assert sqlite_store.delete_session(s["id"]) is False


# --------------------------------------------------------------------------- Supabase (mock)
class _MockPostgrest(BaseHTTPRequestHandler):
    """Tiny stand-in for Supabase's /rest/v1 that records requests and answers from memory."""

    store: dict = {}
    requests: list = []
    fail_next: list = []

    def log_message(self, *args):  # silence
        pass

    def _reply(self, code, body=None):
        raw = b"" if body is None else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        type(self).requests.append({
            "method": self.command, "path": self.path, "body": body,
            "apikey": self.headers.get("apikey"), "auth": self.headers.get("Authorization"),
            "prefer": self.headers.get("Prefer"),
        })
        if type(self).fail_next:
            self._reply(type(self).fail_next.pop(0), {"message": "boom"})
            return
        table = self.path.split("?")[0].replace("/rest/v1/", "")
        rows = type(self).store.setdefault(table, [])
        if self.command == "POST":
            ids = {r.get("id") for r in rows}
            rows.extend(r for r in body if r.get("id") not in ids)
            self._reply(201)
        elif self.command == "PATCH":
            out = []
            for r in rows:
                if "status=eq.running" in self.path and r.get("status") == "running":
                    r.update(body)
                    out.append(r)
            self._reply(200, out)
        else:
            self._reply(200, rows[:1] if "limit=1" in self.path else rows)

    do_GET = do_POST = do_PATCH = do_DELETE = _handle


@pytest.fixture()
def mock_supabase():
    _MockPostgrest.store = {}
    _MockPostgrest.requests = []
    _MockPostgrest.fail_next = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockPostgrest)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", _MockPostgrest
    server.shutdown()


def test_supabase_store_requests(mock_supabase):
    url, mock = mock_supabase
    store = SupabaseStore(url, "service-key-123")
    assert "service-key-123" not in repr(store)
    store.init()
    s = _session()
    store.insert_session(s)
    store.insert_events([_event(s["id"])])
    assert mock.store["sessions"][0]["id"] == s["id"]
    assert len(mock.store["events"]) == 1
    post = [r for r in mock.requests if r["method"] == "POST"][0]
    assert post["apikey"] == "service-key-123" and post["auth"] == "Bearer service-key-123"
    assert "ignore-duplicates" in post["prefer"]
    assert store.mark_interrupted_sessions() == 1
    assert store.get_session(s["id"])["id"] == s["id"]


def test_supabase_error_classification(mock_supabase):
    url, mock = mock_supabase
    store = SupabaseStore(url, "k")
    mock.fail_next = [503]
    with pytest.raises(StoreError) as exc:
        store.ping()
    assert exc.value.retryable is True
    mock.fail_next = [400]
    with pytest.raises(StoreError) as exc:
        store.insert_events([_event(str(uuid.uuid4()))])
    assert exc.value.retryable is False
    dead = SupabaseStore("http://127.0.0.1:9", "k", timeout=2)
    with pytest.raises(StoreError) as exc:
        dead.ping()
    assert exc.value.retryable is True


# --------------------------------------------------------------------------- queue
class FlakyStore(SQLiteStore):
    """SQLite store whose first ``fail_times`` writes raise a retryable error."""

    def __init__(self, fail_times):
        super().__init__(":memory:")
        self.fail_times = fail_times
        self.event_batches = []

    def insert_events(self, rows):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise StoreError("database is locked", retryable=True)
        self.event_batches.append(len(rows))
        super().insert_events(rows)


def test_queue_retries_and_batches(monkeypatch):
    async def scenario():
        store = FlakyStore(fail_times=1)
        store.init()
        q = PersistenceQueue(store, flush_interval=0.05)
        # make backoff instant for the test
        real_sleep = asyncio.sleep
        monkeypatch.setattr("backend.app.services.persistence.asyncio.sleep", lambda s: real_sleep(0.01))
        await q.start()
        s = _session()
        q.put(SESSION_START, s)
        for i in range(5):
            q.put(EVENTS, [_event(s["id"], at=f"2026-01-05T10:0{i}:00.000+00:00")])
        q.put(SNAPSHOTS, [])
        q.put(SESSION_UPDATE, (s["id"], {"total_entries": 5}))
        assert await q.flush(5)
        await q.stop()
        return store, q

    store, q = asyncio.run(scenario())
    assert store.get_session(store.list_sessions(1, 0)[0]["id"])["total_entries"] == 5
    assert sum(store.event_batches) == 5 and len(store.event_batches) == 1  # one batched write after retry
    assert q.status["ok"] and q.failed == 0


def test_queue_drops_permanent_failures_and_continues():
    async def scenario():
        store = SQLiteStore(":memory:")
        store.init()
        q = PersistenceQueue(store, flush_interval=0.05)
        await q.start()
        q.put(EVENTS, [_event("no-such-session")])  # FK violation -> permanent
        s = _session()
        q.put(SESSION_START, s)
        assert await q.flush(5)
        await q.stop()
        return store, q

    store, q = asyncio.run(scenario())
    assert q.failed == 1
    assert len(store.list_sessions(5, 0)) == 1


def test_queue_bounded():
    q = PersistenceQueue(SQLiteStore(":memory:"), max_pending=3)
    for i in range(5):
        q.put(EVENTS, [i])
    assert q.pending == 3 and q.dropped == 2
