"""SQLite store - zero-setup local persistence (default). Same tables as database/schema.sql.

Inserts use ``ON CONFLICT(id) DO NOTHING`` rather than ``INSERT OR IGNORE``: the latter would
also silently swallow CHECK / NOT NULL violations, hiding bad rows instead of reporting them.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .base import EVENT_TYPES, Store, StoreError, check_session_fields

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    ended_at         TEXT,
    status           TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','interrupted')),
    detector         TEXT,
    total_entries    INTEGER NOT NULL DEFAULT 0,
    total_exits      INTEGER NOT NULL DEFAULT 0,
    max_occupancy    INTEGER NOT NULL DEFAULT 0,
    peak_live_count  INTEGER NOT NULL DEFAULT 0,
    frames_processed INTEGER NOT NULL DEFAULT 0,
    settings         TEXT,
    heatmap          TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions (started_at DESC);

CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL CHECK (event_type IN ('entry','exit','capacity_exceeded','capacity_cleared')),
    occurred_at     TEXT NOT NULL,
    occupancy_after INTEGER NOT NULL,
    live_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id, occurred_at);

CREATE TABLE IF NOT EXISTS occupancy_snapshots (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    captured_at    TEXT NOT NULL,
    avg_live_count REAL NOT NULL,
    max_live_count INTEGER NOT NULL,
    occupancy      INTEGER NOT NULL,
    fps            REAL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON occupancy_snapshots (session_id, captured_at);

CREATE TABLE IF NOT EXISTS app_settings (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    data       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_JSON_COLS = ("settings", "heatmap")
_LIST_COLS = (
    "id, name, started_at, ended_at, status, detector, total_entries, total_exits, "
    "max_occupancy, peak_live_count, frames_processed"
)


class SQLiteStore(Store):
    name = "sqlite"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path) if str(path) != ":memory:" else path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # -- helpers ------------------------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StoreError("store not initialised", retryable=False)
        return self._conn

    def _run(self, fn):
        with self._lock:
            try:
                return fn(self._db())
            except sqlite3.IntegrityError as exc:
                raise StoreError(f"SQLite constraint error: {exc}", retryable=False) from exc
            except sqlite3.OperationalError as exc:
                raise StoreError(f"SQLite error: {exc}", retryable=True) from exc

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        d = dict(row)
        for col in _JSON_COLS:
            if col in d and isinstance(d[col], str):
                d[col] = json.loads(d[col])
        return d

    @staticmethod
    def _dump(fields: dict) -> dict:
        return {k: (json.dumps(v) if k in _JSON_COLS and v is not None else v) for k, v in fields.items()}

    # -- lifecycle ----------------------------------------------------------------
    def init(self) -> None:
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        if isinstance(self.path, Path):
            conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def ping(self) -> None:
        self._run(lambda db: db.execute("SELECT 1").fetchone())

    def mark_interrupted_sessions(self) -> int:
        def fn(db):
            cur = db.execute("UPDATE sessions SET status = 'interrupted' WHERE status = 'running'")
            db.commit()
            return cur.rowcount
        return self._run(fn)

    # -- writes ---------------------------------------------------------------------
    def insert_session(self, row: dict) -> None:
        data = self._dump(row)
        cols = ", ".join(data)
        marks = ", ".join("?" for _ in data)

        def fn(db):
            db.execute(f"INSERT INTO sessions ({cols}) VALUES ({marks}) ON CONFLICT(id) DO NOTHING", tuple(data.values()))
            db.commit()
        self._run(fn)

    def update_session(self, session_id: str, fields: dict) -> None:
        check_session_fields(fields)
        if not fields:
            return
        data = self._dump(fields)
        assignments = ", ".join(f"{k} = ?" for k in data)

        def fn(db):
            db.execute(f"UPDATE sessions SET {assignments} WHERE id = ?", (*data.values(), session_id))
            db.commit()
        self._run(fn)

    def insert_events(self, rows: list[dict]) -> None:
        if not rows:
            return

        def fn(db):
            db.executemany(
                "INSERT INTO events (id, session_id, event_type, occurred_at, occupancy_after, live_count) "
                "VALUES (:id, :session_id, :event_type, :occurred_at, :occupancy_after, :live_count) "
                "ON CONFLICT(id) DO NOTHING",
                rows,
            )
            db.commit()
        self._run(fn)

    def insert_snapshots(self, rows: list[dict]) -> None:
        if not rows:
            return

        def fn(db):
            db.executemany(
                "INSERT INTO occupancy_snapshots "
                "(id, session_id, captured_at, avg_live_count, max_live_count, occupancy, fps) "
                "VALUES (:id, :session_id, :captured_at, :avg_live_count, :max_live_count, :occupancy, :fps) "
                "ON CONFLICT(id) DO NOTHING",
                rows,
            )
            db.commit()
        self._run(fn)

    def save_settings(self, data: dict, updated_at: str) -> None:
        def fn(db):
            db.execute(
                "INSERT INTO app_settings (id, data, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                (json.dumps(data), updated_at),
            )
            db.commit()
        self._run(fn)

    # -- reads ----------------------------------------------------------------------
    def list_sessions(self, limit: int, offset: int) -> list[dict]:
        return self._run(lambda db: [
            self._row(r) for r in db.execute(
                f"SELECT {_LIST_COLS} FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?", (limit, offset)
            )
        ])

    def get_session(self, session_id: str) -> dict | None:
        def fn(db):
            r = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return self._row(r) if r else None
        return self._run(fn)

    def delete_session(self, session_id: str) -> bool:
        def fn(db):
            cur = db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            db.commit()
            return cur.rowcount > 0
        return self._run(fn)

    def sessions_between(self, start_iso: str, end_iso: str) -> list[dict]:
        return self._run(lambda db: [
            self._row(r) for r in db.execute(
                f"SELECT {_LIST_COLS} FROM sessions WHERE started_at >= ? AND started_at < ? ORDER BY started_at ASC",
                (start_iso, end_iso),
            )
        ])

    def get_events(self, start_iso: str, end_iso: str, event_types: tuple[str, ...] | None = None) -> list[dict]:
        types = tuple(t for t in (event_types or EVENT_TYPES) if t in EVENT_TYPES)
        marks = ", ".join("?" for _ in types)

        def fn(db):
            return [dict(r) for r in db.execute(
                f"SELECT id, session_id, event_type, occurred_at, occupancy_after, live_count FROM events "
                f"WHERE occurred_at >= ? AND occurred_at < ? AND event_type IN ({marks}) ORDER BY occurred_at ASC",
                (start_iso, end_iso, *types),
            )]
        return self._run(fn)

    def recent_events(self, limit: int) -> list[dict]:
        return self._run(lambda db: [dict(r) for r in db.execute(
            "SELECT e.id, e.session_id, e.event_type, e.occurred_at, e.occupancy_after, e.live_count, "
            "s.name AS session_name FROM events e JOIN sessions s ON s.id = e.session_id "
            "ORDER BY e.occurred_at DESC LIMIT ?", (limit,),
        )])

    def get_snapshots(self, session_id: str) -> list[dict]:
        return self._run(lambda db: [dict(r) for r in db.execute(
            "SELECT captured_at, avg_live_count, max_live_count, occupancy, fps FROM occupancy_snapshots "
            "WHERE session_id = ? ORDER BY captured_at ASC", (session_id,),
        )])

    def get_settings(self) -> dict | None:
        def fn(db):
            r = db.execute("SELECT data FROM app_settings WHERE id = 1").fetchone()
            return json.loads(r["data"]) if r else None
        return self._run(fn)
