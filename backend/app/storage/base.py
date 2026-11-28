"""Storage interface. Implementations are synchronous; async code calls them via a thread.

Only numeric analytics are stored: session totals, crossing/capacity events, occupancy
snapshots, a coarse heatmap and admin settings. No images, no identities.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

SESSION_UPDATABLE = frozenset({
    "name", "ended_at", "status", "total_entries", "total_exits", "max_occupancy",
    "peak_live_count", "frames_processed", "settings", "heatmap",
})
EVENT_TYPES = ("entry", "exit", "capacity_exceeded", "capacity_cleared")


class StoreError(RuntimeError):
    """A storage operation failed. ``retryable`` is False for errors that will never succeed
    on retry (bad data, constraint violations, auth problems)."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class Store(ABC):
    name: str = "store"

    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def ping(self) -> None: ...

    @abstractmethod
    def mark_interrupted_sessions(self) -> int:
        """Sessions still 'running' at startup were cut off by a crash/restart."""

    @abstractmethod
    def insert_session(self, row: dict) -> None: ...

    @abstractmethod
    def update_session(self, session_id: str, fields: dict) -> None: ...

    @abstractmethod
    def insert_events(self, rows: list[dict]) -> None: ...

    @abstractmethod
    def insert_snapshots(self, rows: list[dict]) -> None: ...

    @abstractmethod
    def list_sessions(self, limit: int, offset: int) -> list[dict]:
        """Newest first, without the heatmap column."""

    @abstractmethod
    def get_session(self, session_id: str) -> dict | None: ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and (via ON DELETE CASCADE) its events and snapshots."""

    @abstractmethod
    def sessions_between(self, start_iso: str, end_iso: str) -> list[dict]:
        """Sessions with start <= started_at < end (list columns only), oldest first."""

    @abstractmethod
    def get_events(self, start_iso: str, end_iso: str, event_types: tuple[str, ...] | None = None) -> list[dict]:
        """Events with start <= occurred_at < end, oldest first."""

    @abstractmethod
    def recent_events(self, limit: int) -> list[dict]:
        """Newest first, each with ``session_name``."""

    @abstractmethod
    def get_snapshots(self, session_id: str) -> list[dict]: ...

    @abstractmethod
    def get_settings(self) -> dict | None: ...

    @abstractmethod
    def save_settings(self, data: dict, updated_at: str) -> None: ...

    def close(self) -> None:  # pragma: no cover - optional
        pass


def check_session_fields(fields: dict) -> None:
    bad = set(fields) - SESSION_UPDATABLE
    if bad:
        raise StoreError(f"cannot update session fields: {sorted(bad)}", retryable=False)
