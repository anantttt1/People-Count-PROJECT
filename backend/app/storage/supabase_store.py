"""Supabase PostgreSQL store via the PostgREST API (``/rest/v1``).

Runs on the server only and authenticates with the service-role key, which bypasses RLS.
The tables have RLS enabled with no anon policies (see database/schema.sql), so the public
anon key can't read or write anything even if it leaks.

Uses only the standard library (urllib) so there is no extra client dependency.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from urllib.parse import quote

from .base import EVENT_TYPES, Store, StoreError, check_session_fields

log = logging.getLogger(__name__)
PAGE = 1000          # PostgREST default max rows per request on Supabase
TIMEOUT = 15

_LIST_COLS = (
    "id,name,started_at,ended_at,status,detector,total_entries,total_exits,"
    "max_occupancy,peak_live_count,frames_processed"
)


def _q(value: str) -> str:
    return quote(str(value), safe="")


class SupabaseStore(Store):
    name = "supabase"

    def __init__(self, url: str, service_key: str, timeout: float = TIMEOUT) -> None:
        self.base = url.rstrip("/") + "/rest/v1"
        self._key = service_key
        self.timeout = timeout

    def __repr__(self) -> str:  # never print the key
        return f"SupabaseStore({self.base!r})"

    # -- HTTP -------------------------------------------------------------------------
    def _request(self, method: str, path: str, body=None, prefer: str | None = None):
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
        }
        data = None
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            retryable = exc.code >= 500 or exc.code in (408, 429)
            table = path.split("?")[0].lstrip("/")
            raise StoreError(f"Supabase {method} {table} failed: HTTP {exc.code}: {detail}", retryable) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise StoreError(f"Cannot reach Supabase: {exc}", retryable=True) from exc
        if not raw:
            return None
        return json.loads(raw)

    def _select_all(self, path: str) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        sep = "&" if "?" in path else "?"
        while True:
            page = self._request("GET", f"{path}{sep}limit={PAGE}&offset={offset}") or []
            rows.extend(page)
            if len(page) < PAGE:
                return rows
            offset += PAGE

    # -- lifecycle ----------------------------------------------------------------------
    def init(self) -> None:
        # Tables are created by database/schema.sql in the Supabase SQL editor.
        try:
            self.ping()
        except StoreError as exc:
            raise StoreError(
                f"{exc}. Check SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY and that database/schema.sql was run.",
                exc.retryable,
            ) from exc

    def ping(self) -> None:
        self._request("GET", "/app_settings?select=id&limit=1")

    def mark_interrupted_sessions(self) -> int:
        rows = self._request(
            "PATCH", "/sessions?status=eq.running", {"status": "interrupted"}, prefer="return=representation"
        ) or []
        return len(rows)

    # -- writes ---------------------------------------------------------------------------
    def insert_session(self, row: dict) -> None:
        self._request("POST", "/sessions", [row], prefer="resolution=ignore-duplicates,return=minimal")

    def update_session(self, session_id: str, fields: dict) -> None:
        check_session_fields(fields)
        if fields:
            self._request("PATCH", f"/sessions?id=eq.{_q(session_id)}", fields, prefer="return=minimal")

    def insert_events(self, rows: list[dict]) -> None:
        if rows:
            self._request("POST", "/events", rows, prefer="resolution=ignore-duplicates,return=minimal")

    def insert_snapshots(self, rows: list[dict]) -> None:
        if rows:
            self._request("POST", "/occupancy_snapshots", rows, prefer="resolution=ignore-duplicates,return=minimal")

    def save_settings(self, data: dict, updated_at: str) -> None:
        self._request(
            "POST", "/app_settings", [{"id": 1, "data": data, "updated_at": updated_at}],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    # -- reads ------------------------------------------------------------------------------
    def list_sessions(self, limit: int, offset: int) -> list[dict]:
        return self._request(
            "GET", f"/sessions?select={_LIST_COLS}&order=started_at.desc&limit={int(limit)}&offset={int(offset)}"
        ) or []

    def get_session(self, session_id: str) -> dict | None:
        rows = self._request("GET", f"/sessions?select=*&id=eq.{_q(session_id)}&limit=1") or []
        return rows[0] if rows else None

    def delete_session(self, session_id: str) -> bool:
        rows = self._request(
            "DELETE", f"/sessions?id=eq.{_q(session_id)}", prefer="return=representation"
        ) or []
        return len(rows) > 0

    def sessions_between(self, start_iso: str, end_iso: str) -> list[dict]:
        return self._select_all(
            f"/sessions?select={_LIST_COLS}&started_at=gte.{_q(start_iso)}&started_at=lt.{_q(end_iso)}"
            "&order=started_at.asc,id.asc"
        )

    def get_events(self, start_iso: str, end_iso: str, event_types: tuple[str, ...] | None = None) -> list[dict]:
        types = ",".join(t for t in (event_types or EVENT_TYPES) if t in EVENT_TYPES)
        return self._select_all(
            "/events?select=id,session_id,event_type,occurred_at,occupancy_after,live_count"
            f"&occurred_at=gte.{_q(start_iso)}&occurred_at=lt.{_q(end_iso)}"
            f"&event_type=in.({types})&order=occurred_at.asc,id.asc"
        )

    def recent_events(self, limit: int) -> list[dict]:
        rows = self._request(
            "GET",
            "/events?select=id,session_id,event_type,occurred_at,occupancy_after,live_count,sessions(name)"
            f"&order=occurred_at.desc&limit={int(limit)}",
        ) or []
        for r in rows:
            r["session_name"] = (r.pop("sessions", None) or {}).get("name")
        return rows

    def get_snapshots(self, session_id: str) -> list[dict]:
        return self._select_all(
            f"/occupancy_snapshots?select=captured_at,avg_live_count,max_live_count,occupancy,fps"
            f"&session_id=eq.{_q(session_id)}&order=captured_at.asc"
        )

    def get_settings(self) -> dict | None:
        rows = self._request("GET", "/app_settings?select=data&id=eq.1&limit=1") or []
        return rows[0]["data"] if rows else None
