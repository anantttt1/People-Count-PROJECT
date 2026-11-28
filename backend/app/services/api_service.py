"""Query logic behind the REST API: analytics, session history, CSV export, admin settings.

Synchronous on purpose (stores are synchronous); routes call these through a worker thread.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta, timezone

from ..schemas import DetectionSettings
from ..storage.base import Store
from . import analytics as A

log = logging.getLogger(__name__)

MAX_DAYS = 366
MAX_WEEKS = 104
_CSV_DANGEROUS = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value):
    """Neutralise spreadsheet formula injection (e.g. a session named '=HYPERLINK(...)')."""
    if isinstance(value, str) and value.startswith(_CSV_DANGEROUS):
        return "'" + value
    return value


def to_csv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(header)
    for r in rows:
        w.writerow([csv_safe(v) for v in r])
    return buf.getvalue()


class ApiService:
    def __init__(self, store: Store) -> None:
        self.store = store

    # ------------------------------------------------------------------ settings
    def get_defaults(self) -> DetectionSettings:
        data = self.store.get_settings()
        if not data:
            return DetectionSettings()
        try:
            return DetectionSettings.model_validate(data)
        except Exception:  # noqa: BLE001 - stored settings from an older version: fall back safely
            log.warning("Stored admin settings are invalid; using defaults")
            return DetectionSettings()

    def save_defaults(self, settings: DetectionSettings) -> DetectionSettings:
        self.store.save_settings(settings.model_dump(), datetime.now(timezone.utc).isoformat())
        return settings

    # ------------------------------------------------------------------ sessions
    def list_sessions(self, limit: int, offset: int) -> list[dict]:
        return self.store.list_sessions(limit, offset)

    def get_session(self, session_id: str) -> dict | None:
        s = self.store.get_session(session_id)
        if s is None:
            return None
        s["snapshots"] = self.store.get_snapshots(session_id)
        return s

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete_session(session_id)

    def recent_events(self, limit: int) -> list[dict]:
        return self.store.recent_events(limit)

    # ------------------------------------------------------------------ analytics
    def _events(self, start_day: date, days: int, tz) -> list[dict]:
        lo, hi = A.local_range(start_day, days, tz)
        return self.store.get_events(lo, hi)

    def summary(self, tz_name: str | None, live_sessions: int) -> dict:
        tz = A.get_zone(tz_name)
        today = A.today_in(tz)
        return A.summary(self._events(today, 1, tz), today, tz, live_sessions)

    def hourly(self, tz_name: str | None, day: date | None) -> dict:
        tz = A.get_zone(tz_name)
        day = day or A.today_in(tz)
        return A.hourly(self._events(day, 1, tz), day, tz)

    def daily(self, tz_name: str | None, days: int, end_day: date | None = None) -> dict:
        tz = A.get_zone(tz_name)
        days = _clamp(days, 1, MAX_DAYS)
        end_day = end_day or A.today_in(tz)
        start = end_day - timedelta(days=days - 1)
        lo, hi = A.local_range(start, days, tz)
        rows = A.daily(self.store.get_events(lo, hi), self.store.sessions_between(lo, hi), start, days, tz)
        return {"timezone": str(tz), "start": start.isoformat(), "end": end_day.isoformat(), "days": rows}

    def weekly(self, tz_name: str | None, weeks: int) -> dict:
        tz = A.get_zone(tz_name)
        weeks = _clamp(weeks, 1, MAX_WEEKS)
        first = A.week_start(A.today_in(tz)) - timedelta(weeks=weeks - 1)
        events = self._events(first, weeks * 7, tz)
        return {"timezone": str(tz), "weeks": A.weekly(events, first, weeks, tz)}

    def peak_hours(self, tz_name: str | None, days: int) -> dict:
        tz = A.get_zone(tz_name)
        days = _clamp(days, 1, MAX_DAYS)
        start = A.today_in(tz) - timedelta(days=days - 1)
        return {"timezone": str(tz), **A.peak_hours(self._events(start, days, tz), start, days, tz)}

    # ------------------------------------------------------------------ CSV
    def events_csv(self, tz_name: str | None, start: date, end: date) -> str:
        tz = A.get_zone(tz_name)
        if end < start:
            raise A.AnalyticsError("end date is before start date")
        days = (end - start).days + 1
        if days > MAX_DAYS:
            raise A.AnalyticsError(f"at most {MAX_DAYS} days per export")
        lo, hi = A.local_range(start, days, tz)
        events = self.store.get_events(lo, hi)
        names: dict[str, str] = {}
        for sid in {e["session_id"] for e in events}:
            s = self.store.get_session(sid)
            names[sid] = s["name"] if s else ""
        rows = []
        for e in events:
            t = A.parse_ts(e["occurred_at"])
            rows.append([
                t.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S"), A.to_db_iso(t), names.get(e["session_id"], ""),
                e["session_id"], e["event_type"], e["occupancy_after"], e["live_count"],
            ])
        return to_csv(
            [f"time_local ({tz})", "time_utc", "session_name", "session_id", "event", "occupancy_after", "people_in_frame"],
            rows,
        )

    def sessions_csv(self, tz_name: str | None) -> str:
        tz = A.get_zone(tz_name)
        rows = []
        offset = 0
        while True:
            page = self.store.list_sessions(500, offset)
            for s in page:
                fmt = lambda v: A.parse_ts(v).astimezone(tz).strftime("%Y-%m-%d %H:%M:%S") if v else ""  # noqa: E731
                rows.append([
                    s["name"], fmt(s["started_at"]), fmt(s.get("ended_at")), s["status"], s["total_entries"],
                    s["total_exits"], s["total_entries"] - s["total_exits"], s["max_occupancy"],
                    s["peak_live_count"], s.get("frames_processed", 0), s.get("detector") or "", s["id"],
                ])
            if len(page) < 500:
                break
            offset += 500
        return to_csv(
            ["name", f"started ({tz})", f"ended ({tz})", "status", "entries", "exits", "occupancy_at_end",
             "max_occupancy", "peak_people_in_frame", "frames_processed", "detector", "session_id"],
            rows,
        )

    def daily_csv(self, tz_name: str | None, days: int) -> str:
        data = self.daily(tz_name, days)
        return to_csv(
            ["date", "entries", "exits", "net", "traffic", "max_occupancy", "sessions"],
            [[d["date"], d["entries"], d["exits"], d["net"], d["traffic"],
              "" if d["max_occupancy"] is None else d["max_occupancy"], d["sessions"]] for d in data["days"]],
        )


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))
