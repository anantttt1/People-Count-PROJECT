"""Analytics computed from stored rows only (events, sessions). No sampling, no estimates.

All grouping happens in the viewer's time zone (IANA name sent by the browser), so "today",
"9 AM" and "this week" mean what the user expects, including across DST changes.
Traffic = entries + exits (line crossings).
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_FRACTION = re.compile(r"\.(\d+)")


class AnalyticsError(ValueError):
    pass


def get_zone(name: str | None) -> ZoneInfo:
    if not name:
        return ZoneInfo("UTC")
    if len(name) > 64 or not re.fullmatch(r"[A-Za-z0-9_+\-/]+", name):
        raise AnalyticsError("invalid time zone")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise AnalyticsError(f"unknown time zone {name!r}") from exc


def parse_ts(value: str) -> datetime:
    """Parse timestamps from SQLite (ms precision) and PostgREST (variable precision, Z or +00:00)."""
    s = value.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = _FRACTION.sub(lambda m: "." + (m.group(1) + "000000")[:6], s, count=1)
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def to_db_iso(dt: datetime) -> str:
    """Same format the backend writes (UTC, millisecond precision) so text comparison in
    SQLite orders correctly."""
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def local_range(start_day: date, days: int, tz: ZoneInfo) -> tuple[str, str]:
    """UTC ISO bounds [start of start_day, start of start_day + days) in local time."""
    start = datetime.combine(start_day, time.min, tzinfo=tz)
    end = datetime.combine(start_day + timedelta(days=days), time.min, tzinfo=tz)
    return to_db_iso(start), to_db_iso(end)


def today_in(tz: ZoneInfo, now: datetime | None = None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(tz).date()


def _local(ev: dict, tz: ZoneInfo) -> datetime:
    return parse_ts(ev["occurred_at"]).astimezone(tz)


def _crossings(events: list[dict]) -> list[dict]:
    return [e for e in events if e["event_type"] in ("entry", "exit")]


# ----------------------------------------------------------------------------- views
def hourly(events: list[dict], day: date, tz: ZoneInfo) -> dict:
    hours = [{"hour": h, "entries": 0, "exits": 0, "traffic": 0, "max_occupancy": None} for h in range(24)]
    for ev in _crossings(events):
        t = _local(ev, tz)
        if t.date() != day:
            continue
        b = hours[t.hour]
        b["entries" if ev["event_type"] == "entry" else "exits"] += 1
        b["traffic"] += 1
        occ = ev.get("occupancy_after")
        if occ is not None:
            b["max_occupancy"] = occ if b["max_occupancy"] is None else max(b["max_occupancy"], occ)
    return {"date": day.isoformat(), "timezone": str(tz), "hours": hours}


def daily(events: list[dict], sessions: list[dict], start_day: date, days: int, tz: ZoneInfo) -> list[dict]:
    rows = {
        (start_day + timedelta(days=i)): {
            "date": (start_day + timedelta(days=i)).isoformat(),
            "entries": 0, "exits": 0, "traffic": 0, "net": 0, "max_occupancy": None, "sessions": 0,
        }
        for i in range(days)
    }
    for ev in _crossings(events):
        t = _local(ev, tz)
        row = rows.get(t.date())
        if row is None:
            continue
        row["entries" if ev["event_type"] == "entry" else "exits"] += 1
        row["traffic"] += 1
        occ = ev.get("occupancy_after")
        if occ is not None:
            row["max_occupancy"] = occ if row["max_occupancy"] is None else max(row["max_occupancy"], occ)
    for s in sessions:
        d = parse_ts(s["started_at"]).astimezone(tz).date()
        if d in rows:
            rows[d]["sessions"] += 1
    for row in rows.values():
        row["net"] = row["entries"] - row["exits"]
    return [rows[k] for k in sorted(rows)]


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def weekly(events: list[dict], first_monday: date, weeks: int, tz: ZoneInfo) -> list[dict]:
    rows = {}
    for i in range(weeks):
        ws = first_monday + timedelta(weeks=i)
        iso = ws.isocalendar()
        rows[ws] = {"week_start": ws.isoformat(), "label": f"{iso[0]}-W{iso[1]:02d}",
                    "entries": 0, "exits": 0, "traffic": 0}
    for ev in _crossings(events):
        row = rows.get(week_start(_local(ev, tz).date()))
        if row is None:
            continue
        row["entries" if ev["event_type"] == "entry" else "exits"] += 1
        row["traffic"] += 1
    return [rows[k] for k in sorted(rows)]


def peak_hours(events: list[dict], start_day: date, days: int, tz: ZoneInfo, top: int = 3) -> dict:
    """Average crossings per hour of day over the range, and the busiest hours."""
    totals = [0] * 24
    entries = [0] * 24
    end_day = start_day + timedelta(days=days)
    active_days: set[date] = set()
    for ev in _crossings(events):
        t = _local(ev, tz)
        if not start_day <= t.date() < end_day:
            continue
        totals[t.hour] += 1
        active_days.add(t.date())
        if ev["event_type"] == "entry":
            entries[t.hour] += 1
    hours = [
        {"hour": h, "traffic": totals[h], "entries": entries[h], "avg_per_day": round(totals[h] / days, 2)}
        for h in range(24)
    ]
    ranked = sorted((h for h in hours if h["traffic"] > 0), key=lambda h: (-h["traffic"], h["hour"]))[:top]
    return {"days": days, "active_days": len(active_days), "hours": hours, "peak": ranked}


def summary(today_events: list[dict], today: date, tz: ZoneInfo, live_sessions: int) -> dict:
    h = hourly(today_events, today, tz)["hours"]
    crossings = _crossings(today_events)
    entries = sum(1 for e in crossings if e["event_type"] == "entry")
    exits = len(crossings) - entries
    occs = [e["occupancy_after"] for e in crossings if e.get("occupancy_after") is not None]
    busiest = max(h, key=lambda b: (b["traffic"], -b["hour"]))
    alerts = sum(1 for e in today_events if e["event_type"] == "capacity_exceeded")
    return {
        "date": today.isoformat(),
        "timezone": str(tz),
        "entries": entries,
        "exits": exits,
        "traffic": len(crossings),
        "max_occupancy": max(occs) if occs else None,
        "busiest_hour": busiest["hour"] if busiest["traffic"] else None,
        "busiest_hour_traffic": busiest["traffic"],
        "capacity_alerts": alerts,
        "live_sessions": live_sessions,
    }
