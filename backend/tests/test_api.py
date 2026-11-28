"""REST API through the real FastAPI app: access control, cross-site protection, validation,
analytics on known data and CSV export."""
from __future__ import annotations

import csv
import io
import uuid

import pytest

from backend.app.config import ConfigError, load_settings
from backend.tests.conftest import ORIGIN, TOKEN

PROTECTED = [
    "/api/live", "/api/settings", "/api/sessions", "/api/events/recent",
    "/api/analytics/summary", "/api/analytics/hourly", "/api/analytics/daily",
    "/api/analytics/weekly", "/api/analytics/peak-hours", "/api/export/sessions.csv",
]


# ----------------------------------------------------------------------------- auth
def test_open_mode_without_access_token(make_client):
    client, _ = make_client()
    assert client.get("/api/auth/status").json() == {"auth_enabled": False, "authenticated": True}
    for path in PROTECTED:
        assert client.get(path).status_code == 200, path


def test_protected_endpoints_need_login(make_client):
    client, _ = make_client(access_token=TOKEN)
    for path in PROTECTED:
        r = client.get(path)
        assert r.status_code == 401, path
    # public endpoints still work, without internal details
    health = client.get("/api/health").json()
    assert "details" not in health and health["detector"] == "failed"
    assert client.get("/api/config").json()["auth_enabled"] is True


def test_login_sets_httponly_cookie_and_logout_clears_it(make_client):
    client, _ = make_client(access_token=TOKEN)
    r = client.post("/api/auth/login", json={"token": TOKEN}, headers={"Origin": ORIGIN})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "sv_session=" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    assert TOKEN not in r.headers["set-cookie"]          # the token itself never goes back
    assert client.get("/api/sessions").status_code == 200
    assert "details" in client.get("/api/health").json()

    assert client.post("/api/auth/logout", headers={"Origin": ORIGIN}).status_code == 200
    assert client.get("/api/sessions").status_code == 401


def test_forged_or_expired_cookie_rejected(make_client):
    client, _ = make_client(access_token=TOKEN)
    client.cookies.set("sv_session", "v1.9999999999.nonce.forgedsignature")
    assert client.get("/api/sessions").status_code == 401
    client.cookies.set("sv_session", "garbage")
    assert client.get("/api/sessions").status_code == 401


def test_login_rate_limited_after_repeated_failures(make_client):
    client, _ = make_client(access_token=TOKEN)
    for _ in range(5):
        assert client.post("/api/auth/login", json={"token": "wrong"}, headers={"Origin": ORIGIN}).status_code == 401
    r = client.post("/api/auth/login", json={"token": TOKEN}, headers={"Origin": ORIGIN})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_login_input_validated(make_client):
    client, _ = make_client(access_token=TOKEN)
    assert client.post("/api/auth/login", json={}, headers={"Origin": ORIGIN}).status_code == 422
    assert client.post("/api/auth/login", json={"token": "x" * 600}, headers={"Origin": ORIGIN}).status_code == 422
    assert client.post("/api/auth/login", json={"token": TOKEN, "admin": True}, headers={"Origin": ORIGIN}).status_code == 422


# ----------------------------------------------------------------------------- cross-site / hosts / headers
def test_cross_site_writes_blocked(make_client):
    client, _ = make_client()
    body = {"confidence": 0.5, "low_confidence": 0.2}
    assert client.put("/api/settings", json=body, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.put("/api/settings", json=body, headers={"Referer": "https://evil.example/page"}).status_code == 403
    assert client.put("/api/settings", json=body, headers={"Origin": "null"}).status_code == 403
    assert client.put("/api/settings", json=body, headers={"Origin": ORIGIN}).status_code == 200


def test_cors_only_for_configured_origins(make_client):
    client, _ = make_client(allowed_origins=("https://counter.example.com",))
    pre = {"Access-Control-Request-Method": "PUT", "Access-Control-Request-Headers": "content-type"}
    ok = client.options("/api/settings", headers={"Origin": "https://counter.example.com", **pre})
    assert ok.headers.get("access-control-allow-origin") == "https://counter.example.com"
    bad = client.options("/api/settings", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in bad.headers


def test_unknown_host_header_rejected(make_client):
    client, _ = make_client()
    assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 400


def test_security_headers(make_client):
    client, _ = make_client()
    r = client.get("/api/health")
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "camera=(self)" in r.headers["permissions-policy"]
    assert r.headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in r.headers           # dev: plain http
    assert client.get("/api/docs").status_code == 200              # docs in development only


def test_production_hides_docs_and_sends_hsts(make_client):
    client, _ = make_client(environment="production", access_token=TOKEN)
    assert client.get("/api/docs").status_code == 401 or client.get("/api/docs").status_code == 404
    assert "strict-transport-security" in client.get("/api/health").headers


def test_production_config_requires_secrets(monkeypatch):
    for k in ("ACCESS_TOKEN", "SECRET_KEY", "ALLOWED_ORIGINS", "STORAGE_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(ConfigError, match="ACCESS_TOKEN"):
        load_settings()
    monkeypatch.setenv("ACCESS_TOKEN", "a" * 20)
    with pytest.raises(ConfigError, match="SECRET_KEY"):
        load_settings()
    monkeypatch.setenv("SECRET_KEY", "b" * 40)
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    with pytest.raises(ConfigError, match="explicit origins"):
        load_settings()
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://counter.example.com")
    monkeypatch.setenv("STORAGE_BACKEND", "supabase")
    with pytest.raises(ConfigError, match="SUPABASE_URL"):
        load_settings()


# ----------------------------------------------------------------------------- settings
def test_settings_validation_and_persistence(make_client, make_settings):
    client, state = make_client()
    h = {"Origin": ORIGIN}
    assert client.put("/api/settings", json={"confidence": 0.3, "low_confidence": 0.5}, headers=h).status_code == 422
    assert client.put("/api/settings", json={"confidence": 2}, headers=h).status_code == 422
    assert client.put("/api/settings", json={"unknown": 1}, headers=h).status_code == 422
    assert client.put("/api/settings", json={"line": {"x1": .5, "y1": .5, "x2": .5, "y2": .5}}, headers=h).status_code == 422

    new = {"confidence": 0.55, "low_confidence": 0.2, "max_capacity": 12, "invert_direction": True,
           "line": {"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0}}
    r = client.put("/api/settings", json=new, headers=h)
    assert r.status_code == 200 and r.json()["max_capacity"] == 12
    assert client.get("/api/settings").json()["line"]["x1"] == 0.5
    assert state.defaults.max_capacity == 12          # new live sessions pick this up
    # stored in the database, not just memory
    assert state.store.get_settings()["confidence"] == 0.55


# ----------------------------------------------------------------------------- analytics on known data
def _seed(store):
    """One session on 2026-11-20 (Asia/Kolkata = UTC+5:30) with crossings at known times."""
    sid = str(uuid.uuid4())
    store.insert_session({
        "id": sid, "name": "=HYPERLINK(\"http://evil\")", "started_at": "2026-11-20T03:00:00.000+00:00",
        "status": "running", "detector": "yolo-onnx:yolo26n.onnx", "settings": {}, "total_entries": 0,
        "total_exits": 0, "max_occupancy": 0, "peak_live_count": 0,
    })
    # 03:10Z = 08:40 IST, 03:40Z = 09:10 IST, 04:05Z = 09:35 IST, 12:00Z = 17:30 IST
    times = [("entry", "03:10", 1), ("entry", "03:40", 2), ("exit", "04:05", 1), ("entry", "12:00", 2),
             ("capacity_exceeded", "12:00", 2)]
    store.insert_events([
        {"id": str(uuid.uuid4()), "session_id": sid, "event_type": kind,
         "occurred_at": f"2026-11-20T{t}:00.000+00:00", "occupancy_after": occ, "live_count": 1}
        for kind, t, occ in times
    ])
    store.update_session(sid, {"total_entries": 3, "total_exits": 1, "max_occupancy": 2, "peak_live_count": 2,
                               "status": "completed", "ended_at": "2026-11-20T13:00:00.000+00:00"})
    return sid


def test_hourly_daily_peak_in_viewer_timezone(make_client):
    client, state = make_client()
    _seed(state.store)
    tz = "Asia/Kolkata"
    hours = client.get("/api/analytics/hourly", params={"tz": tz, "date": "2026-11-20"}).json()["hours"]
    by_hour = {h["hour"]: (h["entries"], h["exits"]) for h in hours if h["traffic"]}
    assert by_hour == {8: (1, 0), 9: (1, 1), 17: (1, 0)}

    # the same data in UTC lands in different hours
    utc = client.get("/api/analytics/hourly", params={"tz": "UTC", "date": "2026-11-20"}).json()["hours"]
    assert {h["hour"] for h in utc if h["traffic"]} == {3, 4, 12}

    peak = client.get("/api/analytics/peak-hours", params={"tz": tz, "days": 366}).json()
    assert peak["peak"][0]["hour"] == 9 and peak["peak"][0]["traffic"] == 2

    r = client.get("/api/analytics/hourly", params={"tz": "Not/AZone", "date": "2026-11-20"})
    assert r.status_code == 400
    assert client.get("/api/analytics/hourly", params={"tz": "../../etc/passwd"}).status_code == 400
    assert client.get("/api/analytics/daily", params={"days": 1000}).status_code == 422


def test_sessions_history_detail_delete(make_client):
    client, state = make_client()
    sid = _seed(state.store)
    rows = client.get("/api/sessions").json()["sessions"]
    assert [r["id"] for r in rows] == [sid] and rows[0]["total_entries"] == 3 and rows[0]["live"] is False
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["status"] == "completed" and "snapshots" in detail
    assert client.get("/api/sessions/not-a-uuid").status_code == 422
    assert client.get(f"/api/sessions/{uuid.uuid4()}").status_code == 404

    recent = client.get("/api/events/recent", params={"limit": 2}).json()["events"]
    assert len(recent) == 2 and recent[0]["session_name"].startswith("=HYPERLINK")

    # a session that is running live can't be deleted
    state.registry.sessions[sid] = object()
    assert client.delete(f"/api/sessions/{sid}", headers={"Origin": ORIGIN}).status_code == 409
    state.registry.sessions.clear()
    assert client.delete(f"/api/sessions/{sid}", headers={"Origin": ORIGIN}).status_code == 200
    assert client.get("/api/sessions").json()["sessions"] == []
    assert state.store.get_events("2026-01-01", "2027-01-01") == []      # cascaded


def test_csv_exports(make_client):
    client, state = make_client()
    _seed(state.store)
    r = client.get("/api/export/events.csv", params={"start": "2026-11-20", "end": "2026-11-20", "tz": "Asia/Kolkata"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text.lstrip("﻿"))))
    assert rows[0][0] == "time_local (Asia/Kolkata)"
    assert [row[4] for row in rows[1:]] == ["entry", "entry", "exit", "entry", "capacity_exceeded"]
    assert rows[1][0] == "2026-11-20 08:40:00"
    assert rows[1][2].startswith("'=HYPERLINK")           # formula injection neutralised

    sessions = list(csv.reader(io.StringIO(client.get("/api/export/sessions.csv").text.lstrip("﻿"))))
    assert sessions[1][0].startswith("'=") and sessions[1][4:7] == ["3", "1", "2"]

    daily = client.get("/api/export/daily.csv", params={"days": 366, "tz": "Asia/Kolkata"}).text
    assert "2026-11-20,3,1,2,4,2,1" in daily

    bad = client.get("/api/export/events.csv", params={"start": "2026-11-20", "end": "2026-11-01"})
    assert bad.status_code == 400
