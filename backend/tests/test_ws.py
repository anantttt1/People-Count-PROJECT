"""Live WebSocket through the real FastAPI app.

The main test streams a moving scene of real people (person crops from a real photo) as JPEG
frames - exactly what the browser sends - and the real YOLO26 model + ByteTrack + line counter
must produce the right counts, which must then show up in the database and analytics.
"""
from __future__ import annotations

import json

import cv2
import pytest
from starlette.websockets import WebSocketDisconnect

from backend.tests import assets
from backend.tests.conftest import ORIGIN, TOKEN, onnx_loader, wait_until
from backend.tests.scenes import background, doorway_movers, render

WS = "/ws/stream"
HDR = {"Origin": ORIGIN, "Host": "localhost:8000"}   # the test client defaults to Host: testserver
HAVE_YOLO = assets.model_path("yolo26n.onnx") is not None and assets.image("bus.jpg") is not None
needs_yolo = pytest.mark.skipif(not HAVE_YOLO, reason="yolo26n.onnx / bus.jpg not available")


def jpeg(img, width=640) -> bytes:
    """Same as the browser: scale to <= 640 px wide, JPEG quality ~0.7."""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    return buf.tobytes()


def recv(ws, kind=None):
    while True:
        msg = json.loads(ws.receive_text())
        if kind is None or msg["type"] == kind or msg["type"] == "error":
            return msg


def start(ws, **settings):
    ready = recv(ws, "ready")
    ws.send_text(json.dumps({"type": "start", "name": "Front door", "settings": settings or None}))
    return ready, recv(ws, "session")


# ----------------------------------------------------------------------------- handshake security
def test_ws_rejects_foreign_origin(make_client):
    client, _ = make_client()
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(WS, headers={"Origin": "https://evil.example", "Host": "localhost:8000"}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


def test_ws_requires_login_when_auth_enabled(make_client):
    client, _ = make_client(access_token=TOKEN)
    with client.websocket_connect(WS, headers=HDR) as ws:
        msg = json.loads(ws.receive_text())
        assert msg["code"] == "auth_required"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401
    # after login the same handshake works
    client.post("/api/auth/login", json={"token": TOKEN}, headers={"Origin": ORIGIN})
    cookie = client.cookies.get("sv_session")
    assert cookie
    # the test client doesn't attach its cookie jar to WebSocket handshakes; a browser does
    with client.websocket_connect(WS, headers={**HDR, "Cookie": f"sv_session={cookie}"}) as ws:
        assert json.loads(ws.receive_text())["type"] == "ready"


# ----------------------------------------------------------------------------- protocol validation
def test_ws_protocol_errors_are_reported_not_fatal(make_client):
    client, _ = make_client(loader=onnx_loader) if HAVE_YOLO else make_client()
    with client.websocket_connect(WS, headers=HDR) as ws:
        recv(ws, "ready")
        ws.send_bytes(b"\xff\xd8\xff")
        assert recv(ws)["code"] == "no_session"
        ws.send_text("{not json")
        assert recv(ws)["code"] == "bad_json"
        ws.send_text(json.dumps({"type": "launch_missiles"}))
        assert recv(ws)["code"] == "invalid_message"
        ws.send_text(json.dumps({"type": "start", "settings": {"confidence": 5}}))
        assert recv(ws)["code"] == "invalid_message"
        ws.send_text(json.dumps({"type": "config", "settings": {}}))
        assert recv(ws)["code"] == "no_session"
        ws.send_text("x" * 20_000)
        assert recv(ws)["code"] == "message_too_large"
        ws.send_text(json.dumps({"type": "ping"}))
        assert recv(ws)["type"] == "pong"            # connection still healthy


def test_ws_start_fails_cleanly_without_detector(make_client):
    client, _ = make_client()            # loader raises -> detector failed
    with client.websocket_connect(WS, headers=HDR) as ws:
        ready = recv(ws, "ready")
        assert ready["detector"]["status"] == "failed"
        ws.send_text(json.dumps({"type": "start"}))
        err = recv(ws)
        assert err["code"] == "detector_failed" and "intentionally disabled" in err["message"]


@needs_yolo
def test_ws_rejects_bad_frames(make_client):
    client, _ = make_client(loader=onnx_loader, max_frame_dimension=640)
    with client.websocket_connect(WS, headers=HDR) as ws:
        start(ws)
        ws.send_bytes(b"GIF89a" + b"\0" * 100)
        assert recv(ws)["code"] == "bad_frame"
        big = cv2.imencode(".jpg", background())[1].tobytes()        # 960 px wide > 640 limit
        ws.send_bytes(big)
        assert "limit" in recv(ws)["message"]
        ws.send_bytes(jpeg(background()))                              # valid frame still works
        assert recv(ws)["type"] == "result"


@needs_yolo
def test_ws_session_limit(make_client):
    client, _ = make_client(loader=onnx_loader, max_concurrent_sessions=1)
    with client.websocket_connect(WS, headers=HDR) as a, client.websocket_connect(WS, headers=HDR) as b:
        start(a)
        recv(b, "ready")
        b.send_text(json.dumps({"type": "start"}))
        assert recv(b)["code"] == "too_many_sessions"


# ----------------------------------------------------------------------------- the real thing
@needs_yolo
def test_ws_real_yolo_counts_and_persists(make_client):
    client, state = make_client(loader=onnx_loader)
    bus = assets.image("bus.jpg")
    movers, n = doorway_movers(bus)
    bg = background()

    results, events, ids = [], [], set()
    with client.websocket_connect(WS, headers=HDR) as ws:
        ready, session = start(ws, max_capacity=None)
        assert ready["detector"]["model"] == "yolo26n.onnx"
        sid = session["session"]["id"]
        for f in range(0, n, 2):                     # ~15 fps of a 30 fps scene
            ws.send_bytes(jpeg(render(movers, f, bg)))
            msg = recv(ws, "result")
            assert msg["type"] == "result", msg
            results.append(msg)
            events += msg["events"]
            ids.update(t["id"] for t in msg["tracks"])
            for t in msg["tracks"]:
                x1, y1, x2, y2 = t["box"]
                assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and 0 < t["score"] <= 1

        last = results[-1]["counts"]
        assert (last["entries"], last["exits"], last["occupancy"]) == (2, 1, 1), last
        assert [e["type"] for e in events] == ["entry", "exit", "entry"]
        assert len(ids) == 3, ids                     # one ID per person -> no double counting
        assert max(r["counts"]["live"] for r in results) == 1
        assert results[-1]["perf"]["detector"].startswith("yolo-onnx")
        assert any("heatmap" in r for r in results)

        ws.send_text(json.dumps({"type": "stop"}))
        stopped = recv(ws, "stopped")
        assert stopped["session_id"] == sid and stopped["counts"]["entries"] == 2

    # everything reaches the database through the async queue
    assert wait_until(lambda: state.queue.pending == 0)
    row = client.get(f"/api/sessions/{sid}").json()
    assert row["status"] == "completed" and row["ended_at"]
    assert (row["total_entries"], row["total_exits"], row["max_occupancy"]) == (2, 1, 1)
    assert row["frames_processed"] == len(results)
    assert row["snapshots"] and row["heatmap"]["cols"] > 0
    summary = client.get("/api/analytics/summary").json()
    assert (summary["entries"], summary["exits"], summary["traffic"]) == (2, 1, 3)
    recent = client.get("/api/events/recent").json()["events"]
    assert [e["event_type"] for e in recent] == ["entry", "exit", "entry"]   # newest first


@needs_yolo
def test_ws_pause_resume_reset_and_live_config(make_client):
    client, state = make_client(loader=onnx_loader)
    bg = background()
    with client.websocket_connect(WS, headers=HDR) as ws:
        _, first = start(ws)
        ws.send_bytes(jpeg(bg))
        assert recv(ws, "result")["counts"]["live"] == 0

        ws.send_text(json.dumps({"type": "pause"}))
        assert recv(ws, "paused")["type"] == "paused"
        ws.send_bytes(jpeg(bg))
        assert recv(ws)["code"] == "paused"
        ws.send_text(json.dumps({"type": "resume"}))
        assert recv(ws, "resumed")["type"] == "resumed"
        ws.send_bytes(jpeg(bg))
        assert recv(ws, "result")["type"] == "result"

        new = {"confidence": 0.6, "low_confidence": 0.2, "max_capacity": 3,
               "line": {"x1": 0.5, "y1": 0.05, "x2": 0.5, "y2": 0.95}}
        ws.send_text(json.dumps({"type": "config", "settings": new}))
        applied = recv(ws, "config_applied")
        assert applied["settings"]["line"]["x1"] == 0.5
        assert state.registry.public()[0]["id"] == first["session"]["id"]

        ws.send_text(json.dumps({"type": "reset"}))
        second = recv(ws, "session")
        assert second["session"]["id"] != first["session"]["id"]
        assert second["session"]["settings"]["max_capacity"] == 3      # settings carried over
        assert second["counts"]["entries"] == 0

    assert wait_until(lambda: state.queue.pending == 0)
    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 2 and {s["status"] for s in sessions} == {"completed"}   # closed on disconnect
    assert state.registry.sessions == {}
