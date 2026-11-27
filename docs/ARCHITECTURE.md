# Architecture

Smart Vision AI is a browser + Python application. The browser owns the camera; the
Python backend owns the AI. Only numbers leave the AI pipeline.

```
 Browser (dashboard.html)                         Python backend (FastAPI, uvicorn)
 ───────────────────────────                      ─────────────────────────────────────────────
 getUserMedia (explicit permission)
        │ <video> preview (local)
        ▼
 canvas → JPEG (≤640 px wide)  ──── binary ────▶  /ws/stream  (origin + auth checked)
                                   WebSocket          │
                                                      ▼
                                              StreamSession (one per connection)
                                                ├─ decode JPEG (OpenCV)
                                                ├─ Detector: YOLO26 person class only
                                                │     (fallback: YOLO11 → OpenCV HOG)
                                                ├─ ByteTracker: Kalman + 2-stage IoU matching
                                                ├─ LineCounter: side-of-line + hysteresis
                                                ├─ Heatmap grid, crowd density, capacity
                                                └─ frame is discarded (never stored)
 overlay canvas (boxes, IDs,  ◀─── JSON ───────────────┘
 line, heatmap), live stats,        result per frame        │ events / snapshots / session totals
 activity feed, alerts                                      ▼
                                              PersistenceQueue (async, batched, retry/backoff)
                                                      │
 Analytics view (Chart.js) ◀── REST /api/* ──  Store ─┴─ SupabaseStore (PostgREST, service key, RLS)
 CSV export, sessions, settings                      └─ SQLiteStore (local default / dev)
```

## Why frames go to the backend

The spec asks for Python + OpenCV + Ultralytics YOLO. Those run server-side, so the browser
sends compressed frames over a WebSocket. To keep this private:

* frames are processed in memory and dropped immediately - no recording, no disk writes;
* run the backend on the same machine or LAN (the default binds to `127.0.0.1`);
* only counts, timestamps and coarse heatmap cells are stored.

Flow control is stop-and-wait: the browser sends the next frame only after the previous
result arrives, so a slow CPU lowers the detection FPS instead of building a backlog.

## Modules

| Path | Responsibility |
|------|----------------|
| `backend/app/config.py` | Environment settings, validation of unsafe configs |
| `backend/app/security.py` | Session cookie, login rate limit, origin checks, security headers |
| `backend/app/vision/detector.py` | YOLO / HOG detectors behind one interface, loader with fallbacks |
| `backend/app/vision/kalman.py` | Constant-velocity Kalman filter (box centre, aspect, height) |
| `backend/app/vision/tracker.py` | ByteTrack multi-object tracker |
| `backend/app/vision/line_counter.py` | Entry/exit counting on a virtual line |
| `backend/app/vision/heatmap.py`, `density.py` | Occupancy heatmap, crowd coverage |
| `backend/app/services/stream_session.py` | Per-connection pipeline and live statistics |
| `backend/app/services/ws_handler.py` | WebSocket protocol (framework-agnostic) |
| `backend/app/services/persistence.py` | Async write queue with retries |
| `backend/app/services/analytics.py` | Hourly / daily / weekly / peak-hour aggregation |
| `backend/app/services/api_service.py` | Query logic behind every REST endpoint |
| `backend/app/storage/` | `Store` interface, SQLite and Supabase implementations |
| `backend/app/api/` | FastAPI routers (thin: validation + calls into services) |
| `frontend/` | Landing page, dashboard, privacy notice (plain HTML/CSS/ES modules) |
| `database/schema.sql` | Supabase tables, indexes, RLS |

## WebSocket protocol (`/ws/stream`)

Client → server (text JSON unless noted):

| Message | Meaning |
|---------|---------|
| `{"type":"start","name":"Front door","settings":{...}}` | Create a counting session |
| *binary JPEG* | One camera frame |
| `{"type":"config","settings":{...}}` | Change line / thresholds / capacity live |
| `{"type":"reset"}` | Close the current session and start a fresh one (after UI confirmation) |
| `{"type":"stop"}` | End the session |
| `{"type":"ping"}` | Keep-alive |

Server → client: `ready`, `session`, `result` (per frame), `error`, `pong`, `stopped`.
A `result` carries normalised boxes with track IDs, counts, events, density, capacity state,
timing and (every ~2 s) the heatmap grid.

## Data model

* `sessions` - one row per counting run: name, start/end, totals, max occupancy, peak live
  count, settings used, heatmap grid.
* `events` - `entry`, `exit`, `capacity_exceeded`, `capacity_cleared` with timestamp and
  occupancy after the event. No images, no track history, no identities.
* `occupancy_snapshots` - every 30 s per running session: average/max people in frame,
  occupancy, detection FPS.
* `app_settings` - admin defaults (one JSON row).

Analytics are computed from these rows, so every number on the dashboard is traceable to
stored events.
