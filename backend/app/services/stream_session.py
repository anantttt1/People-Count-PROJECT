"""One live counting session: detector → tracker → line counter → statistics.

``process_frame`` is synchronous and CPU-heavy; the WebSocket handler runs it in a worker
thread. Nothing here keeps the frame after the call returns.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from ..schemas import DetectionSettings
from ..vision.density import density
from ..vision.detector import Detector
from ..vision.heatmap import Heatmap
from ..vision.line_counter import CountingLine, LineCounter, anchor_point
from ..vision.tracker import ByteTracker

HEATMAP_PUSH_SECONDS = 2.0


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


class StreamSession:
    def __init__(
        self,
        detector: Detector,
        settings: DetectionSettings,
        name: str,
        snapshot_interval: float = 30.0,
        clock: Callable[[], float] = time.time,
        detector_name: str = "",
    ) -> None:
        self.id = str(uuid.uuid4())
        self.name = name
        self.detector = detector
        self.detector_name = detector_name or getattr(detector, "name", "unknown")
        self.clock = clock
        self.snapshot_interval = snapshot_interval
        self.started_at = clock()
        self.settings = settings
        self.tracker = ByteTracker(
            high_thresh=settings.confidence,
            low_thresh=settings.low_confidence,
            max_lost_seconds=settings.lost_track_seconds,
        )
        self.counter = LineCounter(self._line(settings), settings.invert_direction, settings.anchor)
        self.heatmap = Heatmap()

        self.entries = 0
        self.exits = 0
        self.max_occupancy = 0
        self.peak_live = 0
        self.live = 0
        self.frames = 0
        self.capacity_exceeded = False
        self.paused = False
        self.fps = 0.0
        self.inference_ms = 0.0
        self._last_frame_at: float | None = None
        self._last_heatmap_push = 0.0
        self._snap_started = self.started_at
        self._snap_live_sum = 0
        self._snap_live_max = 0
        self._snap_frames = 0

    # -- configuration ----------------------------------------------------------
    @staticmethod
    def _line(s: DetectionSettings) -> CountingLine:
        return CountingLine(s.line.x1, s.line.y1, s.line.x2, s.line.y2)

    def apply_settings(self, settings: DetectionSettings) -> None:
        self.tracker.configure(
            high_thresh=settings.confidence,
            low_thresh=settings.low_confidence,
            max_lost_seconds=settings.lost_track_seconds,
        )
        self.counter.reconfigure(self._line(settings), settings.invert_direction, settings.anchor)
        self.settings = settings

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        """After a pause the gap between frames is not a slow frame: restart the FPS clock."""
        self.paused = False
        self._last_frame_at = None

    @property
    def occupancy(self) -> int:
        # As specified: Current Occupancy = Total Entries - Total Exits
        return self.entries - self.exits

    # -- per frame --------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> tuple[dict, list[dict]]:
        """Run the full pipeline on one frame. Returns (message for the client, rows to persist)."""
        s = self.settings
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        dets = self.detector.detect(frame_bgr, conf=s.low_confidence, iou=s.iou)
        if len(dets) and s.min_box_height > 0:
            dets = dets[(dets[:, 3] - dets[:, 1]) >= s.min_box_height * h]
        now = self.clock()
        tracks = self.tracker.update(dets, now)
        self.inference_ms = (time.perf_counter() - t0) * 1000.0

        if self._last_frame_at is not None:
            dt = now - self._last_frame_at
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else 0.8 * self.fps + 0.2 * inst
        self._last_frame_at = now
        self.frames += 1

        norm_tracks = []
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            box = (
                float(np.clip(x1 / w, 0, 1)), float(np.clip(y1 / h, 0, 1)),
                float(np.clip(x2 / w, 0, 1)), float(np.clip(y2 / h, 0, 1)),
            )
            norm_tracks.append((t.track_id, box, t.score))

        crossings = self.counter.update([(tid, box) for tid, box, _ in norm_tracks], w, h, now)
        events: list[dict] = []
        client_events: list[dict] = []
        for c in crossings:
            if c.direction == "entry":
                self.entries += 1
            else:
                self.exits += 1
            self.max_occupancy = max(self.max_occupancy, self.occupancy)
            events.append(self._event_row(c.direction, now))
            client_events.append({"type": c.direction, "track_id": c.track_id, "ts": now, "occupancy": self.occupancy})

        cap = s.max_capacity
        over = cap is not None and self.occupancy > cap
        if over != self.capacity_exceeded:
            self.capacity_exceeded = over
            kind = "capacity_exceeded" if over else "capacity_cleared"
            events.append(self._event_row(kind, now))
            client_events.append({"type": kind, "ts": now, "occupancy": self.occupancy, "capacity": cap})

        self.live = len(norm_tracks)
        self.peak_live = max(self.peak_live, self.live)
        self.heatmap.add([anchor_point(box, s.anchor) for _, box, _ in norm_tracks])
        self._snap_live_sum += self.live
        self._snap_live_max = max(self._snap_live_max, self.live)
        self._snap_frames += 1

        message = {
            "type": "result",
            "frame": self.frames,
            "ts": now,
            "width": w,
            "height": h,
            "tracks": [
                {"id": tid, "box": [round(v, 4) for v in box], "score": round(score, 3)}
                for tid, box, score in norm_tracks
            ],
            "counts": self.counts(),
            "events": client_events,
            "density": density([box for _, box, _ in norm_tracks]),
            "capacity": {"max": cap, "exceeded": self.capacity_exceeded},
            "perf": {"inference_ms": round(self.inference_ms, 1), "fps": round(self.fps, 1), "detector": self.detector_name},
        }
        if now - self._last_heatmap_push >= HEATMAP_PUSH_SECONDS:
            self._last_heatmap_push = now
            message["heatmap"] = self.heatmap.to_dict()
        return message, events

    def counts(self) -> dict:
        return {
            "live": self.live,
            "entries": self.entries,
            "exits": self.exits,
            "occupancy": self.occupancy,
            "max_occupancy": self.max_occupancy,
            "peak_live": self.peak_live,
        }

    # -- persistence rows -------------------------------------------------------
    def _event_row(self, kind: str, now: float) -> dict:
        return {
            "id": str(uuid.uuid4()),          # client-generated key makes retried inserts idempotent
            "session_id": self.id,
            "event_type": kind,
            "occurred_at": iso(now),
            "occupancy_after": self.occupancy,
            "live_count": self.live,
        }

    def session_row(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "started_at": iso(self.started_at),
            "status": "running",
            "detector": self.detector_name,
            "settings": self.settings.model_dump(),
            "total_entries": 0,
            "total_exits": 0,
            "max_occupancy": 0,
            "peak_live_count": 0,
        }

    def summary_fields(self, ended: bool = False) -> dict:
        fields = {
            "total_entries": self.entries,
            "total_exits": self.exits,
            "max_occupancy": self.max_occupancy,
            "peak_live_count": self.peak_live,
            "frames_processed": self.frames,
            "settings": self.settings.model_dump(),
            "heatmap": self.heatmap.to_dict(),
        }
        if ended:
            fields["ended_at"] = iso(self.clock())
            fields["status"] = "completed"
        return fields

    def snapshot_if_due(self, force: bool = False) -> dict | None:
        now = self.clock()
        if self._snap_frames == 0 or (not force and now - self._snap_started < self.snapshot_interval):
            return None
        row = {
            "id": str(uuid.uuid4()),
            "session_id": self.id,
            "captured_at": iso(now),
            "avg_live_count": round(self._snap_live_sum / self._snap_frames, 2),
            "max_live_count": self._snap_live_max,
            "occupancy": self.occupancy,
            "fps": round(self.fps, 1),
        }
        self._snap_started = now
        self._snap_live_sum = self._snap_live_max = self._snap_frames = 0
        return row
