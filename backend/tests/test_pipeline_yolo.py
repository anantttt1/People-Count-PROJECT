"""End-to-end pipeline with real YOLO26 (ONNX): detection -> ByteTrack -> line counting,
on a moving scene built from real people (see scenes.py)."""
from __future__ import annotations

import pytest

from backend.app.schemas import DetectionSettings
from backend.app.services.stream_session import StreamSession
from backend.app.vision.detector import OnnxYoloDetector
from backend.tests import assets
from backend.tests.scenes import background, doorway_movers, render

MODEL = assets.model_path("yolo26n.onnx")
BUS = assets.image("bus.jpg")
pytestmark = pytest.mark.skipif(MODEL is None or BUS is None, reason="yolo26n.onnx / bus.jpg not available")


class Clock:
    def __init__(self, step):
        self.t, self.step = 1_700_000_000.0, step

    def __call__(self):
        return self.t


@pytest.fixture(scope="module")
def detector():
    return OnnxYoloDetector(MODEL)


@pytest.mark.parametrize("frame_step", [1, 3])  # every frame (30 fps) and every 3rd (≈10 fps)
def test_yolo_pipeline_counts_real_people(detector, frame_step):
    movers, n = doorway_movers(BUS)
    bg = background()
    clock = Clock(step=frame_step / 30.0)
    session = StreamSession(detector, DetectionSettings(max_capacity=None), "yolo-e2e", clock=clock)
    ids_per_person: list[set[int]] = []
    seen_ids: set[int] = set()
    rows = []
    for f in range(0, n, frame_step):
        msg, events = session.process_frame(render(movers, f, bg))
        rows.extend(events)
        seen_ids.update(t["id"] for t in msg["tracks"])
        assert msg["counts"]["live"] <= 1  # never more than one person on screen at a time
        clock.t += clock.step

    assert (session.entries, session.exits, session.occupancy) == (2, 1, 1), session.counts()
    assert session.max_occupancy == 1
    assert [r["event_type"] for r in rows] == ["entry", "exit", "entry"]
    # three people -> three track IDs: no ID switches, so nobody can be double counted
    assert len(seen_ids) == 3, seen_ids


def test_inverted_direction_swaps_entries_and_exits(detector):
    movers, n = doorway_movers(BUS)
    bg = background()
    clock = Clock(step=0.1)
    session = StreamSession(detector, DetectionSettings(max_capacity=None, invert_direction=True), "inv", clock=clock)
    for f in range(0, n, 3):
        session.process_frame(render(movers, f, bg))
        clock.t += clock.step
    assert (session.entries, session.exits, session.occupancy) == (1, 2, -1)


def test_capacity_alert_fires_when_occupancy_exceeds_limit(detector):
    movers, n = doorway_movers(BUS)
    two_entries = [movers[0], movers[2]]          # A and C walk in, nobody leaves
    bg = background()
    clock = Clock(step=0.1)
    session = StreamSession(detector, DetectionSettings(max_capacity=1), "cap", clock=clock)
    kinds, client_kinds = [], []
    for f in range(0, n, 3):
        msg, events = session.process_frame(render(two_entries, f, bg))
        kinds += [e["event_type"] for e in events]
        client_kinds += [e["type"] for e in msg["events"]]
        clock.t += clock.step
    assert kinds == ["entry", "entry", "capacity_exceeded"]
    assert client_kinds == kinds
    assert session.capacity_exceeded and session.occupancy == 2
