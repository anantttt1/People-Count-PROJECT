"""Full pipeline tests: real detector (HOG) on synthetic walking silhouettes, plus exact
logic tests with a scripted detector."""
import numpy as np

from backend.app.schemas import DetectionSettings
from backend.app.services.stream_session import StreamSession
from backend.app.vision.detector import HogDetector
from backend.tests.synthetic import doorway_scene, render


class FakeClock:
    def __init__(self, step=0.1):
        self.t = 1_700_000_000.0
        self.step = step

    def __call__(self):
        return self.t

    def tick(self):
        self.t += self.step


class ScriptedDetector:
    """Returns pre-baked boxes - used only to test counting/stat logic precisely."""
    name = "scripted"
    model = "test"

    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def detect(self, frame, conf, iou):
        out = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return np.array(out, dtype=np.float32).reshape(-1, 5)


def test_real_hog_pipeline_counts_entry_and_exit():
    walkers, n = doorway_scene()
    clock = FakeClock(step=0.1)
    s = StreamSession(HogDetector(), DetectionSettings(max_capacity=None), "test", clock=clock)
    ids = set()
    for f in range(n):
        msg, events = s.process_frame(render(walkers, f))
        ids.update(t["id"] for t in msg["tracks"])
        clock.tick()
    assert s.entries == 1, f"entries={s.entries}"
    assert s.exits == 1, f"exits={s.exits}"
    assert s.occupancy == 0
    assert s.max_occupancy == 1
    assert len(ids) == 2           # two different people, one ID each
    assert s.peak_live == 1


def test_scripted_counts_capacity_events_and_rows():
    frame = np.zeros((400, 640, 3), np.uint8)
    # three people walk down across y=0.55*400=220, one after another
    frames = []
    for person in range(3):
        x = 100 + person * 150
        for i in range(20):
            top = 20 + i * 15
            frames.append([[x, top, x + 50, top + 120, 0.9]])
        frames.append([])
        frames.extend([[]] * 25)   # gap long enough for the track to expire (2.6 s > 2 s)
    clock = FakeClock(step=0.1)
    s = StreamSession(ScriptedDetector(frames), DetectionSettings(max_capacity=2), "cap", clock=clock)
    all_events = []
    kinds = []
    for _ in frames:
        msg, events = s.process_frame(frame)
        all_events += events
        kinds += [e["type"] for e in msg["events"]]
        clock.tick()
    assert s.entries == 3 and s.exits == 0
    assert s.occupancy == 3 and s.max_occupancy == 3
    assert kinds == ["entry", "entry", "entry", "capacity_exceeded"]
    assert [e["event_type"] for e in all_events] == ["entry", "entry", "entry", "capacity_exceeded"]
    assert all(e["session_id"] == s.id for e in all_events)
    assert all_events[-1]["occupancy_after"] == 3
    assert s.capacity_exceeded


def test_settings_update_applies_live_and_moving_line_does_not_count():
    frame = np.zeros((400, 640, 3), np.uint8)
    still = [[[300, 50, 350, 170, 0.9]]] * 10   # person stands above the line
    clock = FakeClock()
    s = StreamSession(ScriptedDetector(still), DetectionSettings(), "cfg", clock=clock)
    for _ in range(5):
        s.process_frame(frame)
        clock.tick()
    new = DetectionSettings(line={"x1": 0.0, "y1": 0.1, "x2": 1.0, "y2": 0.1}, confidence=0.6, low_confidence=0.2)
    s.apply_settings(new)
    for _ in range(5):
        s.process_frame(frame)
        clock.tick()
    assert s.entries == 0 and s.exits == 0
    assert s.tracker.high_thresh == 0.6


def test_message_shape_and_snapshot():
    frame = np.zeros((400, 640, 3), np.uint8)
    clock = FakeClock(step=1.0)
    frames = [[[100, 100, 150, 220, 0.8]]] * 40
    s = StreamSession(ScriptedDetector(frames), DetectionSettings(), "shape", clock=clock, snapshot_interval=30)
    msg = None
    snaps = []
    for _ in range(40):
        msg, _ = s.process_frame(frame)
        snap = s.snapshot_if_due()
        if snap:
            snaps.append(snap)
        clock.tick()
    assert msg["type"] == "result"
    assert set(msg["counts"]) == {"live", "entries", "exits", "occupancy", "max_occupancy", "peak_live"}
    tr = msg["tracks"][0]
    assert tr["id"] == 1 and all(0 <= v <= 1 for v in tr["box"])
    assert msg["perf"]["fps"] > 0
    assert len(snaps) == 1 and snaps[0]["max_live_count"] == 1
    row = s.summary_fields(ended=True)
    assert row["status"] == "completed" and row["heatmap"]["max"] > 0


def test_min_box_height_filters_tiny_detections():
    frame = np.zeros((400, 640, 3), np.uint8)
    frames = [[[100, 100, 110, 105, 0.9]]] * 5      # 5 px tall
    s = StreamSession(ScriptedDetector(frames), DetectionSettings(min_box_height=0.05), "tiny", clock=FakeClock())
    for _ in range(5):
        msg, _ = s.process_frame(frame)
    assert msg["tracks"] == []
