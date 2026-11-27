import pytest

from backend.app.vision.line_counter import CountingLine, LineCounter

W, H = 640, 400
LINE = CountingLine(0.05, 0.5, 0.95, 0.5)


def box_at(cx, cy, w=0.06, h=0.2):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def walk(counter, track_id, points, t0=0.0):
    events = []
    for i, (x, y) in enumerate(points):
        events += counter.update([(track_id, box_at(x, y))], W, H, now=t0 + i * 0.1)
    return events


def ys(a, b, n=20):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def test_walking_down_is_entry_and_up_is_exit():
    c = LineCounter(LINE)
    down = walk(c, 1, [(0.4, y) for y in ys(0.2, 0.8)])
    up = walk(c, 2, [(0.6, y) for y in ys(0.8, 0.2)])
    assert [e.direction for e in down] == ["entry"]
    assert [e.direction for e in up] == ["exit"]


def test_invert_direction():
    c = LineCounter(LINE, invert=True)
    assert [e.direction for e in walk(c, 1, [(0.4, y) for y in ys(0.2, 0.8)])] == ["exit"]


def test_same_track_back_and_forth_counts_each_real_crossing():
    c = LineCounter(LINE)
    path = [(0.5, y) for y in ys(0.2, 0.8)] + [(0.5, y) for y in ys(0.8, 0.2)] + [(0.5, y) for y in ys(0.2, 0.8)]
    assert [e.direction for e in walk(c, 7, path)] == ["entry", "exit", "entry"]


def test_standing_on_the_line_and_swaying_is_not_counted():
    c = LineCounter(LINE, hysteresis=0.015)   # band ~ +/- 11 px on a 640x400 frame
    path = [(0.5, 0.3)] + [(0.5, 0.5 + (0.012 if i % 2 else -0.012)) for i in range(40)]
    assert walk(c, 3, path) == []


def test_crossing_outside_the_segment_is_ignored():
    c = LineCounter(CountingLine(0.4, 0.5, 0.6, 0.5))
    assert walk(c, 1, [(0.1, y) for y in ys(0.2, 0.8)]) == []
    assert [e.direction for e in walk(c, 2, [(0.5, y) for y in ys(0.2, 0.8)])] == ["entry"]


def test_track_first_seen_beyond_line_is_not_counted():
    c = LineCounter(LINE)
    assert walk(c, 1, [(0.5, y) for y in ys(0.7, 0.9)]) == []


def test_bottom_anchor_uses_feet():
    c = LineCounter(LINE, anchor="bottom")
    # box centre stays above the line (0.35..0.42) but the feet (centre + 0.1) cross it
    events = walk(c, 1, [(0.5, y) for y in ys(0.30, 0.45)])
    assert [e.direction for e in events] == ["entry"]


def test_diagonal_line():
    c = LineCounter(CountingLine(0.1, 0.1, 0.9, 0.9))
    # moving from top-right to bottom-left crosses the diagonal
    pts = [(0.8 - 0.6 * i / 19, 0.2 + 0.6 * i / 19) for i in range(20)]
    assert len(walk(c, 1, pts)) == 1


def test_reconfigure_never_creates_counts():
    c = LineCounter(LINE)
    walk(c, 1, [(0.5, 0.3)] * 5)
    c.reconfigure(CountingLine(0.05, 0.2, 0.95, 0.2), invert=False, anchor="center")
    # person stays still; moving the line over them must not count
    assert walk(c, 1, [(0.5, 0.3)] * 5, t0=1.0) == []


def test_invalid_line_rejected():
    with pytest.raises(ValueError):
        CountingLine(0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        CountingLine(-0.1, 0.5, 0.5, 0.5)
