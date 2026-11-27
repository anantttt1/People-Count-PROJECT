"""Entry / exit counting on a virtual line.

For each tracked person we follow one anchor point (box centre or bottom-centre) and
work out which side of the line it is on, using the sign of the 2-D cross product.

Duplicate-count protection:
* hysteresis band - a point within ``hysteresis`` x frame-diagonal of the line is "on the
  line" and never changes the recorded side, so a person standing on the line and swaying
  is not counted again and again;
* a person is counted only when their *stable* side flips, so the same track can't be
  counted twice in the same direction without first crossing back;
* the crossing must happen within the drawn segment (not its infinite extension);
* only confirmed tracks are fed in (the tracker suppresses one-frame flickers).

Direction: a crossing from the negative side to the positive side is an *entry*
(``invert=True`` swaps this). For a left-to-right horizontal line in image coordinates
the positive side is *below* the line, i.e. default entry = moving down the image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

Anchor = Literal["center", "bottom"]
Direction = Literal["entry", "exit"]


@dataclass(frozen=True)
class CountingLine:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for v in (self.x1, self.y1, self.x2, self.y2):
            if not 0.0 <= v <= 1.0:
                raise ValueError("line coordinates must be normalised to [0, 1]")
        if abs(self.x1 - self.x2) < 1e-3 and abs(self.y1 - self.y2) < 1e-3:
            raise ValueError("line endpoints must differ")


@dataclass(frozen=True)
class Crossing:
    track_id: int
    direction: Direction
    timestamp: float


@dataclass
class _TrackSide:
    side: int = 0                     # -1, +1, or 0 (not yet known)
    last_point: tuple[float, float] | None = None
    last_seen: float = 0.0
    counted: list[Direction] = field(default_factory=list)


def anchor_point(box: tuple[float, float, float, float], anchor: Anchor) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    if anchor == "bottom":
        return ((x1 + x2) / 2.0, y2)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class LineCounter:
    def __init__(
        self,
        line: CountingLine,
        invert: bool = False,
        anchor: Anchor = "center",
        hysteresis: float = 0.015,
        forget_after_seconds: float = 10.0,
    ) -> None:
        self.line = line
        self.invert = invert
        self.anchor = anchor
        self.hysteresis = hysteresis
        self.forget_after_seconds = forget_after_seconds
        self._state: dict[int, _TrackSide] = {}

    def reconfigure(self, line: CountingLine, invert: bool, anchor: Anchor) -> None:
        """Change geometry. Per-track sides are re-learned so moving the line never counts anyone."""
        if (line, invert, anchor) != (self.line, self.invert, self.anchor):
            self.line, self.invert, self.anchor = line, invert, anchor
            self._state.clear()

    def reset(self) -> None:
        self._state.clear()

    def _signed_distance(self, px: float, py: float, w: float, h: float) -> tuple[float, float]:
        """Return (signed perpendicular distance in px, projection t along the segment)."""
        ax, ay = self.line.x1 * w, self.line.y1 * h
        bx, by = self.line.x2 * w, self.line.y2 * h
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        length = length_sq ** 0.5
        cross = dx * (py - ay) - dy * (px - ax)
        t = ((px - ax) * dx + (py - ay) * dy) / length_sq
        return cross / length, t

    def update(
        self,
        tracks: Iterable[tuple[int, tuple[float, float, float, float]]],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> list[Crossing]:
        """``tracks``: (track_id, normalised box). Returns crossings detected this frame."""
        w, h = float(frame_width), float(frame_height)
        margin = self.hysteresis * (w * w + h * h) ** 0.5
        crossings: list[Crossing] = []

        for track_id, box in tracks:
            nx, ny = anchor_point(box, self.anchor)
            px, py = nx * w, ny * h
            dist, _ = self._signed_distance(px, py, w, h)
            side = 1 if dist > margin else -1 if dist < -margin else 0

            st = self._state.get(track_id)
            if st is None:
                st = self._state[track_id] = _TrackSide(side=side, last_point=(px, py), last_seen=now)
                continue
            prev_point = st.last_point
            st.last_point = (px, py)
            st.last_seen = now
            if side == 0:
                continue
            if st.side == 0:
                st.side = side
                continue
            if side == st.side:
                continue

            # Stable side flipped. Check the crossing happened within the drawn segment.
            mx = (px + prev_point[0]) / 2.0 if prev_point else px
            my = (py + prev_point[1]) / 2.0 if prev_point else py
            _, t = self._signed_distance(mx, my, w, h)
            previous_side, st.side = st.side, side
            if not -0.05 <= t <= 1.05:
                continue
            forward = previous_side < 0 < side
            direction: Direction = "entry" if forward != self.invert else "exit"
            st.counted.append(direction)
            crossings.append(Crossing(track_id=track_id, direction=direction, timestamp=now))

        stale = [tid for tid, st in self._state.items() if now - st.last_seen > self.forget_after_seconds]
        for tid in stale:
            del self._state[tid]
        return crossings
