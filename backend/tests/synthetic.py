"""Synthetic test scenes: drawn human silhouettes walking across the frame.

Used only by tests. The detectors run for real on these images; nothing here fakes a
detection result. OpenCV's HOG pedestrian detector reliably fires on these silhouettes,
which lets the whole pipeline be exercised without a camera or model download.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BACKGROUND = (186, 192, 198)


def draw_person(canvas: np.ndarray, cx: float, top: float, height: float, color=(45, 42, 40)) -> None:
    s = height / 180.0

    def p(x: float, y: float) -> tuple[int, int]:
        return int(cx + x * s), int(top + y * s)

    cv2.circle(canvas, p(0, 14), max(int(13 * s), 1), color, -1)
    cv2.ellipse(canvas, p(0, 32), (max(int(8 * s), 1), max(int(5 * s), 1)), 0, 0, 360, color, -1)
    cv2.fillPoly(canvas, [np.array([p(-22, 34), p(22, 34), p(18, 100), p(-18, 100)])], color)
    cv2.line(canvas, p(-21, 38), p(-30, 95), color, max(int(9 * s), 1))
    cv2.line(canvas, p(21, 38), p(30, 95), color, max(int(9 * s), 1))
    cv2.line(canvas, p(-9, 98), p(-14, 176), color, max(int(13 * s), 1))
    cv2.line(canvas, p(9, 98), p(14, 176), color, max(int(13 * s), 1))


@dataclass
class Walker:
    """A silhouette whose top edge moves linearly from y_start to y_end between two frames."""

    cx: float
    y_start: float
    y_end: float
    first_frame: int
    last_frame: int
    height: float = 170.0
    color: tuple[int, int, int] = (45, 42, 40)

    def top_at(self, frame: int) -> float | None:
        if not self.first_frame <= frame <= self.last_frame:
            return None
        t = (frame - self.first_frame) / max(self.last_frame - self.first_frame, 1)
        return self.y_start + (self.y_end - self.y_start) * t


def render(walkers: list[Walker], frame: int, width: int = 640, height: int = 400) -> np.ndarray:
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    for w in walkers:
        top = w.top_at(frame)
        if top is not None:
            draw_person(img, w.cx, top, w.height, w.color)
    return img


def doorway_scene() -> tuple[list[Walker], int]:
    """Person A walks down (entry with default line), later person B walks up (exit).

    Frame size 640x400, counting line at y = 0.55 (220 px). The silhouette centre moves
    from well above to well below the line (and back for B).
    """
    walkers = [
        Walker(cx=230, y_start=-20, y_end=250, first_frame=0, last_frame=60),
        Walker(cx=420, y_start=250, y_end=-20, first_frame=75, last_frame=135, color=(60, 50, 44)),
    ]
    return walkers, 145
