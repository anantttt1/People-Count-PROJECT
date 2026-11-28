"""A moving test scene made from *real* people: person crops from Ultralytics' bus.jpg sample
photo are moved across a plain background, crossing the counting line.

The detector still runs for real on every frame - this only supplies motion that a still
photo can't. Used by the pipeline test and by the browser end-to-end test (written to a
.y4m file and fed to Chromium as a fake camera).

Default scene (frame 960x540, counting line at y = 0.55):
  * person A walks DOWN across the line      -> entry
  * person B walks UP across the line        -> exit
  * person C walks DOWN across the line      -> entry
Expected totals: 2 entries, 1 exit, occupancy 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Person boxes in bus.jpg (810x1080), the three people in full view.
BUS_PEOPLE = {
    "A": (46, 399, 237, 902),
    "B": (223, 405, 345, 860),
    "C": (668, 395, 810, 880),
}
W, H = 960, 540


@dataclass
class Mover:
    sprite: np.ndarray
    cx: float
    y_start: float   # top edge
    y_end: float
    first: int
    last: int

    def top(self, frame: int) -> float | None:
        if not self.first <= frame <= self.last:
            return None
        t = (frame - self.first) / max(self.last - self.first, 1)
        return self.y_start + (self.y_end - self.y_start) * t


def _sprite(bus: np.ndarray, key: str, height: int) -> np.ndarray:
    x1, y1, x2, y2 = BUS_PEOPLE[key]
    crop = bus[y1:y2, x1:x2]
    scale = height / crop.shape[0]
    return cv2.resize(crop, (max(int(crop.shape[1] * scale), 1), height), interpolation=cv2.INTER_AREA)


def background() -> np.ndarray:
    bg = np.empty((H, W, 3), np.uint8)
    bg[:] = (178, 184, 190)
    # floor tiles so the scene is not perfectly flat
    for x in range(0, W, 80):
        cv2.line(bg, (x, 0), (x, H), (168, 174, 180), 1)
    for y in range(0, H, 60):
        cv2.line(bg, (0, y), (W, y), (168, 174, 180), 1)
    return bg


def doorway_movers(bus: np.ndarray, fps_scale: float = 1.0) -> tuple[list[Mover], int]:
    """fps_scale > 1 slows everyone down (more frames for the same motion)."""
    h = 230
    f = lambda n: int(round(n * fps_scale))  # noqa: E731
    movers = [
        Mover(_sprite(bus, "A", h), cx=230, y_start=-h * 0.75, y_end=H - h * 0.2, first=f(0), last=f(70)),
        Mover(_sprite(bus, "B", h), cx=520, y_start=H - h * 0.2, y_end=-h * 0.75, first=f(85), last=f(155)),
        Mover(_sprite(bus, "C", h), cx=760, y_start=-h * 0.75, y_end=H - h * 0.2, first=f(170), last=f(240)),
    ]
    return movers, f(255)


def render(movers: list[Mover], frame: int, bg: np.ndarray | None = None) -> np.ndarray:
    img = (bg if bg is not None else background()).copy()
    for m in movers:
        top = m.top(frame)
        if top is None:
            continue
        sh, sw = m.sprite.shape[:2]
        x0, y0 = int(m.cx - sw / 2), int(round(top))
        ys, xs = max(y0, 0), max(x0, 0)
        ye, xe = min(y0 + sh, H), min(x0 + sw, W)
        if ye > ys and xe > xs:
            img[ys:ye, xs:xe] = m.sprite[ys - y0:ye - y0, xs - x0:xe - x0]
    return img


def write_y4m(path: Path, frames, fps: int = 30) -> None:
    """Write frames (BGR) as a YUV4MPEG2 (4:2:0) file - the format Chromium's fake camera reads."""
    path = Path(path)
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for img in frames:
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            f.write(b"FRAME\n")
            f.write(yuv.tobytes())
