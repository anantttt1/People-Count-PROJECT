"""Crowd density from the share of the frame covered by people.

Overlapping boxes would be double counted by summing areas, so coverage is measured on a
low-resolution occupancy mask (union of boxes). Levels are heuristics for a typical
doorway / room camera and are documented as such in the UI.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

_MASK_W, _MASK_H = 96, 54
LEVELS = ((0.15, "low"), (0.35, "moderate"), (1.01, "high"))


def coverage(boxes: Iterable[tuple[float, float, float, float]]) -> float:
    mask = np.zeros((_MASK_H, _MASK_W), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        c1, c2 = int(max(x1, 0.0) * _MASK_W), int(np.ceil(min(x2, 1.0) * _MASK_W))
        r1, r2 = int(max(y1, 0.0) * _MASK_H), int(np.ceil(min(y2, 1.0) * _MASK_H))
        if c2 > c1 and r2 > r1:
            mask[r1:r2, c1:c2] = True
    return float(mask.mean())


def density(boxes: list[tuple[float, float, float, float]]) -> dict:
    cov = coverage(boxes)
    level = next(name for limit, name in LEVELS if cov < limit)
    return {"coverage": round(cov, 3), "level": level, "people": len(boxes)}
