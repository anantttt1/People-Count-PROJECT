"""Coarse occupancy heatmap.

Each processed frame adds one hit to the grid cell under every tracked person's anchor
point. The grid is deliberately coarse (32 x 18 cells by default) - it shows *where* people
spend time, never who they are, and it is the only spatial data that gets stored.
"""
from __future__ import annotations

import numpy as np


class Heatmap:
    def __init__(self, cols: int = 32, rows: int = 18) -> None:
        self.cols = cols
        self.rows = rows
        self.grid = np.zeros((rows, cols), dtype=np.int64)

    def add(self, points: list[tuple[float, float]]) -> None:
        for x, y in points:
            c = min(max(int(x * self.cols), 0), self.cols - 1)
            r = min(max(int(y * self.rows), 0), self.rows - 1)
            self.grid[r, c] += 1

    def reset(self) -> None:
        self.grid[:] = 0

    def to_dict(self) -> dict:
        """Raw counts (ints) plus size - JSON friendly and small enough to store per session."""
        return {"cols": self.cols, "rows": self.rows, "max": int(self.grid.max()), "cells": self.grid.tolist()}
