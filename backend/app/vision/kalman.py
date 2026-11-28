"""Constant-velocity Kalman filter over the box centre, width and height (XYWH).

State:       [cx, cy, w, h, vcx, vcy, vw, vh]
Measurement: [cx, cy, w, h]

Why XYWH and not ByteTrack's original XYAH (centre, aspect ratio, height): the original
filter trusts its aspect-ratio estimate far more than new measurements, so the aspect adapts
very slowly. When a person walks into view from the frame edge - the normal case for a
doorway camera - their box grows taller while the width stays the same, the predicted box
drifts away from the detection and the tracker assigns a new ID. Modelling width and height
independently (as BoT-SORT does) follows that change; this was verified with real YOLO
detections in ``tests/test_pipeline_yolo.py``.

Position noise is proportional to width (x, w) and height (y, h), so near (large) and far
(small) people behave alike.
"""
from __future__ import annotations

import numpy as np

_STD_POS = 1.0 / 20
_STD_VEL = 1.0 / 160


class KalmanFilterXYWH:
    def __init__(self) -> None:
        ndim = 4
        self._motion = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = 1.0
        self._update = np.eye(ndim, 2 * ndim)

    @staticmethod
    def _wh(vec: np.ndarray) -> tuple[float, float]:
        return max(float(vec[2]), 1.0), max(float(vec[3]), 1.0)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4)]
        w, h = self._wh(measurement)
        std = [
            2 * _STD_POS * w, 2 * _STD_POS * h, 2 * _STD_POS * w, 2 * _STD_POS * h,
            10 * _STD_VEL * w, 10 * _STD_VEL * h, 10 * _STD_VEL * w, 10 * _STD_VEL * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, h = self._wh(mean)
        std = [
            _STD_POS * w, _STD_POS * h, _STD_POS * w, _STD_POS * h,
            _STD_VEL * w, _STD_VEL * h, _STD_VEL * w, _STD_VEL * h,
        ]
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + np.diag(np.square(std))
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, h = self._wh(mean)
        std = [_STD_POS * w, _STD_POS * h, _STD_POS * w, _STD_POS * h]
        return self._update @ mean, self._update @ cov @ self._update.T + np.diag(np.square(std))

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        # K = P H^T S^-1, solved without an explicit inverse
        gain = np.linalg.solve(proj_cov, (cov @ self._update.T).T).T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ gain.T
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov
