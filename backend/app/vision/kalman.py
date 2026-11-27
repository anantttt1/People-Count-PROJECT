"""Constant-velocity Kalman filter over (cx, cy, aspect, height), as used by SORT/ByteTrack.

State:       [cx, cy, a, h, vcx, vcy, va, vh]
Measurement: [cx, cy, a, h]   where a = width / height
Noise is scaled by box height so near (large) and far (small) people behave alike.
"""
from __future__ import annotations

import numpy as np

_STD_POS = 1.0 / 20
_STD_VEL = 1.0 / 160


class KalmanFilterXYAH:
    def __init__(self) -> None:
        ndim = 4
        self._motion = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = 1.0
        self._update = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4)]
        h = max(float(measurement[3]), 1.0)
        std = [
            2 * _STD_POS * h, 2 * _STD_POS * h, 1e-2, 2 * _STD_POS * h,
            10 * _STD_VEL * h, 10 * _STD_VEL * h, 1e-5, 10 * _STD_VEL * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1.0)
        std = [_STD_POS * h, _STD_POS * h, 1e-2, _STD_POS * h, _STD_VEL * h, _STD_VEL * h, 1e-5, _STD_VEL * h]
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + np.diag(np.square(std))
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1.0)
        std = [_STD_POS * h, _STD_POS * h, 1e-1, _STD_POS * h]
        return self._update @ mean, self._update @ cov @ self._update.T + np.diag(np.square(std))

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        # K = P H^T S^-1, solved without an explicit inverse
        gain = np.linalg.solve(proj_cov, (cov @ self._update.T).T).T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ gain.T
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov
