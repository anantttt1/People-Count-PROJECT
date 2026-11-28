"""ByteTrack multi-object tracker (Zhang et al., 2022), implemented for per-session use.

Why our own implementation instead of ``model.track(persist=True)``: Ultralytics keeps one
tracker per model instance, so two browsers sharing a model would mix their track IDs.
Here every counting session owns an independent ``ByteTracker``.

Algorithm per frame:
1. Split detections into high-score (>= high_thresh) and low-score (low..high).
2. Predict every tracked and recently-lost track forward with a Kalman filter.
3. Match high-score detections to those tracks by IoU (Hungarian algorithm).
4. Match the low-score detections to tracks still unmatched - this keeps partially
   occluded people (whose detector score drops) attached to their existing ID.
5. Tracks with no match become "lost" and are kept for ``max_lost_seconds`` so a person who
   disappears behind someone for a moment gets the same ID back.
6. New tracks start only from high-score detections and must be matched on a second frame
   before they are reported (suppresses one-frame false positives).
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from .kalman import KalmanFilterXYWH

try:  # SciPy gives an optimal assignment; a greedy fallback keeps things working without it.
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:  # pragma: no cover - scipy is in requirements
    _lsa = None


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class Track:
    __slots__ = (
        "_tlwh", "score", "mean", "cov", "state", "is_activated", "track_id",
        "frame_id", "start_frame", "last_seen", "hits",
    )

    def __init__(self, tlbr: np.ndarray, score: float) -> None:
        x1, y1, x2, y2 = (float(v) for v in tlbr)
        self._tlwh = np.array([x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=float)
        self.score = float(score)
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.state = TrackState.NEW
        self.is_activated = False
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.last_seen = 0.0
        self.hits = 0

    # -- geometry -------------------------------------------------------------
    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        cx, cy, w, h = self.mean[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, w, h])

    @property
    def tlbr(self) -> np.ndarray:
        t = self.tlwh
        return np.array([t[0], t[1], t[0] + t[2], t[1] + t[3]])

    @staticmethod
    def _xywh(tlwh: np.ndarray) -> np.ndarray:
        x, y, w, h = tlwh
        return np.array([x + w / 2, y + h / 2, w, h])

    # -- lifecycle --------------------------------------------------------------
    def predict(self, kf: KalmanFilterXYWH) -> None:
        if self.mean is None:
            return
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[6] = mean[7] = 0.0  # don't let the size velocity run away while lost
        self.mean, self.cov = kf.predict(mean, self.cov)

    def activate(self, kf: KalmanFilterXYWH, frame_id: int, now: float) -> None:
        self.mean, self.cov = kf.initiate(self._xywh(self._tlwh))
        self.state = TrackState.TRACKED
        self.frame_id = self.start_frame = frame_id
        self.last_seen = now
        self.hits = 1
        self.is_activated = frame_id == 1

    def update(self, kf: KalmanFilterXYWH, det: "Track", frame_id: int, now: float) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, self._xywh(det._tlwh))
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = det.score
        self.frame_id = frame_id
        self.last_seen = now
        self.hits += 1

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes in (x1, y1, x2, y2) format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _iou_distance(tracks: list[Track], dets: list[Track], fuse_score: bool = False) -> np.ndarray:
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)))
    iou = iou_matrix(np.array([t.tlbr for t in tracks]), np.array([d.tlbr for d in dets]))
    if fuse_score:
        iou = iou * np.array([d.score for d in dets])[None, :]
    return 1.0 - iou


def _linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    matches: list[tuple[int, int]] = []
    if _lsa is not None:
        r_idx, c_idx = _lsa(cost)
        for r, c in zip(r_idx, c_idx):
            if cost[r, c] <= thresh:
                matches.append((int(r), int(c)))
    else:  # greedy fallback
        used_r: set[int] = set()
        used_c: set[int] = set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), cols)
            if cost[r, c] > thresh:
                break
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            matches.append((r, c))
    matched_r = {m[0] for m in matches}
    matched_c = {m[1] for m in matches}
    return matches, [r for r in range(rows) if r not in matched_r], [c for c in range(cols) if c not in matched_c]


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.4,
        low_thresh: float = 0.15,
        new_track_thresh: float | None = None,
        match_thresh: float = 0.8,
        max_lost_seconds: float = 2.0,
    ) -> None:
        self.kf = KalmanFilterXYWH()
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.frame_id = 0
        self._next_id = 0
        self.configure(high_thresh, low_thresh, new_track_thresh, match_thresh, max_lost_seconds)

    def configure(
        self,
        high_thresh: float,
        low_thresh: float,
        new_track_thresh: float | None = None,
        match_thresh: float = 0.8,
        max_lost_seconds: float = 2.0,
    ) -> None:
        if not 0.0 < low_thresh < high_thresh < 1.0:
            raise ValueError("expected 0 < low_thresh < high_thresh < 1")
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh if new_track_thresh is not None else min(high_thresh + 0.05, 0.95)
        self.match_thresh = match_thresh
        self.max_lost_seconds = max_lost_seconds

    def reset(self) -> None:
        self.tracked.clear()
        self.lost.clear()
        self.frame_id = 0
        self._next_id = 0

    def _assign_id(self, track: Track) -> None:
        if track.track_id == 0:
            self._next_id += 1
            track.track_id = self._next_id

    def update(self, detections: np.ndarray, now: float) -> list[Track]:
        """Advance one frame.

        ``detections``: array (N, 5) of x1, y1, x2, y2, score in pixels.
        Returns confirmed, currently visible tracks.
        """
        self.frame_id += 1
        dets = np.asarray(detections, dtype=float).reshape(-1, 5)
        scores = dets[:, 4]
        high = [Track(d[:4], d[4]) for d in dets[scores >= self.high_thresh]]
        low = [Track(d[:4], d[4]) for d in dets[(scores >= self.low_thresh) & (scores < self.high_thresh)]]

        activated: list[Track] = []
        refound: list[Track] = []
        newly_lost: list[Track] = []
        removed: list[Track] = []

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = _join(confirmed, self.lost)
        for t in pool:
            t.predict(self.kf)

        # 1) high-score detections vs all confirmed + lost tracks
        matches, u_track, u_high = _linear_assignment(_iou_distance(pool, high, fuse_score=True), self.match_thresh)
        for ti, di in matches:
            track = pool[ti]
            was_tracked = track.state == TrackState.TRACKED
            track.update(self.kf, high[di], self.frame_id, now)
            (activated if was_tracked else refound).append(track)

        # 2) low-score detections vs tracks that were visible last frame and are still unmatched
        remaining = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        matches, u_remaining, _ = _linear_assignment(_iou_distance(remaining, low), 0.5)
        for ti, di in matches:
            remaining[ti].update(self.kf, low[di], self.frame_id, now)
            activated.append(remaining[ti])
        for ti in u_remaining:
            track = remaining[ti]
            if track.state != TrackState.LOST:
                track.mark_lost()
                newly_lost.append(track)

        # 3) unconfirmed tracks (seen once) vs leftover high-score detections
        leftover = [high[i] for i in u_high]
        matches, u_unconf, u_left = _linear_assignment(_iou_distance(unconfirmed, leftover, fuse_score=True), 0.7)
        for ti, di in matches:
            unconfirmed[ti].update(self.kf, leftover[di], self.frame_id, now)
            activated.append(unconfirmed[ti])
        for ti in u_unconf:
            unconfirmed[ti].mark_removed()
            removed.append(unconfirmed[ti])

        # 4) start new tracks
        for di in u_left:
            track = leftover[di]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kf, self.frame_id, now)
            activated.append(track)

        # 5) expire tracks that have been lost too long
        for track in self.lost:
            if now - track.last_seen > self.max_lost_seconds:
                track.mark_removed()
                removed.append(track)

        self.tracked = [t for t in self.tracked if t.state == TrackState.TRACKED]
        self.tracked = _join(self.tracked, activated)
        self.tracked = _join(self.tracked, refound)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(newly_lost)
        self.lost = _sub(self.lost, removed)
        self.lost = [t for t in self.lost if t.state == TrackState.LOST]
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)

        output = []
        for t in self.tracked:
            if t.is_activated:
                self._assign_id(t)
                output.append(t)
        return output


def _join(a: list[Track], b: list[Track]) -> list[Track]:
    seen = {id(t) for t in a}
    out = list(a)
    for t in b:
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


def _sub(a: list[Track], b: list[Track]) -> list[Track]:
    drop = {id(t) for t in b}
    return [t for t in a if id(t) not in drop]


def _remove_duplicates(tracked: list[Track], lost: list[Track]) -> tuple[list[Track], list[Track]]:
    """If a tracked and a lost track overlap almost perfectly, keep the longer-lived one."""
    if not tracked or not lost:
        return tracked, lost
    dist = _iou_distance(tracked, lost)
    drop_t: set[int] = set()
    drop_l: set[int] = set()
    for p, q in zip(*np.where(dist < 0.15)):
        age_t = tracked[p].frame_id - tracked[p].start_frame
        age_l = lost[q].frame_id - lost[q].start_frame
        if age_t > age_l:
            drop_l.add(int(q))
        else:
            drop_t.add(int(p))
    return [t for i, t in enumerate(tracked) if i not in drop_t], [t for i, t in enumerate(lost) if i not in drop_l]
