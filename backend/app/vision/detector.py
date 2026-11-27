"""Person detectors behind one small interface.

* ``YoloDetector`` - Ultralytics YOLO (default ``yolo26n.pt``), person class only.
* ``HogDetector``  - OpenCV's built-in HOG + linear SVM pedestrian detector. No download,
  CPU only, clearly less accurate. Used only as a fallback when YOLO cannot be loaded, and
  the UI says so.

``load_detector`` tries the configured model, then the fallback model, then HOG.
Every detector returns an ``(N, 5)`` float32 array: x1, y1, x2, y2, score (pixels).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

EMPTY = np.zeros((0, 5), dtype=np.float32)


class DetectorUnavailable(RuntimeError):
    pass


class Detector(Protocol):
    name: str
    model: str

    def detect(self, frame_bgr: np.ndarray, conf: float, iou: float) -> np.ndarray: ...


@dataclass
class DetectorInfo:
    status: str = "loading"            # loading | ready | degraded | failed
    name: str = ""
    model: str = ""
    device: str = ""
    message: str = ""
    load_seconds: float = 0.0
    attempts: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "status": self.status,
            "name": self.name,
            "model": self.model,
            "device": self.device,
            "message": self.message,
        }


class YoloDetector:
    name = "yolo"

    def __init__(self, model_path: str, device: str = "", imgsz: int = 640) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable("ultralytics is not installed (pip install -r requirements.txt)") from exc
        self.model = model_path
        self.imgsz = imgsz
        self._lock = threading.Lock()  # one inference at a time on a shared model
        self._yolo = YOLO(model_path)
        names = self._yolo.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        person = [int(k) for k, v in items if str(v).lower() == "person"]
        if not person:
            raise DetectorUnavailable(f"model {model_path} has no 'person' class")
        self.person_class = person[0]
        self._kwargs: dict = {"imgsz": imgsz, "classes": [self.person_class], "verbose": False, "max_det": 300}
        if device:
            self._kwargs["device"] = device
        # Warm-up so the first real frame isn't slow, and fail fast if the model is broken.
        self.detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), conf=0.25, iou=0.5)
        try:
            self.device = str(next(self._yolo.model.parameters()).device)
        except Exception:  # noqa: BLE001 - exported models have no torch parameters
            self.device = device or "auto"

    def detect(self, frame_bgr: np.ndarray, conf: float, iou: float) -> np.ndarray:
        with self._lock:
            result = self._yolo.predict(frame_bgr, conf=conf, iou=iou, **self._kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return EMPTY.copy()
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes.conf.cpu().numpy().astype(np.float32)
        return np.concatenate([xyxy, scores[:, None]], axis=1)


class HogDetector:
    """OpenCV HOG pedestrian detector.

    The SVM returns a margin, not a probability; it is mapped to 0..1 with
    ``1 - exp(-margin)`` so thresholds behave roughly like YOLO confidences.
    Works best on upright, full-body people roughly 100-400 px tall.
    """

    name = "opencv-hog"
    model = "HOG + linear SVM (OpenCV built-in)"
    device = "cpu"

    def __init__(self, max_width: int = 640) -> None:
        self.max_width = max_width
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray, conf: float, iou: float) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_width / float(w))
        img = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
        with self._lock:
            rects, weights = self._hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return EMPTY.copy()
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        scores = 1.0 - np.exp(-np.clip(weights, 0, None))
        boxes = [[float(x), float(y), float(bw), float(bh)] for x, y, bw, bh in rects]
        keep = cv2.dnn.NMSBoxes(boxes, scores.tolist(), float(conf), float(iou))
        keep = np.asarray(keep, dtype=int).reshape(-1)
        out = []
        for i in keep:
            x, y, bw, bh = boxes[i]
            out.append([x / scale, y / scale, (x + bw) / scale, (y + bh) / scale, float(scores[i])])
        return np.asarray(out, dtype=np.float32).reshape(-1, 5)


def load_detector(model: str, fallback_model: str, device: str, imgsz: int, allow_hog: bool) -> tuple[Detector, DetectorInfo]:
    """Load the best available detector. Raises DetectorUnavailable if nothing works."""
    info = DetectorInfo()
    start = time.perf_counter()
    for candidate in [m for m in (model, fallback_model) if m]:
        try:
            det = YoloDetector(candidate, device=device, imgsz=imgsz)
        except Exception as exc:  # noqa: BLE001 - any load failure moves to the next option
            msg = f"{candidate}: {type(exc).__name__}: {exc}"
            log.warning("YOLO model failed to load - %s", msg)
            info.attempts.append(msg)
            continue
        info.status = "ready" if candidate == model else "degraded"
        info.name, info.model, info.device = det.name, candidate, det.device
        info.message = "" if candidate == model else f"Using fallback model {candidate}; {model} failed to load."
        info.load_seconds = time.perf_counter() - start
        log.info("Detector ready: YOLO %s on %s (%.1fs)", candidate, det.device, info.load_seconds)
        return det, info

    if allow_hog:
        det = HogDetector(max_width=imgsz)
        info.status = "degraded"
        info.name, info.model, info.device = det.name, det.model, det.device
        info.message = (
            "YOLO could not be loaded, so the much less accurate OpenCV HOG detector is running. "
            "Install the requirements and check the server log to enable YOLO."
        )
        info.load_seconds = time.perf_counter() - start
        log.warning("Detector fallback: OpenCV HOG (YOLO unavailable: %s)", "; ".join(info.attempts))
        return det, info

    info.status = "failed"
    info.message = "No person detector could be loaded: " + "; ".join(info.attempts)
    raise DetectorUnavailable(info.message)
