"""Person detectors behind one small interface.

* ``OnnxYoloDetector`` - an Ultralytics YOLO model exported to ONNX (default
  ``yolo26n.onnx``), run with onnxruntime. No PyTorch needed. Handles both export styles:
  end-to-end / NMS-free (YOLO26: ``(1, 300, 6)``) and raw anchors (YOLO11/v8: ``(1, 84, N)``,
  NMS applied here).
* ``YoloDetector`` - Ultralytics YOLO through the ``ultralytics`` package (``.pt`` weights,
  GPU, TensorRT ...). Optional: ``pip install -r requirements-ultralytics.txt``.
* ``HogDetector``  - OpenCV's built-in HOG + linear SVM pedestrian detector. No download,
  CPU only, clearly less accurate. Used only as a fallback when YOLO cannot be loaded, and
  the UI says so.

``load_detector`` tries the configured model, then the fallback model, then HOG.
Every detector returns an ``(N, 5)`` float32 array: x1, y1, x2, y2, score (pixels).
Only the COCO ``person`` class is ever returned.
"""
from __future__ import annotations

import ast
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .model_files import resolve_model

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


def letterbox(img: np.ndarray, new_w: int, new_h: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad with grey (114) to exactly new_w x new_h,
    the same preprocessing Ultralytics uses. Returns (image, gain, pad_left, pad_top)."""
    h, w = img.shape[:2]
    gain = min(new_w / w, new_h / h)
    rw, rh = max(int(round(w * gain)), 1), max(int(round(h * gain)), 1)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rw, rh) != (w, h) else img
    left, top = (new_w - rw) // 2, (new_h - rh) // 2
    out = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    out[top:top + rh, left:left + rw] = resized
    return out, gain, left, top


class OnnxYoloDetector:
    name = "yolo-onnx"

    def __init__(self, model_path: str | Path, device: str = "", imgsz: int = 640, threads: int = 0) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorUnavailable("onnxruntime is not installed (pip install -r requirements.txt)") from exc
        path = Path(model_path)
        self.model = path.name
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or max(1, os.cpu_count() or 1)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        elif device.startswith("cuda"):
            log.warning("YOLO_DEVICE=%s requested but onnxruntime has no CUDA provider; using CPU", device)
        try:
            self._sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception as exc:  # noqa: BLE001
            raise DetectorUnavailable(f"onnxruntime could not load {path.name}: {exc}") from exc
        self.device = "cuda" if self._sess.get_providers()[0] == "CUDAExecutionProvider" else "cpu"

        meta = self._sess.get_modelmeta().custom_metadata_map
        if meta.get("task", "detect") != "detect":
            raise DetectorUnavailable(f"{path.name} is a '{meta.get('task')}' model; a detect model is required")
        try:
            names = ast.literal_eval(meta.get("names", "{}"))
        except (ValueError, SyntaxError):
            names = {}
        person = [int(k) for k, v in dict(names).items() if str(v).lower() == "person"]
        if not person:
            raise DetectorUnavailable(f"{path.name} has no 'person' class in its metadata (not an Ultralytics export?)")
        self.person_class = person[0]
        self.num_classes = len(names)

        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape  # [1, 3, H, W]; H/W are strings for dynamic exports
        self.fixed_shape = isinstance(shape[2], int) and isinstance(shape[3], int)
        self.stride = int(meta.get("stride", 32) or 32)
        self.imgsz = (int(shape[3]), int(shape[2])) if self.fixed_shape else (imgsz, imgsz)
        out_shape = self._sess.get_outputs()[0].shape
        self.end2end = str(meta.get("end2end", "")).lower() == "true" or (len(out_shape) == 3 and out_shape[-1] == 6)
        self._lock = threading.Lock()
        # Warm-up and fail fast if the graph can't run.
        self.detect(np.zeros((self.imgsz[1], self.imgsz[0], 3), dtype=np.uint8), conf=0.25, iou=0.5)

    def _input_size(self, w: int, h: int) -> tuple[int, int]:
        if self.fixed_shape:
            return self.imgsz
        # Dynamic export: rectangular input (multiple of stride) -> less padding, faster.
        size = self.imgsz[0]
        gain = min(size / w, size / h)
        nw = int(np.ceil(w * gain / self.stride) * self.stride)
        nh = int(np.ceil(h * gain / self.stride) * self.stride)
        return nw, nh

    def detect(self, frame_bgr: np.ndarray, conf: float, iou: float) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        in_w, in_h = self._input_size(w, h)
        img, gain, left, top = letterbox(frame_bgr, in_w, in_h)
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        with self._lock:
            out = self._sess.run(None, {self._input_name: blob})[0]

        if self.end2end:
            # (1, max_det, 6): x1, y1, x2, y2, score, class - already NMS-free.
            pred = out[0]
            pred = pred[(pred[:, 5].astype(int) == self.person_class) & (pred[:, 4] >= conf)]
            boxes, scores = pred[:, :4].astype(np.float32), pred[:, 4].astype(np.float32)
        else:
            # (1, 4 + classes, anchors): cx, cy, w, h, class scores (sigmoid applied).
            pred = out[0]
            scores = pred[4 + self.person_class]
            keep = scores >= conf
            if not keep.any():
                return EMPTY.copy()
            cx, cy, bw, bh = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
            scores = scores[keep].astype(np.float32)
            xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
            idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), float(conf), float(iou))
            idx = np.asarray(idx, dtype=int).reshape(-1)[:300]
            xywh, scores = xywh[idx], scores[idx]
            boxes = np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], axis=1)

        if len(boxes) == 0:
            return EMPTY.copy()
        boxes = boxes.astype(np.float32)
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / gain, 0, w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / gain, 0, h)
        valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        return np.concatenate([boxes[valid], scores[valid, None]], axis=1).astype(np.float32)


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


def build_yolo(candidate: str, device: str, imgsz: int, model_sha256: str = "", allow_download: bool = True) -> Detector:
    """ONNX files run on onnxruntime; anything else (.pt, .engine, ...) goes to Ultralytics."""
    if candidate.lower().endswith(".onnx"):
        path = resolve_model(candidate, expected_sha256=model_sha256, allow_download=allow_download)
        return OnnxYoloDetector(path, device=device, imgsz=imgsz)
    return YoloDetector(candidate, device=device, imgsz=imgsz)


def load_detector(
    model: str,
    fallback_model: str,
    device: str,
    imgsz: int,
    allow_hog: bool,
    model_sha256: str = "",
    allow_download: bool = True,
) -> tuple[Detector, DetectorInfo]:
    """Load the best available detector. Raises DetectorUnavailable if nothing works."""
    info = DetectorInfo()
    start = time.perf_counter()
    for candidate in [m for m in (model, fallback_model) if m]:
        try:
            det = build_yolo(
                candidate, device, imgsz,
                model_sha256=model_sha256 if candidate == model else "",
                allow_download=allow_download,
            )
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
