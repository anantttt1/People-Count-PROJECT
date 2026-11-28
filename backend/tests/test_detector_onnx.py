"""Real YOLO inference (Ultralytics ONNX exports via onnxruntime) on real photos.

Needs models/yolo26n.onnx (+ yolo11n.onnx) and the sample photos; fetch them with
``python -m backend.tests.assets``. Skips when they are not present.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.vision.detector import OnnxYoloDetector, letterbox, load_detector
from backend.app.vision.model_files import ModelFileError, resolve_model
from backend.tests import assets

YOLO26 = assets.model_path("yolo26n.onnx")
YOLO11 = assets.model_path("yolo11n.onnx")
BUS = assets.image("bus.jpg")
ZIDANE = assets.image("zidane.jpg")

need26 = pytest.mark.skipif(YOLO26 is None or BUS is None, reason="yolo26n.onnx / bus.jpg not available")
need11 = pytest.mark.skipif(YOLO11 is None or BUS is None, reason="yolo11n.onnx / bus.jpg not available")


@pytest.fixture(scope="module")
def det26():
    return OnnxYoloDetector(YOLO26)


@pytest.fixture(scope="module")
def det11():
    return OnnxYoloDetector(YOLO11)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_letterbox_geometry():
    img = np.zeros((720, 1280, 3), np.uint8)
    out, gain, left, top = letterbox(img, 640, 640)
    assert out.shape == (640, 640, 3)
    assert gain == pytest.approx(0.5) and left == 0 and top == 140


@need26
def test_yolo26_finds_the_people_on_the_bus_photo(det26):
    assert det26.end2end and det26.person_class == 0
    dets = det26.detect(BUS, conf=0.4, iou=0.5)
    # bus.jpg: three pedestrians in full view + one half cut off at the left edge.
    # The bus itself (COCO class 5) must NOT be returned.
    assert len(dets) == 4, dets
    h, w = BUS.shape[:2]
    assert (dets[:, 0] >= 0).all() and (dets[:, 2] <= w).all() and (dets[:, 3] <= h).all()
    assert (dets[:, 4] >= 0.4).all()
    # every box is person-shaped (taller than wide)
    assert ((dets[:, 3] - dets[:, 1]) > (dets[:, 2] - dets[:, 0])).all()
    # the partly visible man touches the left edge and scores lower than the others
    edge = dets[dets[:, 0] < 5]
    assert len(edge) == 1 and edge[0, 4] < dets[:, 4].max()
    # a strict threshold keeps only the three fully visible people
    assert len(det26.detect(BUS, conf=0.7, iou=0.5)) == 3


@need26
def test_yolo26_two_people_on_zidane_photo(det26):
    if ZIDANE is None:
        pytest.skip("zidane.jpg not available")
    dets = det26.detect(ZIDANE, conf=0.4, iou=0.5)
    assert len(dets) == 2


@need26
def test_no_people_in_blank_frame(det26):
    assert len(det26.detect(np.full((480, 640, 3), 128, np.uint8), conf=0.25, iou=0.5)) == 0


@need11
def test_yolo11_raw_output_with_nms_agrees_with_yolo26(det11, det26):
    assert not det11.end2end and not det11.fixed_shape
    d11 = det11.detect(BUS, conf=0.4, iou=0.5)
    d26 = det26.detect(BUS, conf=0.4, iou=0.5)
    assert len(d11) == 4
    for box in d26:  # each YOLO26 person has a matching YOLO11 person
        assert max(_iou(box, other) for other in d11) > 0.6


@need26
def test_load_detector_prefers_configured_onnx_model():
    det, info = load_detector("yolo26n.onnx", "yolo11n.onnx", "", 640, True, allow_download=False)
    assert info.status == "ready" and info.name == "yolo-onnx" and info.model == "yolo26n.onnx"


@need26
def test_load_detector_falls_back_then_hog():
    det, info = load_detector("missing-model.onnx", "yolo26n.onnx", "", 640, True, allow_download=False)
    assert info.status == "degraded" and info.model == "yolo26n.onnx" and info.attempts
    det, info = load_detector("missing-a.onnx", "missing-b.onnx", "", 640, True, allow_download=False)
    assert info.name == "opencv-hog" and info.status == "degraded" and len(info.attempts) == 2


@need26
def test_sha256_pinning():
    with pytest.raises(ModelFileError):
        resolve_model("yolo26n.onnx", expected_sha256="0" * 64, allow_download=False)
