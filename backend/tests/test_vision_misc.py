import cv2
import numpy as np
import pytest

from backend.app.vision.density import coverage, density
from backend.app.vision.detector import HogDetector, load_detector
from backend.app.vision.frames import FrameError, decode_jpeg, jpeg_size
from backend.app.vision.heatmap import Heatmap
from backend.tests.synthetic import BACKGROUND, draw_person


def jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def test_coverage_uses_union_not_sum():
    a = (0.0, 0.0, 0.5, 0.5)
    assert coverage([a]) == pytest.approx(0.25, abs=0.02)
    assert coverage([a, a]) == pytest.approx(0.25, abs=0.02)   # identical boxes counted once
    assert density([])["level"] == "low"
    assert density([(0, 0, 1, 1)])["level"] == "high"


def test_heatmap_accumulates_and_clamps():
    hm = Heatmap(cols=4, rows=2)
    hm.add([(0.1, 0.1), (0.1, 0.1), (0.99, 0.99), (1.5, -1)])
    d = hm.to_dict()
    assert d["cells"][0][0] == 2
    assert d["cells"][1][3] == 1
    assert d["cells"][0][3] == 1      # out-of-range point clamped to the edge
    assert d["max"] == 2


def test_jpeg_size_and_decode():
    img = np.zeros((120, 200, 3), np.uint8)
    data = jpeg(img)
    assert jpeg_size(data) == (200, 120)
    assert decode_jpeg(data, 1_000_000, 1920).shape == (120, 200, 3)


def test_rejects_non_jpeg_and_oversized():
    ok, png = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(FrameError):
        decode_jpeg(png.tobytes(), 1_000_000, 1920)
    with pytest.raises(FrameError):
        decode_jpeg(b"\xff\xd8garbage", 1_000_000, 1920)
    big = jpeg(np.zeros((10, 2000, 3), np.uint8))
    with pytest.raises(FrameError, match="limit"):
        decode_jpeg(big, 5_000_000, 1920)
    with pytest.raises(FrameError, match="bytes"):
        decode_jpeg(big, 100, 4096)


def test_decompression_bomb_header_rejected_before_decode():
    data = bytearray(jpeg(np.zeros((16, 16, 3), np.uint8)))
    # patch the SOF0 dimensions to 60000 x 60000
    i = data.index(b"\xff\xc0")
    data[i + 5:i + 9] = (60000).to_bytes(2, "big") + (60000).to_bytes(2, "big")
    assert jpeg_size(bytes(data)) == (60000, 60000)
    with pytest.raises(FrameError, match="limit"):
        decode_jpeg(bytes(data), 1_000_000, 1920)


def test_hog_detects_silhouettes_and_ignores_empty_scene():
    det = HogDetector()
    img = np.empty((400, 640, 3), np.uint8)
    img[:] = BACKGROUND
    assert len(det.detect(img, conf=0.3, iou=0.5)) == 0
    draw_person(img, 200, 80, 180)
    draw_person(img, 460, 100, 180)
    out = det.detect(img, conf=0.3, iou=0.5)
    assert len(out) == 2
    centres = sorted(((b[0] + b[2]) / 2) for b in out)
    assert abs(centres[0] - 200) < 30 and abs(centres[1] - 460) < 30
    assert ((out[:, 4] > 0) & (out[:, 4] <= 1)).all()


def test_loader_falls_back_to_hog_when_yolo_missing():
    # ultralytics is not importable in this test environment -> HOG fallback, flagged degraded
    try:
        import ultralytics  # noqa: F401
        pytest.skip("ultralytics installed; fallback path not exercised")
    except ImportError:
        pass
    det, info = load_detector("yolo26n.pt", "yolo11n.pt", "", 640, allow_hog=True)
    assert det.name == "opencv-hog"
    assert info.status == "degraded" and "HOG" in info.message
    assert len(info.attempts) == 2


def test_loader_fails_cleanly_without_fallback():
    try:
        import ultralytics  # noqa: F401
        pytest.skip("ultralytics installed")
    except ImportError:
        pass
    from backend.app.vision.detector import DetectorUnavailable
    with pytest.raises(DetectorUnavailable):
        load_detector("yolo26n.pt", "", "", 640, allow_hog=False)
