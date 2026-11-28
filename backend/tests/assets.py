"""Real test inputs (photos with people, YOLO models) - downloaded, never committed.

    python -m backend.tests.assets      # fetch bus.jpg / zidane.jpg + the default ONNX models

Tests that need these skip cleanly when they are missing.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
_IMAGES = {
    # Ultralytics' standard sample images (people + a bus / two people on a pitch)
    "bus.jpg": "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
    "zidane.jpg": "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg",
}


def image(name: str) -> np.ndarray | None:
    path = ASSETS / name
    return cv2.imread(str(path)) if path.is_file() else None


def model_path(name: str) -> Path | None:
    from backend.app.vision.model_files import ModelFileError, resolve_model

    try:
        return resolve_model(name, allow_download=False)
    except ModelFileError:
        return None


def main() -> int:
    from backend.app.vision.model_files import resolve_model

    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, url in _IMAGES.items():
        dest = ASSETS / name
        if not dest.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, dest)
    for model in ("yolo26n.onnx", "yolo11n.onnx"):
        print("model:", resolve_model(model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
