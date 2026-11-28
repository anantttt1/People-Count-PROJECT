"""Locating and downloading YOLO model files.

* A path that exists on disk is used as is.
* A bare file name is looked up in ``MODELS_DIR`` (``<repo>/models``).
* The official Ultralytics ONNX exports listed in ``OFFICIAL_ONNX`` are downloaded on first
  use from Ultralytics' GitHub releases over HTTPS, written atomically, then validated
  (it must be an Ultralytics *detect* model with a ``person`` class). If ``YOLO_MODEL_SHA256``
  is set, the file must match that hash exactly.

ONNX files are plain data for onnxruntime - unlike pickled ``.pt`` checkpoints they cannot
run code when loaded - which is why ONNX is the default model format.
``.pt`` files are handled by Ultralytics itself (it downloads its own official weights).
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from ..config import REPO_ROOT

log = logging.getLogger(__name__)

MODELS_DIR = REPO_ROOT / "models"
_RELEASES = "https://github.com/ultralytics/assets/releases/download"
OFFICIAL_ONNX = {
    "yolo26n.onnx": f"{_RELEASES}/v8.4.0/yolo26n.onnx",
    "yolo26s.onnx": f"{_RELEASES}/v8.4.0/yolo26s.onnx",   # more accurate, ~3x slower on CPU
    "yolo26m.onnx": f"{_RELEASES}/v8.4.0/yolo26m.onnx",   # GPU recommended
    "yolo11n.onnx": f"{_RELEASES}/v8.3.0/yolo11n.onnx",   # default fallback
}
MAX_MODEL_BYTES = 200 * 1024 * 1024


class ModelFileError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "smart-vision-ai"})
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=timeout) as resp:
            if not resp.geturl().startswith("https://"):
                raise ModelFileError("refusing a model download that was redirected away from HTTPS")
            total = 0
            while chunk := resp.read(1 << 20):
                total += len(chunk)
                if total > MAX_MODEL_BYTES:
                    raise ModelFileError("model download exceeds the size limit")
                out.write(chunk)
        os.replace(tmp, dest)
    except Exception as exc:
        Path(tmp).unlink(missing_ok=True)
        if isinstance(exc, ModelFileError):
            raise
        raise ModelFileError(f"could not download {url}: {exc}") from exc


def resolve_model(name: str, expected_sha256: str = "", allow_download: bool = True) -> Path:
    """Return a local path for ``name``, downloading an official ONNX export if needed."""
    if not name:
        raise ModelFileError("no model configured")
    candidate = Path(name).expanduser()
    if candidate.is_file():
        path = candidate
    elif (MODELS_DIR / candidate.name).is_file() and candidate.name == name:
        path = MODELS_DIR / candidate.name
    elif name in OFFICIAL_ONNX and allow_download:
        path = MODELS_DIR / name
        _download(OFFICIAL_ONNX[name], path)
    elif name in OFFICIAL_ONNX:
        raise ModelFileError(f"{name} is not in {MODELS_DIR} and downloads are disabled")
    else:
        raise ModelFileError(
            f"model file {name!r} not found (looked in the working directory and {MODELS_DIR}). "
            f"Auto-download supports: {', '.join(sorted(OFFICIAL_ONNX))}"
        )
    digest = sha256_of(path)
    if expected_sha256 and digest.lower() != expected_sha256.strip().lower():
        raise ModelFileError(f"{path.name} SHA-256 {digest} does not match YOLO_MODEL_SHA256")
    log.info("Model file %s (sha256 %s)", path, digest)
    return path
