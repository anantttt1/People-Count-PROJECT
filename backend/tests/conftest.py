"""Shared fixtures: an app instance with its own SQLite file and configurable settings."""
from __future__ import annotations

import time
from dataclasses import replace

import pytest

from backend.app.config import Settings
from backend.app.vision.detector import DetectorInfo, DetectorUnavailable

BASE_URL = "http://localhost:8000"
ORIGIN = "http://localhost:8000"
TOKEN = "correct-horse-battery-staple"


def no_detector(_settings):
    raise DetectorUnavailable("detector intentionally disabled for API tests")


def onnx_loader(_settings):
    """Real YOLO26n (ONNX) - same code path as production, without downloads."""
    from backend.app.vision.detector import OnnxYoloDetector
    from backend.tests import assets

    path = assets.model_path("yolo26n.onnx")
    if path is None:
        raise DetectorUnavailable("yolo26n.onnx not available")
    det = OnnxYoloDetector(path)
    return det, DetectorInfo(status="ready", name=det.name, model="yolo26n.onnx", device=det.device)


@pytest.fixture
def make_settings(tmp_path):
    def _make(**overrides) -> Settings:
        base = Settings(sqlite_path=tmp_path / "test.db", secret_key="s" * 64, allow_model_download=False)
        return replace(base, **overrides)
    return _make


@pytest.fixture
def make_client(make_settings):
    """Yields a factory; every client is closed (lifespan shutdown) at the end of the test."""
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    clients = []

    def _make(loader=no_detector, wait_for_detector=True, **overrides):
        app = create_app(settings=make_settings(**overrides), detector_loader=loader)
        client = TestClient(app, base_url=BASE_URL)
        client.__enter__()
        clients.append(client)
        state = app.state.app_state
        if wait_for_detector:
            deadline = time.monotonic() + 60
            while state.detector_info.status == "loading" and time.monotonic() < deadline:
                time.sleep(0.05)
        return client, state

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def wait_until(predicate, timeout=10.0, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()
