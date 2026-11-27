"""Logging configuration. Logs never contain frames, images or secrets - only events and errors."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(configure_logging, "_done", False):
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Ultralytics is chatty at INFO; keep its warnings and errors.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    configure_logging._done = True  # type: ignore[attr-defined]
