"""Safe decoding of camera frames received from the browser.

Only baseline/progressive JPEG is accepted. Image dimensions are read from the JPEG header
*before* decoding so a tiny file that claims to be 50 000 x 50 000 pixels (a decompression
bomb) is rejected without allocating memory for it.
"""
from __future__ import annotations

import cv2
import numpy as np

# SOF markers carrying frame dimensions (excludes DHT C4, JPG C8, DAC CC)
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class FrameError(ValueError):
    pass


def jpeg_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) from a JPEG header, or raise FrameError."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise FrameError("frame is not a JPEG image")
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            raise FrameError("corrupt JPEG header")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = (data[i + 2] << 8) | data[i + 3]
        if length < 2:
            raise FrameError("corrupt JPEG segment")
        if marker in _SOF:
            if i + 9 > n:
                break
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            if width == 0 or height == 0:
                raise FrameError("JPEG has zero dimensions")
            return width, height
        if marker == 0xDA:  # start of scan before any SOF
            break
        i += 2 + length
    raise FrameError("JPEG dimensions not found")


def decode_jpeg(data: bytes, max_bytes: int, max_dimension: int) -> np.ndarray:
    if len(data) > max_bytes:
        raise FrameError(f"frame is {len(data)} bytes; limit is {max_bytes}")
    width, height = jpeg_size(data)
    if width > max_dimension or height > max_dimension:
        raise FrameError(f"frame is {width}x{height}; limit is {max_dimension}px per side")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise FrameError("JPEG could not be decoded")
    return frame
