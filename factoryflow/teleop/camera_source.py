"""Camera acquisition for teleop.

Confirmed hardware path (master doc, Part 6.3): Iriun Webcam turns the
teleop iPhone into a standard OS-level virtual webcam, so acquisition is
just cv2.VideoCapture against whichever device index Iriun registered as
-- nothing custom to build for the capture itself. USB is preferred over
Wi-Fi (~1-5 ms latency, no Wi-Fi dependency).

OpenCV does not expose camera *names* on most backends, so pick the Iriun
device by index. list_camera_indices() helps you find it once; after that,
pass the same index in via --camera-index or FACTORYFLOW_CAMERA_INDEX.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


def _default_backend() -> int:
    # CAP_DSHOW avoids a multi-second open delay for many virtual cameras
    # (including Iriun) on Windows; other platforms use OpenCV's default.
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


@dataclass
class FrameSource:
    """A thin, closeable wrapper around a cv2.VideoCapture."""

    cap: cv2.VideoCapture
    width: int
    height: int
    fps: int
    device_index: int

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def list_camera_indices(max_index: int = 8) -> list[int]:
    """Probe indices 0..max_index-1 and return the ones that open successfully.

    Run this once at setup time to find which index Iriun Webcam landed on
    (it varies by machine/driver order), then reuse that index.
    """

    backend = _default_backend()
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(idx)
        cap.release()
    return found


def open_camera(
    device_index: Optional[int] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
) -> FrameSource:
    """Open the teleop camera feed.

    device_index: which OS camera index to open. Falls back to the
        FACTORYFLOW_CAMERA_INDEX env var, then 0. Use list_camera_indices()
        to discover which index is "Iriun Webcam" on your machine.
    """

    if device_index is None:
        device_index = int(os.environ.get("FACTORYFLOW_CAMERA_INDEX", "0"))

    cap = cv2.VideoCapture(device_index, _default_backend())
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open camera at index {device_index}. "
            "Confirm 'Iriun Webcam' is running and connected (USB preferred), "
            "then use list_camera_indices() to find the right index."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or fps

    return FrameSource(
        cap=cap,
        width=actual_width,
        height=actual_height,
        fps=actual_fps,
        device_index=device_index,
    )
