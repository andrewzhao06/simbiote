"""Camera acquisition for teleop.

Two ways to get the teleop iPhone's camera into OpenCV:

1. **As a V4L2 device** (`/dev/videoN`) -- what Iriun Webcam does. A phone-webcam
   client running on the host publishes frames into a `v4l2loopback` device, and
   acquisition is just `cv2.VideoCapture(index)`. Nothing custom to build.

   Caveat on the GB10: Iriun's Linux client ships **x86-64 only** (the .deb is
   marked `Architecture: all` but `/usr/local/bin/iriunwebcam` is an x86-64 ELF),
   so it cannot run on this aarch64 box. DroidCam's Linux client has native
   arm64 builds and an iOS app, and works the same way. Either one needs the
   `v4l2loopback` kernel module, which needs root to install.

2. **As a network stream** (`http://…/video`, `rtsp://…`) -- most phone-webcam
   apps, DroidCam included, also serve MJPEG/RTSP straight off the phone.
   OpenCV opens those URLs directly, so this path needs **no kernel module and
   no root at all**. On a box where you can't `sudo`, this is the one that works.

`open_camera()` takes either, so the rest of the teleop chain doesn't care
which you used. See docs/TELEOP_IPHONE_CAMERA.md for the setup walkthrough.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

# A camera source is either an OS device index or a stream URL / device path.
CameraSource = Union[int, str]

_URL_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


def _default_backend() -> int:
    # CAP_DSHOW avoids a multi-second open delay for many virtual cameras
    # (including Iriun) on Windows; other platforms use OpenCV's default.
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def _is_stream(source: CameraSource) -> bool:
    return isinstance(source, str) and bool(_URL_RE.match(source))


def _coerce_source(source: CameraSource) -> CameraSource:
    """Accept '2' as index 2, but leave URLs and /dev/video paths as strings."""

    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class FrameSource:
    """A thin, closeable wrapper around a cv2.VideoCapture.

    `rotate` (0/90/180/270, clockwise) is applied to every frame on the way
    out. A phone held in portrait streams a landscape image with the scene
    turned on its side, and teleop cares about which way is up: `ik_bridge`
    maps image-y to the arm's height axis, so an uncorrected 90 degree feed
    silently swaps the operator's up/down and left/right.
    """

    cap: cv2.VideoCapture
    width: int
    height: int
    fps: int
    source: CameraSource
    rotate: int = 0

    @property
    def device_index(self) -> Optional[int]:
        """Backwards-compatible accessor; None when the source is a stream."""

        return self.source if isinstance(self.source, int) else None

    def _orient(self, frame: np.ndarray) -> np.ndarray:
        code = _ROTATIONS.get(self.rotate)
        return frame if code is None else cv2.rotate(frame, code)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return self._orient(frame)

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class ThreadedFrameSource:
    """Decouples capture from processing, and always hands back the newest frame.

    Reading straight from a network stream is blocking, so a serial loop costs
    `camera_time + inference_time` per iteration -- over Wi-Fi that measured
    ~6 FPS even though the camera alone managed 18 and the model 11.5. A
    reader thread overlaps the two, so the loop runs at whichever is slower
    instead of their sum.

    It also *drops* frames the consumer was too slow to take. That's what you
    want for teleop: a stale frame is worse than no frame, because the arm
    would be chasing where the operator's hand used to be.
    """

    def __init__(self, base: FrameSource):
        self._base = base
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    # Mirror FrameSource's attributes so callers can't tell the difference.
    @property
    def width(self) -> int:
        return self._base.width

    @property
    def height(self) -> int:
        return self._base.height

    @property
    def fps(self) -> int:
        return self._base.fps

    @property
    def source(self) -> CameraSource:
        return self._base.source

    @property
    def device_index(self) -> Optional[int]:
        return self._base.device_index

    def _pump(self) -> None:
        while not self._stopped.is_set():
            frame = self._base.read()
            if frame is None:
                self._stopped.set()
                break
            with self._lock:
                self._latest = frame

    def read(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Return the most recent frame, waiting for the first one to land."""

        deadline = time.time() + timeout
        while True:
            with self._lock:
                frame = self._latest
                self._latest = None  # so the caller can tell fresh from repeat
            if frame is not None:
                return frame
            if self._stopped.is_set() or time.time() > deadline:
                return None
            time.sleep(0.001)

    def release(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=1.0)
        self._base.release()

    def __enter__(self) -> "ThreadedFrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def list_video_devices() -> list[str]:
    """Return the /dev/video* nodes that exist (Linux only, no probing)."""

    return sorted(glob.glob("/dev/video*"))


def list_camera_indices(max_index: int = 8) -> list[int]:
    """Probe indices 0..max_index-1 and return the ones that open successfully.

    Run this once at setup time to find which index the phone-webcam client
    landed on (it varies by machine/driver order), then reuse that index.
    OpenCV does not expose camera *names* on most backends, so this is how you
    identify the device.
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


def resolve_source(
    source: Optional[CameraSource] = None,
    device_index: Optional[int] = None,
) -> CameraSource:
    """Work out which camera to open.

    Precedence: explicit `source` > explicit `device_index` >
    $SIMBIOTE_CAMERA_URL > $SIMBIOTE_CAMERA_INDEX > 0.
    """

    if source is not None:
        return _coerce_source(source)
    if device_index is not None:
        return device_index

    url = os.environ.get("SIMBIOTE_CAMERA_URL")
    if url:
        return url
    return int(os.environ.get("SIMBIOTE_CAMERA_INDEX", "0"))


def _open_failure_message(source: CameraSource) -> str:
    if _is_stream(source):
        return (
            f"could not open camera stream {source!r}.\n"
            "  - Check the phone and the GB10 are on the same network and the URL is "
            "reachable (try: curl -I <url>).\n"
            "  - Confirm the phone-webcam app is running and streaming.\n"
            "  See docs/TELEOP_IPHONE_CAMERA.md."
        )

    devices = list_video_devices()
    if not devices:
        return (
            f"could not open camera at index {source}: no /dev/video* devices exist "
            "on this machine at all.\n"
            "  - No v4l2loopback module is loaded, so no phone-webcam client has "
            "published a device.\n"
            "  - On this aarch64 box the Iriun Linux client will not run (it is an "
            "x86-64 binary).\n"
            "  - Easiest fix, no root required: stream from the phone and pass the "
            "URL instead, e.g. --camera-url http://<phone-ip>:4747/video\n"
            "  See docs/TELEOP_IPHONE_CAMERA.md."
        )

    return (
        f"could not open camera at index {source}. Existing devices: "
        f"{', '.join(devices)}. Use list_camera_indices() to find the right index, "
        "then pass it via --camera-index or SIMBIOTE_CAMERA_INDEX."
    )


def open_camera(
    device_index: Optional[int] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    source: Optional[CameraSource] = None,
    rotate: int = 0,
    threaded: bool = False,
    open_retries: int = 5,
    open_retry_delay: float = 2.0,
) -> Union[FrameSource, ThreadedFrameSource]:
    """Open the teleop camera feed.

    source: an OS camera index (int), a `/dev/videoN` path, or a stream URL
        such as `http://192.168.1.50:4747/video`. Falls back to
        $SIMBIOTE_CAMERA_URL, then $SIMBIOTE_CAMERA_INDEX, then index 0.
    device_index: legacy alias for an integer `source`.
    rotate: 0/90/180/270 clockwise, to square up a phone held in portrait.
        Falls back to $SIMBIOTE_CAMERA_ROTATE.
    threaded: read frames on a background thread and always return the newest.
        Worth it for live sources; leave off for video files, where dropping
        frames loses data rather than staleness.
    open_retries: how many times to retry a failed open before giving up.
        Phone webcam servers accept one client at a time and take a moment to
        release the slot after the previous one disconnects.
    """

    resolved = resolve_source(source=source, device_index=device_index)

    # Phone webcam servers (DroidCam included) accept exactly one client at a
    # time, and they don't free the slot the instant the previous one goes
    # away. So a session started right after a probe, or right after a
    # restart, hits a busy server and used to die on the spot. Retry briefly
    # rather than making the operator re-run the command.
    last_error: Optional[BaseException] = None
    for attempt in range(open_retries + 1):
        try:
            if _is_stream(resolved):
                cap = cv2.VideoCapture(resolved)
            else:
                cap = cv2.VideoCapture(resolved, _default_backend())
        except cv2.error as exc:  # pragma: no cover - backend dependent
            cap, last_error = None, exc

        if cap is not None and cap.isOpened():
            break
        if cap is not None:
            cap.release()
        if attempt < open_retries:
            print(
                f"[camera] {resolved} not ready (attempt {attempt + 1}/{open_retries + 1}), "
                f"retrying in {open_retry_delay:.0f}s"
            )
            time.sleep(open_retry_delay)
    else:
        raise RuntimeError(_open_failure_message(resolved)) from last_error

    if not _is_stream(resolved):
        # Stream servers dictate their own resolution; only ask a local device.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    # Keep latency down: without this OpenCV buffers frames and teleop drifts
    # further behind the operator's hand the longer the session runs.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:  # pragma: no cover - backend dependent
        pass

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or fps

    rotate = int(os.environ.get("SIMBIOTE_CAMERA_ROTATE", "0")) if not rotate else rotate
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be one of 0/90/180/270, got {rotate!r}")
    if rotate in (90, 270):
        actual_width, actual_height = actual_height, actual_width

    frames = FrameSource(
        cap=cap,
        width=actual_width,
        height=actual_height,
        fps=actual_fps,
        source=resolved,
        rotate=rotate,
    )
    return ThreadedFrameSource(frames) if threaded else frames
