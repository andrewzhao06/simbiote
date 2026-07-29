"""Hand-pose estimation.

Laptop (x86): MediaPipe Hands -- real-time 2D+rough-3D landmarks, CPU only.
GB10 (aarch64): WiLoR -- transformer/ViT 3D hand mesh recovery (MANO), much
higher finger-articulation fidelity, does its own detection end-to-end.

Both live behind get_hand_landmarks(frame) -> Optional[HandLandmarks], so
ik_bridge.py never has to know which backend produced the landmarks (spec
§6.4: "keep both wired behind the same hand_tracking.py interface and decide
which demos better once you can measure real GB10 latency").

On the GB10 the choice is forced, not a preference: MediaPipe publishes no
linux-aarch64 wheel, so `create_tracker()` defaults to WiLoR there. Both
backends emit the same 21-landmark indexing (see the constants below), which
is the OpenPose/MediaPipe hand convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# MediaPipe's 21-landmark hand model; see
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
NUM_LANDMARKS = 21

# Bone connectivity for drawing the hand skeleton, shared by any preview UI.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # knuckle bridge
)


@dataclass
class HandLandmarks:
    """21 (x, y, z) landmarks, normalized to [0, 1] in x/y (image space) and
    a rough relative depth in z (MediaPipe convention: smaller/negative is
    closer to the camera). handedness is "Left" or "Right" as reported by
    the model; confidence is the model's own detection score.
    """

    points: np.ndarray  # shape (21, 3)
    handedness: str
    confidence: float


class HandTracker:
    """Wraps a MediaPipe Hands instance. Construct once, call repeatedly."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp  # imported lazily so the module is importable without it

        self._mp_hands_module = mp.solutions.hands
        self._hands = self._mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def get_hand_landmarks(self, frame: np.ndarray) -> HandLandmarks | None:
        """frame: a BGR image as read from camera_source.FrameSource.read().

        Returns None if no hand is detected in this frame.
        """

        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        hand_lms = result.multi_hand_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

        handedness = "Right"
        confidence = 1.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = classification.label
            confidence = float(classification.score)

        return HandLandmarks(points=points, handedness=handedness, confidence=confidence)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

BACKENDS = ("auto", "mediapipe", "wilor")


def _mediapipe_available() -> bool:
    """True only when the *legacy solutions* API this backend uses is present.

    `import mediapipe` succeeding is not enough. mediapipe 1.0 dropped the
    legacy solutions API entirely (only `mediapipe.tasks` ships), so a plain
    module check reports the backend as usable and the session then dies on
    `mp.solutions.hands` several steps later -- after the camera is already
    open. Probe the module `mp.solutions` actually aliases; `requirements.txt`
    pins `mediapipe<1` to match.
    """
    import importlib.util

    if importlib.util.find_spec("mediapipe") is None:
        return False
    try:
        return importlib.util.find_spec("mediapipe.python.solutions.hands") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def resolve_backend(backend: str = "auto") -> str:
    """Turn 'auto' into a concrete backend name for this machine.

    'auto' prefers MediaPipe when it is actually importable (cheap, CPU-only,
    good enough on a laptop) and otherwise falls back to WiLoR -- which is
    what happens on the GB10, where MediaPipe has no aarch64 wheel.
    """

    if backend not in BACKENDS:
        raise ValueError(f"unknown hand-tracking backend {backend!r}; expected one of {BACKENDS}")
    if backend != "auto":
        return backend
    return "mediapipe" if _mediapipe_available() else "wilor"


def create_tracker(backend: str = "auto", **kwargs):
    """Build a hand tracker exposing get_hand_landmarks(frame)/close().

    backend: 'auto' (default), 'mediapipe', or 'wilor'. Falls back to the
        SIMBIOTE_HAND_BACKEND env var when left at 'auto'.
    """

    if backend == "auto":
        backend = os.environ.get("SIMBIOTE_HAND_BACKEND", "auto")

    resolved = resolve_backend(backend)
    if resolved == "mediapipe":
        if not _mediapipe_available():
            raise RuntimeError(
                "hand-tracking backend 'mediapipe' was requested but a usable "
                "mediapipe is not installed: it is either absent, or version 1.0+, "
                "which removed the `mediapipe.solutions` API this backend uses. "
                "Install `mediapipe>=0.10,<1` (see requirements.txt). On "
                "linux-aarch64 (the GB10) upstream publishes no wheel at all -- "
                "use backend='wilor' instead."
            )
        return HandTracker(**kwargs)

    from simbiote.teleop.wilor_backend import WiLoRHandTracker

    return WiLoRHandTracker(**kwargs)


# ---------------------------------------------------------------------------
# Module-level convenience function matching the spec signature:
#   get_hand_landmarks(frame) -> HandLandmarks
# backed by a lazily-created default tracker.
# ---------------------------------------------------------------------------

_default_tracker = None


def get_hand_landmarks(frame: np.ndarray) -> HandLandmarks | None:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = create_tracker()
    return _default_tracker.get_hand_landmarks(frame)
