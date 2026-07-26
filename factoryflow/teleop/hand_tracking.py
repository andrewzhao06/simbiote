"""Hand-pose estimation.

Today (laptop): MediaPipe Hands -- real-time 2D+rough-3D landmarks, CPU only.
Tomorrow (GB10): WiLoR -- transformer/ViT 3D hand mesh recovery (MANO),
much higher finger-articulation fidelity, does its own detection end-to-end.

Both live behind get_hand_landmarks(frame) -> Optional[HandLandmarks], so
ik_bridge.py never has to know which backend produced the landmarks
(master doc Part 6.4: "keep both wired behind the same hand_tracking.py
interface and decide which demos better once you can measure real GB10
latency").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# MediaPipe's 21-landmark hand model; see
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
NUM_LANDMARKS = 21


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

    def get_hand_landmarks(self, frame: np.ndarray) -> Optional[HandLandmarks]:
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

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module-level convenience function matching the master-doc signature:
#   get_hand_landmarks(frame) -> HandLandmarks
# backed by a lazily-created default tracker.
# ---------------------------------------------------------------------------

_default_tracker: Optional[HandTracker] = None


def get_hand_landmarks(frame: np.ndarray) -> Optional[HandLandmarks]:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = HandTracker()
    return _default_tracker.get_hand_landmarks(frame)
