"""The "Simbiote Teleop" preview window.

Renders the camera feed with the tracked hand skeleton on the left and a
readout of the `RobotAction` that hand is producing on the right, so the
hand -> robot retargeting in `ik_bridge.py` is visible while you operate:
you can see the pinch close the gripper and the wrist position move the arm
target inside the workspace box, live, before any of it reaches a robot.

Pure OpenCV drawing -- no extra GUI dependency, and it degrades to a no-op
when there's no display attached (see `PreviewWindow.available`).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from simbiote.robot_iface.actions import GripperState, RobotAction
from simbiote.teleop.hand_tracking import HAND_CONNECTIONS, HandLandmarks, WRIST
from simbiote.teleop.ik_bridge import (
    BASE_DEADZONE,
    hand_pitch,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Z_MAX,
    WORKSPACE_Z_MIN,
)

WINDOW_NAME = "Simbiote Teleop"
PANEL_WIDTH = 340
MAX_WINDOW_WIDTH = 1180

# BGR.
COL_BG = (28, 26, 24)
COL_TEXT = (235, 235, 235)
COL_DIM = (140, 140, 140)
COL_BONE = (240, 190, 60)
COL_JOINT = (90, 220, 255)
COL_OK = (120, 230, 130)
COL_WARN = (80, 200, 250)
COL_ALERT = (90, 90, 250)
COL_ACCENT = (200, 130, 90)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _displayed_handedness(landmarks: HandLandmarks, mirrored: bool) -> str:
    """The hand the operator is actually holding up.

    When the feed is mirrored, the model sees the operator's right hand as a
    left one, so the raw label is backwards from the operator's point of view.
    """

    if not mirrored:
        return landmarks.handedness
    return {"Left": "Right", "Right": "Left"}.get(landmarks.handedness, landmarks.handedness)


def draw_landmarks(frame: np.ndarray, landmarks: HandLandmarks, mirrored: bool = False) -> None:
    """Draw the 21-point hand skeleton onto `frame` in place."""

    height, width = frame.shape[:2]
    pts = landmarks.points[:, :2] * np.array([width, height], dtype=np.float32)
    pts = pts.astype(np.int32)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), COL_BONE, 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(pts):
        radius = 6 if idx == WRIST else 4
        cv2.circle(frame, (int(x), int(y)), radius, COL_JOINT, -1, cv2.LINE_AA)

    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cv2.rectangle(frame, (int(x0) - 12, int(y0) - 12), (int(x1) + 12, int(y1) + 12), COL_ACCENT, 1)
    cv2.putText(
        frame,
        f"{_displayed_handedness(landmarks, mirrored)} hand",
        (int(x0) - 12, int(y0) - 18),
        FONT,
        0.5,
        COL_ACCENT,
        1,
        cv2.LINE_AA,
    )


def _draw_deadzone(frame: np.ndarray) -> None:
    """Show the base-velocity deadzone the wrist has to leave to drive the base."""

    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    half_w = int(BASE_DEADZONE * width)
    half_h = int(BASE_DEADZONE * height)
    cv2.rectangle(frame, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), COL_DIM, 1)
    cv2.drawMarker(frame, (cx, cy), COL_DIM, cv2.MARKER_CROSS, 14, 1)


def _bar(
    panel: np.ndarray,
    y: int,
    label: str,
    value: float,
    lo: float,
    hi: float,
    unit: str = "",
) -> int:
    """Draw one labelled horizontal gauge; returns the next free y."""

    cv2.putText(panel, label, (16, y), FONT, 0.45, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"{value:+.2f}{unit}",
        (PANEL_WIDTH - 96, y),
        FONT,
        0.45,
        COL_TEXT,
        1,
        cv2.LINE_AA,
    )

    track_y = y + 10
    x0, x1 = 16, PANEL_WIDTH - 16
    cv2.rectangle(panel, (x0, track_y), (x1, track_y + 8), (60, 58, 56), -1)

    span = hi - lo
    t = 0.0 if span == 0 else float(np.clip((value - lo) / span, 0.0, 1.0))
    fill = int(x0 + t * (x1 - x0))
    cv2.rectangle(panel, (x0, track_y), (fill, track_y + 8), COL_ACCENT, -1)

    # Mark zero when the range straddles it, so sign is readable at a glance.
    if lo < 0.0 < hi:
        zero = int(x0 + (-lo / span) * (x1 - x0))
        cv2.line(panel, (zero, track_y - 2), (zero, track_y + 10), COL_TEXT, 1)

    return track_y + 26


def _draw_workspace_map(panel: np.ndarray, y: int, action: RobotAction) -> int:
    """Top-down view of the arm target inside the workspace box (x forward, y left/right)."""

    pose = action.arm_target_pose
    box_h = 110
    x0, x1 = 16, PANEL_WIDTH - 16
    cv2.putText(panel, "arm target (top-down)", (16, y), FONT, 0.45, COL_DIM, 1, cv2.LINE_AA)
    top = y + 8
    cv2.rectangle(panel, (x0, top), (x1, top + box_h), (60, 58, 56), 1)

    if pose is not None:
        px, py, _ = pose.position
        # Panel x <- robot y (left/right), panel y <- robot x (forward, up = far).
        tx = (py - WORKSPACE_Y_MIN) / (WORKSPACE_Y_MAX - WORKSPACE_Y_MIN)
        ty = (px - WORKSPACE_X_MIN) / (WORKSPACE_X_MAX - WORKSPACE_X_MIN)
        # Inset by the marker radius so a target sitting on a workspace limit
        # still draws fully inside the box instead of bleeding over the edge.
        inset = 11
        cx = int(x0 + inset + np.clip(tx, 0, 1) * (x1 - x0 - 2 * inset))
        cy = int(top + box_h - inset - np.clip(ty, 0, 1) * (box_h - 2 * inset))
        colour = COL_ALERT if action.gripper_state == GripperState.CLOSED else COL_OK
        cv2.circle(panel, (cx, cy), 7, colour, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 11, colour, 1, cv2.LINE_AA)

    cv2.putText(panel, "near", (x0 + 4, top + box_h - 4), FONT, 0.35, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(panel, "far", (x0 + 4, top + 12), FONT, 0.35, COL_DIM, 1, cv2.LINE_AA)
    return top + box_h + 22


def render_panel(
    height: int,
    action: RobotAction,
    landmarks: Optional[HandLandmarks],
    fps: float,
    backend: str,
    sink: str,
) -> np.ndarray:
    """Build the right-hand status panel for one frame."""

    panel = np.full((height, PANEL_WIDTH, 3), COL_BG, dtype=np.uint8)

    cv2.putText(panel, "SIMBIOTE TELEOP", (16, 30), FONT, 0.6, COL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"{backend}  |  {fps:4.1f} fps  |  sink: {sink}",
        (16, 50),
        FONT,
        0.4,
        COL_DIM,
        1,
        cv2.LINE_AA,
    )
    cv2.line(panel, (16, 62), (PANEL_WIDTH - 16, 62), (60, 58, 56), 1)

    tracking = landmarks is not None
    cv2.circle(panel, (24, 82), 6, COL_OK if tracking else COL_ALERT, -1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "hand tracked" if tracking else "no hand - arm held",
        (40, 87),
        FONT,
        0.45,
        COL_TEXT if tracking else COL_WARN,
        1,
        cv2.LINE_AA,
    )

    # Which half of the robot the hand is currently flying. Without this the
    # operator can't tell why an off-centre hand isn't driving -- the answer
    # being that they're still pinched and posing the arm instead.
    manipulating = action.gripper_state == GripperState.CLOSED
    label = "MANIPULATE - claw" if manipulating else "DRIVE - base"
    colour = COL_ALERT if manipulating else COL_OK
    cv2.rectangle(panel, (16, 96), (PANEL_WIDTH - 16, 124), colour, -1)
    cv2.putText(panel, label, (26, 115), FONT, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

    y = 150
    pose = action.arm_target_pose
    if pose is not None:
        px, py, pz = pose.position
        y = _bar(panel, y, "arm x (reach)", px, WORKSPACE_X_MIN, WORKSPACE_X_MAX, " m")
        y = _bar(panel, y, "arm y (left/right)", py, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX, " m")
        y = _bar(panel, y, "arm z (height)", pz, WORKSPACE_Z_MIN, WORKSPACE_Z_MAX, " m")

    # Show the raw pitch alongside the velocity it produces: if the robot won't
    # reverse, this tells you instantly whether the gesture isn't registering
    # or the mapping is at fault.
    if landmarks is not None:
        y = _bar(panel, y, "hand pitch (fwd/back)", hand_pitch(landmarks.points), -1.0, 1.0)

    vx, _vy, omega = action.base_velocity
    y = _bar(panel, y, "base vx", vx, -0.6, 0.6, " m/s")
    y = _bar(panel, y, "base omega", omega, -1.2, 1.2, " rad/s")

    closed = action.gripper_state == GripperState.CLOSED
    cv2.putText(panel, "gripper", (16, y), FONT, 0.45, COL_DIM, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (PANEL_WIDTH - 116, y - 14), (PANEL_WIDTH - 16, y + 6),
                  COL_ALERT if closed else COL_OK, -1)
    cv2.putText(
        panel,
        "CLOSED" if closed else "OPEN",
        (PANEL_WIDTH - 104, y),
        FONT,
        0.5,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    y += 28

    y = _draw_workspace_map(panel, y, action)

    for line in [
        "open hand + off-centre -> drive base",
        "pinch -> park base, claw up/down",
        "q or ESC -> quit",
    ]:
        cv2.putText(panel, line, (16, y), FONT, 0.38, COL_DIM, 1, cv2.LINE_AA)
        y += 16

    return panel


class PreviewWindow:
    """The teleop preview window. `show()` returns False when the user quits."""

    def __init__(
        self,
        backend: str = "hand",
        sink: str = "console",
        enabled: bool = True,
        mirrored: bool = False,
    ):
        self.backend = backend
        self.sink = sink
        self.mirrored = mirrored
        self.available = enabled and self._display_available()
        self._created = False
        if enabled and not self.available:
            print(
                "[teleop] no display detected (DISPLAY/WAYLAND_DISPLAY unset) -- "
                "preview window disabled; teleop still runs headless."
            )

    @staticmethod
    def _display_available() -> bool:
        if os.environ.get("SIMBIOTE_FORCE_PREVIEW") == "1":
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def compose(
        self,
        frame: np.ndarray,
        landmarks: Optional[HandLandmarks],
        action: RobotAction,
        fps: float,
    ) -> np.ndarray:
        """Return the full window image (annotated feed + status panel)."""

        canvas = frame.copy()
        _draw_deadzone(canvas)
        if landmarks is not None:
            draw_landmarks(canvas, landmarks, mirrored=self.mirrored)
        panel = render_panel(canvas.shape[0], action, landmarks, fps, self.backend, self.sink)
        return np.hstack([canvas, panel])

    def show(
        self,
        frame: np.ndarray,
        landmarks: Optional[HandLandmarks],
        action: RobotAction,
        fps: float,
    ) -> bool:
        """Draw one frame. Returns False if the operator asked to quit."""

        if not self.available:
            return True

        canvas = self.compose(frame, landmarks, action, fps)

        if not self._created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # WINDOW_NORMAL opens at whatever size the WM feels like, which on
            # this box is small enough that the panel text is unreadable. Ask
            # for the composed image's own size, capped so a 720p feed plus the
            # panel still fits on a 1920-wide screen next to the Isaac window.
            height, width = canvas.shape[:2]
            scale = min(1.0, MAX_WINDOW_WIDTH / width)
            cv2.resizeWindow(WINDOW_NAME, int(width * scale), int(height * scale))
            self._created = True

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        if self._created:
            cv2.destroyAllWindows()
            self._created = False
