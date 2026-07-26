"""Hand-landmark -> RobotAction retargeting.

Today (laptop): a simple analytic mapping, no Jacobian/collision-aware
planning -- deliberately crude, per the master doc ("Simple analytic/
Jacobian IK solver (CPU)"). It exists to prove the teleop chain end-to-end
tonight, not to be precise.
Tomorrow (GB10): swapped for cuMotion/cuRobo, collision-aware real-time
re-planning, same landmarks_to_action() signature so teleop_session.py
doesn't change.

Mapping used tonight:
  - Wrist (x, y) in image space -> arm end-effector (y, z) in a fixed
    workspace box in front of the robot (left-right, up-down).
  - Palm scale (wrist-to-middle-MCP distance) as a depth proxy -> arm
    forward reach (x). MediaPipe's own z is too noisy to trust directly;
    apparent hand size is a more stable stand-in. Calibrate the
    PALM_SCALE_NEAR/FAR constants empirically against the real venue
    camera, the same way Step 1 calibrates its depth mm->m factor rather
    than trusting a nominal value blindly.
  - Thumb-tip/index-tip pinch distance, normalized by palm scale (so it's
    distance-invariant) -> gripper open/closed.
  - Wrist offset from frame center -> base_velocity (vx from vertical
    offset, omega from horizontal offset), with a deadzone so small
    tracking jitter near center doesn't drive the base. vy (strafing) is
    left at 0.0 tonight -- full omnidirectional teleop mapping is a
    tomorrow/cuMotion refinement, not a laptop-tonight one.
  - An EMA filter smooths both the arm target and the base velocity to
    damp landmark jitter (Part 8 risk: "Hand-tracking latency/jitter").

When no hand is visible, the bridge freezes the last commanded arm pose
(don't yank the arm back to neutral) but zeroes base velocity (don't let
the robot keep driving blind).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from factoryflow.robot_iface.actions import GripperState, Pose, RobotAction, neutral_action
from factoryflow.teleop.hand_tracking import (
    HandLandmarks,
    INDEX_TIP,
    MIDDLE_MCP,
    THUMB_TIP,
    WRIST,
)

# Workspace box the wrist position is mapped into, meters, robot base frame.
WORKSPACE_X_MIN, WORKSPACE_X_MAX = 0.25, 0.55  # forward reach (from palm scale)
WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = -0.30, 0.30  # left/right (from image x)
WORKSPACE_Z_MIN, WORKSPACE_Z_MAX = 0.15, 0.65  # up/down (from image y)

# Palm-scale (normalized wrist-to-middle-MCP distance) calibration.
# Smaller apparent hand (farther from camera) -> PALM_SCALE_FAR -> reach far (WORKSPACE_X_MAX).
# Larger apparent hand (closer to camera)     -> PALM_SCALE_NEAR -> reach near (WORKSPACE_X_MIN).
# Re-calibrate against the real teleop camera/lighting before relying on it.
PALM_SCALE_FAR = 0.08
PALM_SCALE_NEAR = 0.22

PINCH_CLOSED_RATIO = 0.35  # thumb-index distance / palm scale, below this = closed

BASE_DEADZONE = 0.10  # fraction of half-frame from center with no base motion
BASE_FORWARD_GAIN = 0.6  # m/s at full vertical deflection
BASE_TURN_GAIN = 1.2  # rad/s at full horizontal deflection

EMA_ALPHA = 0.4  # higher = more responsive, lower = smoother/laggier


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _remap(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if in_hi == in_lo:
        return out_lo
    t = _clamp((value - in_lo) / (in_hi - in_lo), 0.0, 1.0)
    return out_lo + t * (out_hi - out_lo)


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


@dataclass
class IKBridge:
    """Stateful analytic IK: holds the last commanded action for smoothing
    and for freezing the arm pose when hand tracking drops out.
    """

    ema_alpha: float = EMA_ALPHA
    mirror_x: bool = False  # flip left/right if the camera view is mirrored
    _last_action: RobotAction = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._last_action is None:
            self._last_action = neutral_action()

    def reset(self) -> None:
        self._last_action = neutral_action()

    def landmarks_to_action(self, landmarks: Optional[HandLandmarks]) -> RobotAction:
        if landmarks is None:
            frozen = self._last_action
            action = RobotAction(
                base_velocity=(0.0, 0.0, 0.0),
                arm_target_pose=frozen.arm_target_pose,
                gripper_state=frozen.gripper_state,
            )
            self._last_action = action
            return action

        pts = landmarks.points
        wrist_x, wrist_y, _ = pts[WRIST]
        if self.mirror_x:
            wrist_x = 1.0 - wrist_x

        palm_dx = pts[MIDDLE_MCP][0] - pts[WRIST][0]
        palm_dy = pts[MIDDLE_MCP][1] - pts[WRIST][1]
        palm_scale = math.hypot(palm_dx, palm_dy)

        target_x = _remap(palm_scale, PALM_SCALE_NEAR, PALM_SCALE_FAR, WORKSPACE_X_MIN, WORKSPACE_X_MAX)
        target_y = _remap(wrist_x, 0.0, 1.0, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
        target_z = _remap(wrist_y, 0.0, 1.0, WORKSPACE_Z_MAX, WORKSPACE_Z_MIN)  # image y grows downward

        prev_pos = self._last_action.arm_target_pose.position
        smoothed_pos = (
            _lerp(prev_pos[0], target_x, self.ema_alpha),
            _lerp(prev_pos[1], target_y, self.ema_alpha),
            _lerp(prev_pos[2], target_z, self.ema_alpha),
        )

        pinch_dist = math.hypot(
            pts[THUMB_TIP][0] - pts[INDEX_TIP][0], pts[THUMB_TIP][1] - pts[INDEX_TIP][1]
        )
        pinch_ratio = pinch_dist / palm_scale if palm_scale > 1e-6 else 1.0
        gripper_state = GripperState.CLOSED if pinch_ratio < PINCH_CLOSED_RATIO else GripperState.OPEN

        dx = wrist_x - 0.5
        dy = wrist_y - 0.5
        dx = 0.0 if abs(dx) < BASE_DEADZONE else dx
        dy = 0.0 if abs(dy) < BASE_DEADZONE else dy

        target_vx = -dy * 2.0 * BASE_FORWARD_GAIN
        target_omega = -dx * 2.0 * BASE_TURN_GAIN

        prev_vx, _prev_vy, prev_omega = self._last_action.base_velocity
        smoothed_vx = _lerp(prev_vx, target_vx, self.ema_alpha)
        smoothed_omega = _lerp(prev_omega, target_omega, self.ema_alpha)

        action = RobotAction(
            base_velocity=(smoothed_vx, 0.0, smoothed_omega),
            arm_target_pose=Pose(position=smoothed_pos),
            gripper_state=gripper_state,
        )
        self._last_action = action
        return action


# ---------------------------------------------------------------------------
# Module-level convenience function matching the master-doc signature:
#   landmarks_to_action(landmarks) -> RobotAction
# backed by a default stateful bridge (so smoothing/freeze behavior works
# out of the box for simple callers).
# ---------------------------------------------------------------------------

_default_bridge: Optional[IKBridge] = None


def landmarks_to_action(landmarks: Optional[HandLandmarks]) -> RobotAction:
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = IKBridge()
    return _default_bridge.landmarks_to_action(landmarks)
