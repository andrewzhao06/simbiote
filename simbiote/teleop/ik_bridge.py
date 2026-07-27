"""Hand-landmark -> RobotAction retargeting.

Today (laptop): a simple analytic mapping, no Jacobian/collision-aware
planning -- deliberately crude, per the spec ("Simple analytic/
Jacobian IK solver (CPU)"). It exists to prove the teleop chain end-to-end
tonight, not to be precise.
Tomorrow (GB10): swapped for cuMotion/cuRobo, collision-aware real-time
re-planning, same landmarks_to_action() signature so teleop_session.py
doesn't change.

One hand, two jobs -- so pinching switches between them (see `ControlMode`).
Trying to steer the base and pose the arm simultaneously with one hand makes
them fight: tilting to steer would drag the gripper along with it.

DRIVE mode (open hand):
  - Wrist offset from frame center -> base_velocity (vx from vertical
    offset, omega from horizontal offset), with a deadzone so small
    tracking jitter near center doesn't drive the base. vy (strafing) is
    left at 0.0 -- full omnidirectional mapping is a cuMotion refinement.
  - The arm target is *held*, so the gripper doesn't wander while driving.
  - Gripper open.

MANIPULATE mode (pinched):
  - Base velocity forced to zero: you're posing the arm, not driving.
  - Hand height -> arm end-effector height, over `MANIPULATE_Y_SPAN` of the
    frame. x/y hold, so engaging a pinch doesn't also swing the arm sideways.
  - Gripper closed.

Pinch is measured as thumb-tip/index-tip distance normalized by palm scale,
which makes it distance-invariant, and it has separate enter/exit thresholds
so the mode doesn't flicker when a pinch hovers near the boundary.

An EMA filter smooths both the arm target and the base velocity to damp
landmark jitter (spec Part 8 risk: "Hand-tracking latency/jitter").

When no hand is visible, the bridge freezes the last commanded arm pose
(don't yank the arm back to neutral) but zeroes base velocity (don't let
the robot keep driving blind). The mode is deliberately *not* reset on
dropout: a momentary tracking glitch mid-grasp shouldn't open the gripper
and drop whatever it is holding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from simbiote.robot_iface.actions import GripperState, Pose, RobotAction, neutral_action
from simbiote.teleop.hand_tracking import (
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

# Pinch is the mode switch, not just a gripper command (see ControlMode).
# Two thresholds, not one: a single threshold makes the mode flicker whenever
# the operator's pinch hovers near it, which reads as the robot twitching
# between driving and manipulating. Pinch below ENTER to grab, open past EXIT
# to let go.
PINCH_CLOSED_RATIO = 0.35  # thumb-index distance / palm scale, below this = pinched
PINCH_RELEASE_RATIO = 0.50  # must open past this to leave manipulate mode

# How much of the frame's height the hand sweeps to cover the arm's full
# vertical range while manipulating. Less than 1.0 so the operator doesn't have
# to reach the very top and bottom of frame, where tracking is least reliable.
MANIPULATE_Y_SPAN = 0.7

BASE_DEADZONE = 0.10  # fraction of half-frame from center with no base motion
BASE_FORWARD_GAIN = 0.6  # m/s at full vertical deflection
BASE_TURN_GAIN = 1.2  # rad/s at full horizontal deflection

EMA_ALPHA = 0.4  # higher = more responsive, lower = smoother/laggier


# Landmarks arrive as a numpy array, so every derived quantity is a numpy
# scalar unless we cast. RobotAction gets JSON-serialized by demo_logger, and
# json can't encode np.float32 -- so this is the boundary that casts back to
# plain Python floats, once, for everything downstream.
def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _remap(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if in_hi == in_lo:
        return float(out_lo)
    t = _clamp((value - in_lo) / (in_hi - in_lo), 0.0, 1.0)
    return float(out_lo + t * (out_hi - out_lo))


def _lerp(a: float, b: float, alpha: float) -> float:
    return float(a + alpha * (b - a))


class ControlMode(str, Enum):
    """Which part of the robot the hand is currently flying.

    One hand can't drive a base and pose an arm at the same time without the
    two fighting -- tilting the hand to steer would drag the gripper around
    with it. So pinching switches between them:

    * DRIVE (open hand) -- hand position steers the base; the arm holds
      whatever pose it had, so it doesn't wander while the robot moves.
    * MANIPULATE (pinched) -- the base is held still and hand height moves the
      claw up and down, with the pincer closed.

    The mode is not a new field on `RobotAction`: it's fully implied by the
    action itself (manipulate = zero base velocity + closed gripper + a moving
    arm target). `RobotAction` is the ratified cross-team schema that Suraj's
    training code consumes, so it isn't worth widening for a teleop-local
    concept the consumer can already read.
    """

    DRIVE = "drive"
    MANIPULATE = "manipulate"


@dataclass
class IKBridge:
    """Stateful analytic IK: holds the last commanded action for smoothing
    and for freezing the arm pose when hand tracking drops out.
    """

    ema_alpha: float = EMA_ALPHA
    mirror_x: bool = False  # flip left/right if the camera view is mirrored
    _last_action: RobotAction = None  # type: ignore[assignment]
    mode: ControlMode = ControlMode.DRIVE

    def __post_init__(self) -> None:
        if self._last_action is None:
            self._last_action = neutral_action()

    def reset(self) -> None:
        self._last_action = neutral_action()
        self.mode = ControlMode.DRIVE

    def _update_mode(self, pinch_ratio: float) -> ControlMode:
        """Pinch to enter manipulate mode, open past a wider gap to leave it."""

        if self.mode == ControlMode.DRIVE:
            if pinch_ratio < PINCH_CLOSED_RATIO:
                self.mode = ControlMode.MANIPULATE
        elif pinch_ratio > PINCH_RELEASE_RATIO:
            self.mode = ControlMode.DRIVE
        return self.mode

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

        pinch_dist = math.hypot(
            pts[THUMB_TIP][0] - pts[INDEX_TIP][0], pts[THUMB_TIP][1] - pts[INDEX_TIP][1]
        )
        pinch_ratio = pinch_dist / palm_scale if palm_scale > 1e-6 else 1.0
        mode = self._update_mode(pinch_ratio)

        prev_pose = self._last_action.arm_target_pose or Pose(
            position=(WORKSPACE_X_MIN, 0.0, WORKSPACE_Z_MIN)
        )
        prev_pos = prev_pose.position

        if mode == ControlMode.MANIPULATE:
            # Pinched: the base is parked and hand height flies the claw. Only
            # z tracks the hand -- x/y stay where they were, so a pinch doesn't
            # also swing the arm sideways the moment it engages.
            target_z = _remap(
                wrist_y,
                0.5 - MANIPULATE_Y_SPAN / 2.0,
                0.5 + MANIPULATE_Y_SPAN / 2.0,
                WORKSPACE_Z_MAX,
                WORKSPACE_Z_MIN,
            )  # image y grows downward, so hand up -> claw up
            smoothed_pos = (
                prev_pos[0],
                prev_pos[1],
                _lerp(prev_pos[2], target_z, self.ema_alpha),
            )
            action = RobotAction(
                base_velocity=(0.0, 0.0, 0.0),
                arm_target_pose=Pose(position=smoothed_pos),
                gripper_state=GripperState.CLOSED,
            )
            self._last_action = action
            return action

        # Open hand: steer the base, and hold the arm where the operator left
        # it so it doesn't drift while driving.
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
            arm_target_pose=prev_pose,
            gripper_state=GripperState.OPEN,
        )
        self._last_action = action
        return action


# ---------------------------------------------------------------------------
# Module-level convenience function matching the spec signature:
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
