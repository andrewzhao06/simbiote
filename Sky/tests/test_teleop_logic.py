"""Smoke tests for the parts of Step 3 that don't need a camera, MediaPipe,
or pybullet installed -- just the schema and the analytic IK math. Run with:

    python -m pytest Sky/tests/test_teleop_logic.py -v

from the repo root.
"""

import numpy as np
import pytest

from simbiote import demo_logger
from simbiote.robot_iface.actions import GripperState, Pose, RobotAction, neutral_action
from simbiote.teleop.hand_tracking import (
    HandLandmarks,
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    THUMB_TIP,
    WRIST,
)
from simbiote.teleop.ik_bridge import ControlMode, IKBridge


def make_landmarks(wrist_xy=(0.5, 0.5), palm_scale=0.15, pinch_ratio=1.0) -> HandLandmarks:
    wx, wy = wrist_xy
    points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    points[WRIST] = (wx, wy, 0.0)
    points[MIDDLE_MCP] = (wx, wy - palm_scale, 0.0)
    points[THUMB_TIP] = (wx, wy, 0.0)
    points[INDEX_TIP] = (wx + pinch_ratio * palm_scale, wy, 0.0)
    return HandLandmarks(points=points, handedness="Right", confidence=0.95)


# --- robot_iface/actions.py -------------------------------------------------


def test_robot_action_roundtrip_through_dict():
    action = RobotAction(
        base_velocity=(0.1, -0.2, 0.3),
        arm_target_pose=Pose(position=(0.4, 0.0, 0.5)),
        gripper_state=GripperState.CLOSED,
    )
    restored = RobotAction.from_dict(action.to_dict())
    assert restored.base_velocity == action.base_velocity
    assert restored.arm_target_pose.position == action.arm_target_pose.position
    assert restored.gripper_state == GripperState.CLOSED


def test_neutral_action_is_stationary_and_open():
    action = neutral_action()
    assert action.base_velocity == (0.0, 0.0, 0.0)
    assert action.gripper_state == GripperState.OPEN
    assert action.arm_target_pose is not None  # a real "parked" pose, not "no command"


# --- demo_logger.py ----------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMBIOTE_STAGE", str(tmp_path / "stage"))
    demo_logger._active_sessions.clear()
    demo_logger._current_session_id = None
    yield
    demo_logger._active_sessions.clear()
    demo_logger._current_session_id = None


def test_demo_logger_logs_and_exports():
    session_id = demo_logger.start_session("test_session", source="teleop", task="nav")

    for _ in range(3):
        demo_logger.log_action(neutral_action(), source="teleop")

    trajectory = demo_logger.export_trajectory(session_id)
    assert trajectory.session_id == session_id
    assert len(trajectory.steps) == 3
    assert all(step.source == "teleop" for step in trajectory.steps)


def test_demo_logger_export_unknown_session_raises(tmp_path):
    with pytest.raises(KeyError):
        demo_logger.export_trajectory("does-not-exist", stage=tmp_path / "empty_stage")


# --- teleop/ik_bridge.py -----------------------------------------------------


def test_hand_centered_produces_near_zero_base_velocity():
    bridge = IKBridge()
    action = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.5, 0.5)))
    vx, vy, omega = action.base_velocity
    assert abs(vx) < 1e-6
    assert abs(omega) < 1e-6


def test_hand_offset_right_of_center_turns():
    bridge = IKBridge(ema_alpha=1.0)  # no smoothing lag, for a direct assertion
    action = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.9, 0.5)))
    _, _, omega = action.base_velocity
    assert omega != 0.0


def test_pinch_closes_gripper_and_open_hand_opens_it():
    bridge = IKBridge()
    closed = bridge.landmarks_to_action(make_landmarks(pinch_ratio=0.05))
    assert closed.gripper_state == GripperState.CLOSED

    bridge2 = IKBridge()
    opened = bridge2.landmarks_to_action(make_landmarks(pinch_ratio=1.0))
    assert opened.gripper_state == GripperState.OPEN


# --- pinch as a mode switch --------------------------------------------------


def test_pinch_switches_to_manipulate_and_parks_the_base():
    """Pinching stops the base: you're posing the arm, not driving."""

    bridge = IKBridge(ema_alpha=1.0)
    driving = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.9, 0.1)))
    assert bridge.mode == ControlMode.DRIVE
    assert driving.base_velocity != (0.0, 0.0, 0.0)

    # Same off-centre hand, now pinched -> base must stop regardless.
    pinched = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.9, 0.1), pinch_ratio=0.05))
    assert bridge.mode == ControlMode.MANIPULATE
    assert pinched.base_velocity == (0.0, 0.0, 0.0)
    assert pinched.gripper_state == GripperState.CLOSED


def test_hand_height_moves_the_claw_only_while_pinched():
    bridge = IKBridge(ema_alpha=1.0)

    # Driving: the arm target is held, so height changes must not move it.
    low_drive = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.5, 0.8)))
    high_drive = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.5, 0.2)))
    assert low_drive.arm_target_pose.position == high_drive.arm_target_pose.position

    # Pinched: hand up must raise the claw relative to hand down.
    low = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.5, 0.8), pinch_ratio=0.05))
    high = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.5, 0.2), pinch_ratio=0.05))
    assert high.arm_target_pose.position[2] > low.arm_target_pose.position[2]
    # ...and only the height changed.
    assert high.arm_target_pose.position[:2] == low.arm_target_pose.position[:2]


def test_mode_does_not_flicker_at_the_pinch_threshold():
    """Hysteresis: releasing needs a wider gap than grabbing did."""

    bridge = IKBridge(ema_alpha=1.0)
    bridge.landmarks_to_action(make_landmarks(pinch_ratio=0.05))
    assert bridge.mode == ControlMode.MANIPULATE

    # Just above the *enter* threshold is not enough to release.
    bridge.landmarks_to_action(make_landmarks(pinch_ratio=0.40))
    assert bridge.mode == ControlMode.MANIPULATE

    # Opening properly releases.
    bridge.landmarks_to_action(make_landmarks(pinch_ratio=0.9))
    assert bridge.mode == ControlMode.DRIVE


def test_tracking_dropout_mid_grasp_keeps_holding():
    """A glitch shouldn't open the gripper and drop what the robot is carrying."""

    bridge = IKBridge(ema_alpha=1.0)
    bridge.landmarks_to_action(make_landmarks(pinch_ratio=0.05))
    dropped = bridge.landmarks_to_action(None)

    assert bridge.mode == ControlMode.MANIPULATE
    assert dropped.gripper_state == GripperState.CLOSED
    assert dropped.base_velocity == (0.0, 0.0, 0.0)


def test_missing_hand_freezes_arm_but_stops_base():
    bridge = IKBridge(ema_alpha=1.0)
    first = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.7, 0.3)))
    frozen = bridge.landmarks_to_action(None)

    assert frozen.arm_target_pose.position == first.arm_target_pose.position
    assert frozen.base_velocity == (0.0, 0.0, 0.0)


def test_arm_target_stays_within_workspace_bounds():
    from simbiote.teleop.ik_bridge import (
        WORKSPACE_X_MAX,
        WORKSPACE_X_MIN,
        WORKSPACE_Y_MAX,
        WORKSPACE_Y_MIN,
        WORKSPACE_Z_MAX,
        WORKSPACE_Z_MIN,
    )

    bridge = IKBridge(ema_alpha=1.0)
    for wrist_xy in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.0, 1.0), (1.0, 0.0)]:
        action = bridge.landmarks_to_action(make_landmarks(wrist_xy=wrist_xy))
        x, y, z = action.arm_target_pose.position
        assert WORKSPACE_X_MIN - 1e-6 <= x <= WORKSPACE_X_MAX + 1e-6
        assert WORKSPACE_Y_MIN - 1e-6 <= y <= WORKSPACE_Y_MAX + 1e-6
        assert WORKSPACE_Z_MIN - 1e-6 <= z <= WORKSPACE_Z_MAX + 1e-6
