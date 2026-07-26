"""Smoke tests for the parts of Step 3 that don't need a camera, MediaPipe,
or pybullet installed -- just the schema and the analytic IK math. Run with:

    python -m pytest tests/test_teleop_logic.py -v

from the repo root.
"""

import numpy as np
import pytest

from factoryflow.demo_logger import DemoLogger
from factoryflow.robot_iface.actions import GripperState, Pose, RobotAction, neutral_action
from factoryflow.teleop.hand_tracking import (
    HandLandmarks,
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    THUMB_TIP,
    WRIST,
)
from factoryflow.teleop.ik_bridge import IKBridge


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
    assert restored.timestamp == action.timestamp


def test_neutral_action_is_stationary_and_open():
    action = neutral_action()
    assert action.base_velocity == (0.0, 0.0, 0.0)
    assert action.gripper_state == GripperState.OPEN


# --- demo_logger.py ----------------------------------------------------------


def test_demo_logger_logs_and_exports(tmp_path):
    logger = DemoLogger(log_dir=tmp_path)
    session_id = logger.start_session("test_session")

    for _ in range(3):
        logger.log_action(neutral_action(), source="teleop")

    trajectory = logger.export_trajectory(session_id)
    assert trajectory.session_id == session_id
    assert len(trajectory.steps) == 3
    assert all(step.source == "teleop" for step in trajectory.steps)


def test_demo_logger_save_writes_json(tmp_path):
    logger = DemoLogger(log_dir=tmp_path)
    session_id = logger.start_session()
    logger.log_action(neutral_action(), source="teleop")

    out_path = logger.save(session_id)
    assert out_path.exists()
    assert out_path.read_text().strip().startswith("{")


def test_demo_logger_export_unknown_session_raises(tmp_path):
    logger = DemoLogger(log_dir=tmp_path)
    with pytest.raises(KeyError):
        logger.export_trajectory("does-not-exist")


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


def test_missing_hand_freezes_arm_but_stops_base():
    bridge = IKBridge(ema_alpha=1.0)
    first = bridge.landmarks_to_action(make_landmarks(wrist_xy=(0.7, 0.3)))
    frozen = bridge.landmarks_to_action(None)

    assert frozen.arm_target_pose.position == first.arm_target_pose.position
    assert frozen.base_velocity == (0.0, 0.0, 0.0)


def test_arm_target_stays_within_workspace_bounds():
    from factoryflow.teleop.ik_bridge import (
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
