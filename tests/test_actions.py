"""Verification 1 — the shared action schema round-trips losslessly."""

from __future__ import annotations

from factoryflow.robot_iface.actions import GripperState, Pose, RobotAction


def test_pose_round_trip():
    pose = Pose(1.5, -2.25, 0.8, 0.0, 0.0, 0.7071, 0.7071)
    assert Pose.from_dict(pose.to_dict()) == pose


def test_action_round_trip_with_arm_pose():
    action = RobotAction(
        base_velocity=(0.4, -0.1, 0.2),
        arm_target_pose=Pose(1.0, 2.0, 0.9),
        gripper_state=GripperState.CLOSED,
    )
    assert RobotAction.from_dict(action.to_dict()) == action


def test_action_round_trip_without_arm_pose():
    """A nav-only action must survive the trip with arm_target_pose still None —
    a fabricated pose here would corrupt Step 2's fine-tune data."""
    action = RobotAction(base_velocity=(0.6, 0.0, 0.0))
    restored = RobotAction.from_dict(action.to_dict())
    assert restored == action
    assert restored.arm_target_pose is None


def test_gripper_state_serializes_as_plain_string():
    assert RobotAction().to_dict()["gripper_state"] == "open"
