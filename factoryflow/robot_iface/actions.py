"""Shared robot-command schema.

Producers: Step 3 (teleop_session.py) and Step 4 (agentic_session.py).
Consumers: Step 2's robot/sim binding and demo_logger.py.

This is the one interface both control paths (hand-tracking teleop and
agentic/LLM-driven commands) speak, so keep it stable once teammates 3
and 4 have both signed off on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Tuple


class GripperState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class Pose:
    """A 6-DOF pose. position in meters, orientation as an (x, y, z, w) quaternion."""

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"position": list(self.position), "orientation": list(self.orientation)}

    @staticmethod
    def from_dict(d: dict) -> "Pose":
        return Pose(position=tuple(d["position"]), orientation=tuple(d["orientation"]))


@dataclass(frozen=True)
class RobotAction:
    """One command frame for the robot: base velocity + arm target + gripper.

    base_velocity: (vx, vy, omega) in the robot's base frame, m/s and rad/s.
    arm_target_pose: desired end-effector pose in the robot base frame.
    gripper_state: open|closed.
    timestamp: seconds since epoch; defaults to time of construction so
        callers don't have to remember to stamp every frame themselves.
    """

    base_velocity: Tuple[float, float, float]
    arm_target_pose: Pose
    gripper_state: GripperState
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "base_velocity": list(self.base_velocity),
            "arm_target_pose": self.arm_target_pose.to_dict(),
            "gripper_state": self.gripper_state.value,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "RobotAction":
        return RobotAction(
            base_velocity=tuple(d["base_velocity"]),
            arm_target_pose=Pose.from_dict(d["arm_target_pose"]),
            gripper_state=GripperState(d["gripper_state"]),
            timestamp=d["timestamp"],
        )


def neutral_action() -> RobotAction:
    """A safe zero-motion action: stationary base, arm at a fixed default pose, gripper open."""

    return RobotAction(
        base_velocity=(0.0, 0.0, 0.0),
        arm_target_pose=Pose(position=(0.4, 0.0, 0.4)),
        gripper_state=GripperState.OPEN,
    )
