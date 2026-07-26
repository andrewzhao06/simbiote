"""Shared `RobotAction` schema — see spec §6.2 / §6b.2.

Both Step 3 (hand-tracking teleop) and Step 4 (agentic control) are producers
of this schema; Step 2 (this file's owner for now) is the primary consumer —
`bc_pretrain.py` regresses onto `RobotAction`, and `nav_task.py` /
`grasp_task.py` speak a numpy-array version of the same fields internally.

Kept dependency-free (stdlib only: dataclasses + enum) so any of the four
role packages can import it without pulling in PyBullet/Isaac/torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class GripperState(str, Enum):
    """Binary gripper command. Matches spec: `gripper_state: open|closed`."""

    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class Pose:
    """A 6-DOF pose: position in meters, orientation as an xyzw quaternion."""

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def as_tuple(self) -> Tuple[float, ...]:
        return tuple(self.position) + tuple(self.orientation)

    @classmethod
    def from_tuple(cls, values: Tuple[float, ...]) -> "Pose":
        if len(values) != 7:
            raise ValueError(f"Pose.from_tuple expects 7 values, got {len(values)}")
        return cls(position=tuple(values[0:3]), orientation=tuple(values[3:7]))


@dataclass(frozen=True)
class RobotAction:
    """One frame of robot command — exactly the shape in spec §6.2/§6b.2:

        RobotAction(base_velocity: (vx, vy, omega),
                    arm_target_pose: Pose,
                    gripper_state: open|closed)
    """

    base_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, omega)
    arm_target_pose: Pose = field(default_factory=Pose)
    gripper_state: GripperState = GripperState.OPEN

    def to_vector(self) -> Tuple[float, ...]:
        """Flatten to a fixed-length numeric vector for BC/PPO nets.

        Layout: [vx, vy, omega, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]
        gripper is 0.0 = open, 1.0 = closed.
        """
        gripper = 1.0 if self.gripper_state == GripperState.CLOSED else 0.0
        return tuple(self.base_velocity) + self.arm_target_pose.as_tuple() + (gripper,)

    @classmethod
    def from_vector(cls, values: Tuple[float, ...]) -> "RobotAction":
        if len(values) != 11:
            raise ValueError(f"RobotAction.from_vector expects 11 values, got {len(values)}")
        base_velocity = tuple(values[0:3])
        pose = Pose.from_tuple(tuple(values[3:10]))
        gripper = GripperState.CLOSED if values[10] >= 0.5 else GripperState.OPEN
        return cls(base_velocity=base_velocity, arm_target_pose=pose, gripper_state=gripper)


ACTION_VECTOR_DIM = 11
