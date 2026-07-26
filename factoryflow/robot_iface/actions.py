"""The shared action schema — master doc Part 6.2 / 6b.2.

PROPOSAL, not yet ratified. Teammate 3 (hand-tracking teleop) and Teammate 4
(agentic control) both *produce* ``RobotAction``; Teammate 2's fine-tune loop
consumes it. See SCHEMAS_PROPOSAL.md at the repo root for the open questions.

Deliberately stdlib-only frozen dataclasses so Teammate 3 can adopt this
without inheriting any dependency from the agentic side.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["GripperState", "Pose", "RobotAction"]


class GripperState(str, Enum):
    """Binary gripper command.

    The reference robot's Franka Panda hand is a 2-finger parallel gripper, and
    both producers command it open/closed rather than by width — continuous
    width would be a schema change, not a value change, so it is called out in
    the proposal doc as an explicit open question.
    """

    OPEN = "open"
    CLOSED = "closed"

    def __str__(self) -> str:  # keeps f-strings readable in trace output
        return self.value


@dataclass(frozen=True)
class Pose:
    """A 6-DOF pose: position in metres plus an XYZW quaternion.

    Field order and quaternion convention match Stray Scanner's ``odometry.csv``
    (master doc 4.2: ``x, y, z, qx, qy, qz, qw``) so a pose keeps the same shape
    from the phone capture all the way through to a logged trajectory.
    """

    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pose":
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            z=float(d["z"]),
            qx=float(d.get("qx", 0.0)),
            qy=float(d.get("qy", 0.0)),
            qz=float(d.get("qz", 0.0)),
            qw=float(d.get("qw", 1.0)),
        )

    @classmethod
    def from_xyz(cls, xyz: tuple[float, float, float]) -> "Pose":
        return cls(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


@dataclass(frozen=True)
class RobotAction:
    """One control command for the Ridgeback base + Franka arm.

    ``base_velocity`` is ``(vx, vy, omega)`` — the base is omnidirectional
    (4 mecanum wheels) so ``vy`` is a real commandable axis, not padding.

    ``arm_target_pose`` is intentionally optional. A pure navigation step has no
    arm target, and synthesising a dummy pose to fill the field would put
    fabricated arm data into the trajectories Step 2 fine-tunes on.
    """

    base_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_target_pose: Pose | None = None
    gripper_state: GripperState = GripperState.OPEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_velocity": list(self.base_velocity),
            "arm_target_pose": (
                self.arm_target_pose.to_dict() if self.arm_target_pose else None
            ),
            "gripper_state": self.gripper_state.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RobotAction":
        """Adapter seam: if a teammate's counter-proposal changes the wire
        format, this method and :meth:`to_dict` are what change."""
        bv = d.get("base_velocity") or (0.0, 0.0, 0.0)
        pose = d.get("arm_target_pose")
        return cls(
            base_velocity=(float(bv[0]), float(bv[1]), float(bv[2])),
            arm_target_pose=Pose.from_dict(pose) if pose else None,
            gripper_state=GripperState(d.get("gripper_state", "open")),
        )
