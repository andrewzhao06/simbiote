"""Shared `RobotAction` schema — see spec §6.2 / §6b.2.

Both Step 3 (hand-tracking teleop) and Step 4 (agentic control) are producers
of this schema; Step 2 (this file's owner for now) is the primary consumer —
`bc_pretrain.py` regresses onto `RobotAction`, and `nav_task.py` /
`grasp_task.py` speak a numpy-array version of the same fields internally.

This is the RATIFIED merge of the three teammates' independent proposals
(Teammate 2/Suraj, Teammate 3/Sky, Teammate 4/Andrew all wrote a version of
this file before the schema was a group decision). What's kept from each:

- Tuple-based `Pose(position, orientation)` and `RobotAction(base_velocity,
  arm_target_pose, gripper_state)` field shapes -- Suraj's and Sky's versions
  agreed on this independently; it's also what `to_vector()`/`from_vector()`
  (the fixed-length numeric encoding every training script depends on) needs.
- `arm_target_pose: Pose | None` -- Andrew's proposal. A pure navigation
  action has no arm command, and synthesising a fake pose to fill the field
  would put fabricated arm data into the trajectories `bc_pretrain.py`
  fine-tunes on. `to_vector()` still needs a fixed-length encoding, so it
  substitutes a neutral placeholder pose *only* for that numeric view --
  the wire/dict form (what actually gets logged and trained on) keeps `None`.
- `to_dict()`/`from_dict()` directly on `Pose`/`RobotAction`, and
  `neutral_action()` -- Sky's and Andrew's addition; several call sites
  (teleop, agentic, demo_logger) call these directly rather than going
  through `TrajectoryStep`.
- `x`/`y`/`z`/`qx`/`qy`/`qz`/`qw` read-only properties on `Pose` -- Andrew's
  scene-graph code (poses loaded from JSON with `x`/`y`/`z`/quaternion keys)
  reads poses this way; kept as a compatibility view over the same
  `position`/`orientation` tuples rather than a second storage format.

Kept dependency-free (stdlib only: dataclasses + enum) so any of the four
role packages can import it without pulling in PyBullet/Isaac/torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple


class GripperState(str, Enum):
    """Binary gripper command. Matches spec: `gripper_state: open|closed`."""

    OPEN = "open"
    CLOSED = "closed"

    def __str__(self) -> str:  # keeps f-strings/log lines readable
        return self.value


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

    @classmethod
    def from_xyz(cls, xyz: Tuple[float, float, float]) -> "Pose":
        """Convenience constructor for a position-only pose (identity orientation)."""
        return cls(position=(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def to_dict(self) -> dict:
        return {"position": list(self.position), "orientation": list(self.orientation)}

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        """Accepts either the canonical nested form (`position`/`orientation`
        lists) or the flat `x, y, z, qx, qy, qz, qw` form some scene-graph
        fixtures use -- both were in circulation before this schema merged."""
        if "position" in d:
            return cls(
                position=tuple(d["position"]),
                orientation=tuple(d.get("orientation", (0.0, 0.0, 0.0, 1.0))),
            )
        return cls(
            position=(float(d["x"]), float(d["y"]), float(d["z"])),
            orientation=(
                float(d.get("qx", 0.0)),
                float(d.get("qy", 0.0)),
                float(d.get("qz", 0.0)),
                float(d.get("qw", 1.0)),
            ),
        )

    # ---- flat-attribute compatibility view --------------------------------
    # Read-only; storage stays the tuple form above so `to_vector()`/tests/
    # trajectory serialization only ever deal with one representation.

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @property
    def z(self) -> float:
        return self.position[2]

    @property
    def qx(self) -> float:
        return self.orientation[0]

    @property
    def qy(self) -> float:
        return self.orientation[1]

    @property
    def qz(self) -> float:
        return self.orientation[2]

    @property
    def qw(self) -> float:
        return self.orientation[3]


@dataclass(frozen=True)
class RobotAction:
    """One frame of robot command — exactly the shape in spec §6.2/§6b.2:

        RobotAction(base_velocity: (vx, vy, omega),
                    arm_target_pose: Pose | None,
                    gripper_state: open|closed)

    `arm_target_pose` is `None` for actions that don't command the arm (e.g.
    pure navigation) rather than a synthesised default -- see module docstring.
    """

    base_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, omega)
    arm_target_pose: Optional[Pose] = None
    gripper_state: GripperState = GripperState.OPEN

    def to_vector(self) -> Tuple[float, ...]:
        """Flatten to a fixed-length numeric vector for BC/PPO nets.

        Layout: [vx, vy, omega, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]
        gripper is 0.0 = open, 1.0 = closed. When `arm_target_pose` is `None`
        (no arm command this frame), a neutral placeholder pose fills the
        numeric slots -- this view is only for fixed-shape network I/O, never
        serialized back out, so it can't leak a fabricated arm target into a
        logged trajectory.
        """
        gripper = 1.0 if self.gripper_state == GripperState.CLOSED else 0.0
        pose = self.arm_target_pose if self.arm_target_pose is not None else Pose()
        return tuple(self.base_velocity) + pose.as_tuple() + (gripper,)

    @classmethod
    def from_vector(cls, values: Tuple[float, ...]) -> "RobotAction":
        if len(values) != 11:
            raise ValueError(f"RobotAction.from_vector expects 11 values, got {len(values)}")
        base_velocity = tuple(values[0:3])
        pose = Pose.from_tuple(tuple(values[3:10]))
        gripper = GripperState.CLOSED if values[10] >= 0.5 else GripperState.OPEN
        return cls(base_velocity=base_velocity, arm_target_pose=pose, gripper_state=gripper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_velocity": list(self.base_velocity),
            "arm_target_pose": self.arm_target_pose.to_dict() if self.arm_target_pose is not None else None,
            "gripper_state": self.gripper_state.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RobotAction":
        pose = d.get("arm_target_pose")
        bv = d.get("base_velocity") or (0.0, 0.0, 0.0)
        return cls(
            base_velocity=(float(bv[0]), float(bv[1]), float(bv[2])),
            arm_target_pose=Pose.from_dict(pose) if pose else None,
            gripper_state=GripperState(d.get("gripper_state", "open")),
        )


ACTION_VECTOR_DIM = 11


def neutral_action() -> RobotAction:
    """A safe "parked" action: stationary base, arm at a fixed default pose,
    gripper open. Distinct from `arm_target_pose=None` (no arm command) --
    this is the bootstrap/reset value teleop's IK bridge smooths *from* before
    any hand has been tracked, so it needs a real pose to interpolate against.
    """
    return RobotAction(
        base_velocity=(0.0, 0.0, 0.0),
        arm_target_pose=Pose(position=(0.4, 0.0, 0.4)),
        gripper_state=GripperState.OPEN,
    )
