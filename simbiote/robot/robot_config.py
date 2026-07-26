"""Robot configuration — joint names, action limits, spawn pose, sensor mounts.

Per spec §5.4: "same config shape whether it's pointing at a PyBullet URDF
today or ridgeback_franka.usd tomorrow." `RobotConfig` is engine-agnostic;
`sim_env/*_task.py` reads it to build action/observation spaces and to know
which joints/links to drive, without hardcoding names inline.

Swapping engines tomorrow is a one-line change: pass `RIDGEBACK_FRANKA_CONFIG`
instead of `STAND_IN_CONFIG` to `spawn_robot()` / the task envs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


@dataclass(frozen=True)
class ActionLimits:
    """Physical limits used both to clip actions and to build gym Box spaces."""

    max_linear_vel: float = 1.0     # m/s, base vx/vy
    max_angular_vel: float = 1.5    # rad/s, base omega
    arm_workspace_radius: float = 0.9  # m, max EE reach from arm_base_link
    arm_workspace_min_z: float = 0.05
    arm_workspace_max_z: float = 1.3
    gripper_open_width: float = 0.04    # m, per-finger prismatic travel
    gripper_close_width: float = 0.0


@dataclass(frozen=True)
class SpawnPose:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.05)
    yaw: float = 0.0


@dataclass(frozen=True)
class RobotConfig:
    """Engine-agnostic robot description consumed by sim_env/* and robot_iface/skills.py."""

    name: str
    engine: str  # "pybullet" (today) | "isaac_sim" (tomorrow)

    # Asset location — exactly one of these is meaningful depending on `engine`.
    urdf_path: str | None = None   # PyBullet
    usd_path: str | None = None    # Isaac Sim

    base_link_name: str = "base_link"
    arm_joint_names: List[str] = field(default_factory=list)
    gripper_joint_names: List[str] = field(default_factory=list)
    ee_link_name: str = "ee_link"
    arm_base_link_name: str = "arm_base_link"

    action_limits: ActionLimits = field(default_factory=ActionLimits)
    default_spawn_pose: SpawnPose = field(default_factory=SpawnPose)

    # Where sensors/attachment points live, by name -> link they're rigidly mounted to.
    sensor_attachment_points: Dict[str, str] = field(default_factory=dict)

    def resolve_asset_path(self) -> str:
        if self.engine == "pybullet":
            if not self.urdf_path:
                raise ValueError(f"RobotConfig '{self.name}' has engine=pybullet but no urdf_path")
            return self.urdf_path
        if self.engine == "isaac_sim":
            if not self.usd_path:
                raise ValueError(f"RobotConfig '{self.name}' has engine=isaac_sim but no usd_path")
            return self.usd_path
        raise ValueError(f"Unknown engine '{self.engine}'")


# ---------------------------------------------------------------------------
# Today: PyBullet stand-in (§5.4 — "build or find a simple 4-wheel-base + arm
# URDF stand-in for logic testing. Don't chase visual or dimensional accuracy").
# ---------------------------------------------------------------------------
STAND_IN_CONFIG = RobotConfig(
    name="stand_in_pybullet",
    engine="pybullet",
    urdf_path=str(ASSETS_DIR / "robots" / "stand_in_robot.urdf"),
    base_link_name="base_link",
    # shoulder_yaw_joint first: without a yaw DOF the arm is planar and cannot
    # reach anything off the x-z plane (see stand_in_robot.urdf).
    arm_joint_names=["shoulder_yaw_joint", "shoulder_joint", "elbow_joint", "wrist_joint"],
    gripper_joint_names=["left_finger_joint", "right_finger_joint"],
    ee_link_name="ee_link",
    arm_base_link_name="arm_base_link",
    action_limits=ActionLimits(),
    default_spawn_pose=SpawnPose(position=(0.0, 0.0, 0.08), yaw=0.0),
    sensor_attachment_points={"wrist_camera": "wrist_link", "base_lidar": "base_link"},
)

# ---------------------------------------------------------------------------
# Tomorrow: Clearpath Ridgeback + Franka Panda, Isaac Sim built-in asset
# (§5.4 — `RidgebackFranka/ridgeback_franka.usd`, no custom authoring needed).
#
# Joint/link names below were verified against the articulation as actually
# loaded on the GB10 (see Suraj/check_isaac_hospital.py): 12 DOF total, made up
# of 3 dummy planar base joints + panda_joint1..7 + 2 fingers.
#
# `usd_path` is relative to Isaac Sim's asset root, not absolute: resolve it
# with `isaacsim.storage.native.get_assets_root_path()`. It previously pointed
# at `omniverse://localhost/.../Isaac/4.0/...`, which needs a local Nucleus
# server that does not exist on this box and pins a stale asset version.
# ---------------------------------------------------------------------------
RIDGEBACK_FRANKA_CONFIG = RobotConfig(
    name="ridgeback_franka",
    engine="isaac_sim",
    usd_path="/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
    base_link_name="base_link",
    arm_joint_names=[
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
    ],
    gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
    ee_link_name="panda_hand",
    arm_base_link_name="panda_link0",
    action_limits=ActionLimits(
        max_linear_vel=1.0,
        max_angular_vel=1.5,
        arm_workspace_radius=0.85,  # Franka's real reach envelope
        gripper_open_width=0.04,
    ),
    default_spawn_pose=SpawnPose(position=(0.0, 0.0, 0.0), yaw=0.0),
    sensor_attachment_points={"wrist_camera": "panda_hand", "base_lidar": "base_link"},
)


def get_config(name: str) -> RobotConfig:
    """Lookup helper for CLIs (`--robot stand_in` / `--robot ridgeback_franka`)."""
    registry = {
        "stand_in": STAND_IN_CONFIG,
        "stand_in_pybullet": STAND_IN_CONFIG,
        "ridgeback_franka": RIDGEBACK_FRANKA_CONFIG,
    }
    try:
        return registry[name]
    except KeyError as e:
        raise ValueError(f"Unknown robot config '{name}'. Known: {list(registry)}") from e
