"""Shared PyBullet plumbing used by `nav_task.py`, `grasp_task.py` and
`wheelchair_task.py` — connecting, loading URDFs, building a stand-in arena,
and spawning "graspable objects" tagged with the same
`is_graspable`/`grasp_type`/`mass_kg` attributes Step 1's real `.usd` export
carries (spec §5.5's "Object-pick task tie-in" note), since PyBullet/URDF has
no native equivalent of USD's custom schema attributes.

Tomorrow this whole module is replaced by loading `hospital.usd` +
`ridgeback_franka.usd` into Isaac Sim — `nav_task.py`/`grasp_task.py`'s task
logic (reward, obs, action) does not change, only what's imported here does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simbiote.robot.robot_config import RobotConfig


def connect(gui: bool = False) -> int:
    """Start a fresh PyBullet physics client. Always DIRECT (headless) unless
    `gui=True` — training/CI never wants a window."""
    import pybullet as p

    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client)
    return client


def disconnect(client: int) -> None:
    import pybullet as p

    p.disconnect(physicsClientId=client)


def load_ground_plane(client: int) -> int:
    import pybullet as p
    import pybullet_data

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return p.loadURDF("plane.urdf", physicsClientId=client)


def load_robot(config: RobotConfig, client: int, fixed_base: bool = False) -> int:
    import pybullet as p

    pos = config.default_spawn_pose.position
    yaw = config.default_spawn_pose.yaw
    orn = p.getQuaternionFromEuler([0, 0, yaw])
    return p.loadURDF(
        config.resolve_asset_path(),
        basePosition=pos,
        baseOrientation=orn,
        useFixedBase=fixed_base,
        physicsClientId=client,
    )


def joint_name_to_index(body_id: int, client: int) -> dict[str, int]:
    """URDF joint name -> joint/link index. A link's index is defined by the
    index of the joint that attaches it (pybullet convention); base link is -1."""
    import pybullet as p

    mapping = {}
    for j in range(p.getNumJoints(body_id, physicsClientId=client)):
        info = p.getJointInfo(body_id, j, physicsClientId=client)
        joint_name = info[1].decode("utf-8")
        mapping[joint_name] = j
    return mapping


def link_name_to_index(body_id: int, client: int) -> dict[str, int]:
    import pybullet as p

    mapping = {}
    for j in range(p.getNumJoints(body_id, physicsClientId=client)):
        info = p.getJointInfo(body_id, j, physicsClientId=client)
        link_name = info[12].decode("utf-8")
        mapping[link_name] = j
    return mapping


def movable_joint_indices(body_id: int, client: int) -> list[int]:
    """Raw joint indices of every non-fixed joint, in increasing order.

    `p.calculateInverseKinematics()` returns one angle per DOF-bearing
    joint, in *this* order and length -- **not** indexed by raw joint index
    (fixed joints, of which the stand-in URDF has several -- wheel mounts,
    `arm_base_fixed`, `ee_fixed` -- are skipped entirely). Indexing its
    return value directly by raw joint index silently reads the wrong
    (or an out-of-range, wrongly-defaulted) angle whenever any fixed joint
    exists lower in the chain -- always resolve through this list instead.
    """
    import pybullet as p

    indices = []
    for j in range(p.getNumJoints(body_id, physicsClientId=client)):
        info = p.getJointInfo(body_id, j, physicsClientId=client)
        if info[2] != p.JOINT_FIXED:
            indices.append(j)
    return indices


@dataclass
class RobotHandles:
    """Resolved joint/link indices for a loaded `RobotConfig` — computed once
    at load time instead of re-resolving names every simulation step."""

    body_id: int
    client: int
    config: RobotConfig
    arm_joint_indices: list[int]
    gripper_joint_indices: list[int]
    ee_link_index: int
    base_link_index: int
    arm_base_link_index: int
    dof_joint_indices: list[int]  # see movable_joint_indices() -- IK output order
    _ik_position_of: dict[int, int] = field(default_factory=dict)

    def ik_angle(self, joint_angles, raw_joint_index: int) -> float:
        """Look up `raw_joint_index`'s value out of a
        `p.calculateInverseKinematics()` result, correcting for that
        result being indexed over movable joints only (see
        `movable_joint_indices`'s docstring)."""
        return joint_angles[self._ik_position_of[raw_joint_index]]

    @classmethod
    def build(cls, config: RobotConfig, client: int, body_id: int) -> RobotHandles:
        joints = joint_name_to_index(body_id, client)
        links = link_name_to_index(body_id, client)
        dof_indices = movable_joint_indices(body_id, client)
        return cls(
            body_id=body_id,
            client=client,
            config=config,
            arm_joint_indices=[joints[n] for n in config.arm_joint_names],
            gripper_joint_indices=[joints[n] for n in config.gripper_joint_names],
            ee_link_index=links[config.ee_link_name],
            base_link_index=links.get(config.base_link_name, -1),
            arm_base_link_index=links.get(config.arm_base_link_name, -1),
            dof_joint_indices=dof_indices,
            _ik_position_of={raw: pos for pos, raw in enumerate(dof_indices)},
        )


@dataclass
class GraspableObject:
    """Mirrors Step 1's per-object semantic tags (`is_graspable`, `grasp_type`,
    `mass_kg`) so `grasp_task.py`'s reward/approach logic reads attributes
    instead of hardcoding assumptions — the same interface the wheelchair
    task's handle-grasp uses per §5.5."""

    body_id: int
    is_graspable: bool = True
    grasp_type: str = "pinch"
    mass_kg: float = 0.3


def spawn_graspable_box(
    client: int,
    position: tuple[float, float, float],
    half_extents: tuple[float, float, float] = (0.03, 0.03, 0.03),
    mass_kg: float = 0.3,
    grasp_type: str = "pinch",
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> GraspableObject:
    import pybullet as p

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=list(rgba), physicsClientId=client
    )
    body_id = p.createMultiBody(
        baseMass=mass_kg,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(position),
        physicsClientId=client,
    )
    p.changeDynamics(body_id, -1, lateralFriction=0.8, physicsClientId=client)
    return GraspableObject(
        body_id=body_id, is_graspable=True, grasp_type=grasp_type, mass_kg=mass_kg
    )


def spawn_wall(
    client: int,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> int:
    import pybullet as p

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=[0.6, 0.6, 0.6, 1.0],
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0.0,  # static
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(center),
        physicsClientId=client,
    )


def spawn_table(
    client: int,
    top_center_xy: tuple[float, float],
    top_height: float,
    half_extents_xy: tuple[float, float] = (0.3, 0.3),
    thickness: float = 0.04,
) -> int:
    """A static platform so a spawned graspable object rests at a sane
    height instead of free-falling from wherever it was placed (grasp_task.py
    spawns objects at a fixed height with no floor directly underneath)."""
    import pybullet as p

    half_extents = (half_extents_xy[0], half_extents_xy[1], thickness / 2)
    center = (top_center_xy[0], top_center_xy[1], top_height - thickness / 2)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=[0.55, 0.4, 0.25, 1.0],
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(center),
        physicsClientId=client,
    )


def build_stand_in_arena(
    client: int, room_size: float = 4.0, wall_height: float = 0.6
) -> list[int]:
    """A simple box arena standing in for a scanned room (spec §5.4: "Don't
    chase visual or dimensional accuracy here; you're validating reward
    logic and constraint behavior, not producing a demo asset.")."""

    half = room_size / 2.0
    thickness = 0.05
    return [
        spawn_wall(client, (half, 0, wall_height / 2), (thickness, half, wall_height / 2)),
        spawn_wall(client, (-half, 0, wall_height / 2), (thickness, half, wall_height / 2)),
        spawn_wall(client, (0, half, wall_height / 2), (half, thickness, wall_height / 2)),
        spawn_wall(client, (0, -half, wall_height / 2), (half, thickness, wall_height / 2)),
    ]
