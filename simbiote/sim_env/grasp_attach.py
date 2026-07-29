"""Grasp-attach constraint logic — spec §5.2/§5.4, the piece everything else
(including the wheelchair stretch task) depends on getting right first.

Today (PyBullet): `p.createConstraint(..., jointType=p.JOINT_FIXED, ...)`
dynamically welds the end-effector link to a target body at its *current*
relative transform, forming a closed kinematic chain for as long as the
constraint exists. `p.removeConstraint()` releases it.

Tomorrow (Isaac Sim / PhysX 5): re-implement the body of `attach()` /
`release()` against `omni.physx`'s scene interface (or author a
`UsdPhysics.FixedJoint` at grasp time) — the signature below is written to
stay stable across that swap so `nav_task.py`/`grasp_task.py`/
`wheelchair_task.py` never need to change when the engine underneath does.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraspConstraint:
    """Handle returned by `attach()`. Opaque to callers other than `release()`."""

    constraint_id: int
    physics_client: int
    robot_id: int
    ee_link_index: int
    target_body_id: int


def attach(
    robot_id: int,
    ee_link_index: int,
    target_body_id: int,
    physics_client: int = 0,
    max_force: float = 500.0,
) -> GraspConstraint:
    """Weld `target_body_id` to `robot_id`'s `ee_link_index` at its current
    relative pose (i.e. "grab it wherever it currently is relative to the
    gripper" — the caller is responsible for having already driven the EE
    close enough for this to make physical sense).

    Returns a `GraspConstraint` handle; pass it to `release()` to detach.
    """
    import pybullet as p

    if ee_link_index == -1:
        ee_pos, ee_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
    else:
        link_state = p.getLinkState(robot_id, ee_link_index, physicsClientId=physics_client)
        ee_pos, ee_orn = link_state[0], link_state[1]

    target_pos, target_orn = p.getBasePositionAndOrientation(
        target_body_id, physicsClientId=physics_client
    )

    # JOINT_FIXED welds parentFrame (expressed in the EE link's local frame)
    # to childFrame (expressed in the target body's local frame) so they
    # coincide in world space from this instant on. Taking childFrame as
    # identity (the target's own origin) means parentFrame must be set to
    # "the target's current pose as seen from inside the EE link's frame" --
    # i.e. inverse(ee_pose) * target_pose -- otherwise the object would snap
    # to the EE's origin the instant the constraint is created.
    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
    parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
        inv_ee_pos, inv_ee_orn, target_pos, target_orn
    )

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link_index,
        childBodyUniqueId=target_body_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=parent_frame_pos,
        parentFrameOrientation=parent_frame_orn,
        childFramePosition=[0, 0, 0],
        childFrameOrientation=[0, 0, 0, 1],
        physicsClientId=physics_client,
    )
    p.changeConstraint(constraint_id, maxForce=max_force, physicsClientId=physics_client)

    return GraspConstraint(
        constraint_id=constraint_id,
        physics_client=physics_client,
        robot_id=robot_id,
        ee_link_index=ee_link_index,
        target_body_id=target_body_id,
    )


def release(constraint: GraspConstraint | None) -> None:
    """Detach a previously-created grasp. No-op if `constraint` is None so
    call sites can do `release(self._grasp); self._grasp = None` unconditionally.
    """
    if constraint is None:
        return
    import pybullet as p

    p.removeConstraint(constraint.constraint_id, physicsClientId=constraint.physics_client)


def is_holding(constraint: GraspConstraint | None) -> bool:
    return constraint is not None
