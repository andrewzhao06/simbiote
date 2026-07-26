"""Throwaway PyBullet stand-in robot for exercising the Step 3 teleop chain
tonight. This is NOT Step 2's deliverable -- Teammate 2 owns the real
stand-in robot/URDF and, tomorrow, the real ridgeback_franka.usd binding
(master doc Part 5.4/6.7). This module exists only so teleop_session.py has
something concrete to drive end to end before that lands; swap it out for
Step 2's real sim binding as soon as it's available.

Base motion is applied directly (kinematic-ish, via resetBaseVelocity) and
the 2-DOF arm is driven by pybullet's built-in IK toward arm_target_pose --
no attempt at matching the real Ridgeback+Franka's kinematics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import pybullet as p
import pybullet_data

from factoryflow.robot_iface.actions import GripperState, RobotAction

URDF_PATH = Path(__file__).parent / "toy_robot.urdf"

SHOULDER_JOINT = "shoulder"
ELBOW_JOINT = "elbow"
HAND_LINK = "hand"
FINGER_LEFT_JOINT = "finger_left_joint"
FINGER_RIGHT_JOINT = "finger_right_joint"

GRIPPER_OPEN_POS = 0.04
GRIPPER_CLOSED_POS = 0.0

ARM_MOTOR_FORCE = 50.0
GRIPPER_MOTOR_FORCE = 20.0


class ToyRobotEnv:
    """Implements apply_action(RobotAction) against a throwaway PyBullet body."""

    def __init__(self, gui: bool = True, spawn_position=(0.0, 0.0, 0.3)):
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.body_id = p.loadURDF(str(URDF_PATH), basePosition=spawn_position, useFixedBase=False)

        self._joint_index: Dict[str, int] = {}
        self._link_name_by_index: Dict[int, str] = {}
        for i in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, i)
            self._joint_index[info[1].decode("utf-8")] = i
            self._link_name_by_index[i] = info[12].decode("utf-8")

        # "wrist" is the fixed joint whose child link is "hand" (HAND_LINK);
        # joint index doubles as that child link's index in pybullet's convention.
        self._hand_link_index = self._joint_index["wrist"]
        self._movable_joint_indices: List[int] = [
            i
            for i in range(p.getNumJoints(self.body_id))
            if p.getJointInfo(self.body_id, i)[2] != p.JOINT_FIXED
        ]

    def apply_action(self, action: RobotAction) -> None:
        vx, vy, omega = action.base_velocity
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        _, _, yaw = p.getEulerFromQuaternion(base_orn)

        world_vx = vx * math.cos(yaw) - vy * math.sin(yaw)
        world_vy = vx * math.sin(yaw) + vy * math.cos(yaw)
        p.resetBaseVelocity(self.body_id, linearVelocity=[world_vx, world_vy, 0.0], angularVelocity=[0.0, 0.0, omega])

        tx, ty, tz = action.arm_target_pose.position
        world_target = [
            base_pos[0] + tx * math.cos(yaw) - ty * math.sin(yaw),
            base_pos[1] + tx * math.sin(yaw) + ty * math.cos(yaw),
            base_pos[2] + tz,
        ]
        joint_angles = p.calculateInverseKinematics(self.body_id, self._hand_link_index, world_target)
        angle_by_joint_index = dict(zip(self._movable_joint_indices, joint_angles))

        for name in (SHOULDER_JOINT, ELBOW_JOINT):
            idx = self._joint_index[name]
            p.setJointMotorControl2(
                self.body_id, idx, p.POSITION_CONTROL,
                targetPosition=angle_by_joint_index[idx], force=ARM_MOTOR_FORCE,
            )

        finger_target = GRIPPER_OPEN_POS if action.gripper_state == GripperState.OPEN else GRIPPER_CLOSED_POS
        for name in (FINGER_LEFT_JOINT, FINGER_RIGHT_JOINT):
            idx = self._joint_index[name]
            p.setJointMotorControl2(
                self.body_id, idx, p.POSITION_CONTROL,
                targetPosition=finger_target, force=GRIPPER_MOTOR_FORCE,
            )

    def step(self) -> None:
        p.stepSimulation()

    def close(self) -> None:
        p.disconnect(self._client)
