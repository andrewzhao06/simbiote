"""Manipulation task — spec §5.4:

    obs = privileged state (object pose, EE pose)
    action = EE target pose + gripper
    reward = approach progress, grasp success, drop penalty

The RL action here (a small per-step EE displacement + gripper command) is
an internal training parameterization, distinct from the coarser
`RobotAction`/`pick_up(object_id)` interface Step 3/4 call — see
`robot_iface/skills.py`, which wraps a *trained* policy from this task
behind that higher-level call.

Reads `is_graspable` / `grasp_type` / `mass_kg` off the spawned object
(spec §5.5's "Object-pick task tie-in") instead of hardcoding assumptions,
since the wheelchair stretch task's handle-grasp shares this same interface.
"""

from __future__ import annotations

import math
import random
from typing import Any, ClassVar

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

from simbiote.robot.robot_config import STAND_IN_CONFIG, RobotConfig
from simbiote.sim_env import grasp_attach
from simbiote.sim_env import pybullet_scene as scene

# obs = [ee_x,ee_y,ee_z, ee_qx,ee_qy,ee_qz,ee_qw, gripper_opening,
#        obj_x,obj_y,obj_z, rel_x,rel_y,rel_z, rel_dist, mass_kg, is_holding]
OBS_DIM = 17
ACT_DIM = 4  # dx, dy, dz (EE delta), gripper_cmd in [-1, 1]

MAX_EE_STEP = 0.03  # m per action, keeps IK targets from jumping unrealistically
LIFT_HEIGHT = 0.12  # m above spawn height counts as "lifted" (success)
# m, how close the EE must be to trigger attach(). Measured against the actual
# collision geometry rather than picked: the graspable box is 0.03 m half-extent
# and ee_link is a 0.04 x 0.08 x 0.04 box, so once the gripper is flush against
# the object the centre-to-centre distance still cannot drop below ~0.05-0.07 m
# depending on approach axis. At the old 0.06 m a scripted oracle pressing the
# EE into the object bottomed out at 0.061 m and never triggered a grasp, so
# nothing could ever succeed.
GRASP_DIST_THRESHOLD = 0.10


class GraspEnv(gym.Env if gym is not None else object):
    """Approach + grasp + lift a tagged object with the stand-in arm."""

    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    def __init__(
        self,
        robot_config: RobotConfig = STAND_IN_CONFIG,
        max_steps: int = 200,
        sim_steps_per_action: int = 6,
        gui: bool = False,
        seed: int | None = None,
        object_position_override: tuple[float, float, float] | None = None,
    ):
        if gym is None:
            raise ImportError("gymnasium is required for GraspEnv (pip install gymnasium)")

        self.robot_config = robot_config
        self.max_steps = max_steps
        self.sim_steps_per_action = sim_steps_per_action
        self.gui = gui
        self._rng = random.Random(seed)
        # Set by robot_iface/skills.py's pick_up() to target a real tagged
        # scene object instead of a randomized training spawn.
        self.object_position_override = object_position_override

        act_high = np.array([MAX_EE_STEP, MAX_EE_STEP, MAX_EE_STEP, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)
        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._client: int | None = None
        self._robot_id: int | None = None
        self._handles = None
        self._object = None
        self._object_spawn_z = 0.0
        self._arm_base_xy: tuple[float, float] = (0.0, 0.0)
        self._ee_target = np.zeros(3, dtype=np.float32)
        self._grasp: grasp_attach.GraspConstraint | None = None
        self._gripper_closed = False
        self._prev_dist = 0.0
        self._step_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = random.Random(seed)
        if self._client is None:
            self._client = scene.connect(gui=self.gui)
            scene.load_ground_plane(self._client)
        else:
            self._reset_bodies()

        self._robot_id = scene.load_robot(self.robot_config, self._client, fixed_base=True)
        self._handles = scene.RobotHandles.build(self.robot_config, self._client, self._robot_id)
        self._arm_base_xy = self._get_arm_base_xy()

        obj_pos = self.object_position_override or (
            self._rng.uniform(0.35, 0.55),
            self._rng.uniform(-0.2, 0.2),
            0.25,
        )
        object_half_extent_z = 0.03
        scene.spawn_table(
            self._client, (obj_pos[0], obj_pos[1]), top_height=obj_pos[2] - object_half_extent_z
        )
        self._object = scene.spawn_graspable_box(
            self._client, obj_pos, mass_kg=self._rng.uniform(0.1, 0.5)
        )
        self._object_spawn_z = obj_pos[2]

        self._ee_target = np.array(self._get_ee_pos(), dtype=np.float32)
        self._grasp = None
        self._gripper_closed = False
        self._step_count = 0
        self._prev_dist = self._ee_to_object_dist()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        import pybullet as p

        action = np.clip(
            np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high
        )
        delta, gripper_cmd = action[:3], float(action[3])

        limits = self.robot_config.action_limits
        candidate = self._ee_target + delta
        candidate[2] = float(
            np.clip(candidate[2], limits.arm_workspace_min_z, limits.arm_workspace_max_z)
        )
        # `arm_workspace_radius` is documented (robot_config.py) as reach
        # from `arm_base_link`, not from the world origin -- clip relative
        # to the arm's actual base position so valid targets near the arm
        # (which may not be spawned at world (0, 0)) aren't wrongly rejected.
        base_x, base_y = self._arm_base_xy
        radius = math.hypot(candidate[0] - base_x, candidate[1] - base_y)
        if radius > limits.arm_workspace_radius:
            scale = limits.arm_workspace_radius / max(radius, 1e-6)
            candidate[0] = base_x + (candidate[0] - base_x) * scale
            candidate[1] = base_y + (candidate[1] - base_y) * scale
        self._ee_target = candidate

        self._gripper_closed = gripper_cmd > 0.0
        self._drive_arm_to_target()
        self._drive_gripper(self._gripper_closed)
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._client)

        self._step_count += 1
        reward, terminated, info = self._compute_reward_and_status()
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self._client is not None:
            scene.disconnect(self._client)
            self._client = None

    # -- Helpers ------------------------------------------------------------

    def _reset_bodies(self) -> None:
        import pybullet as p

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        scene.load_ground_plane(self._client)
        self._grasp = None

    def _drive_arm_to_target(self) -> None:
        import pybullet as p

        # Position-only IK: this stand-in arm doesn't have enough DOF to also
        # track a fixed end-effector orientation without fighting the position
        # target -- matches spec §5.4's "don't chase accuracy" framing for the
        # stand-in; tomorrow's 7-DOF Franka arm has DOF to spare for both.
        #
        # `currentPositions` matters: the arm has a yaw joint plus two pitch
        # joints, so a given EE position has more than one solution. Solving
        # cold each step lets the solver hop between branches and the arm
        # thrashes -- commanding a target at y=0 could leave the EE at y=0.58.
        # Seeding from the current configuration keeps successive solutions
        # continuous.
        # PyBullet only uses the seeded/null-space solver when limits, ranges
        # and rest poses are all supplied -- passing currentPositions on its own
        # is silently ignored.
        indices = self._handles.dof_joint_indices
        current = [
            state[0]
            for state in p.getJointStates(self._robot_id, indices, physicsClientId=self._client)
        ]
        lower, upper = [], []
        for joint_index in indices:
            info = p.getJointInfo(self._robot_id, joint_index, physicsClientId=self._client)
            joint_lower, joint_upper = info[8], info[9]
            if joint_upper < joint_lower:  # unlimited joint
                joint_lower, joint_upper = -math.pi, math.pi
            lower.append(joint_lower)
            upper.append(joint_upper)
        joint_angles = p.calculateInverseKinematics(
            self._robot_id,
            self._handles.ee_link_index,
            targetPosition=self._ee_target.tolist(),
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=[high - low for low, high in zip(lower, upper, strict=True)],
            restPoses=current,
            jointDamping=[0.05] * len(indices),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self._client,
        )
        for joint_index in self._handles.arm_joint_indices:
            p.setJointMotorControl2(
                self._robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=self._handles.ik_angle(joint_angles, joint_index),
                force=80.0,
                physicsClientId=self._client,
            )

    def _drive_gripper(self, closed: bool) -> None:
        import pybullet as p

        limits = self.robot_config.action_limits
        width = limits.gripper_close_width if closed else limits.gripper_open_width
        for joint_index in self._handles.gripper_joint_indices:
            p.setJointMotorControl2(
                self._robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=width,
                force=20.0,
                physicsClientId=self._client,
            )

    def _get_arm_base_xy(self) -> tuple[float, float]:
        import pybullet as p

        if self._handles.arm_base_link_index < 0:
            # Fallback for robot configs that don't resolve an
            # `arm_base_link` -- treat the workspace as origin-centered
            # (previous behavior) rather than erroring.
            return (0.0, 0.0)
        state = p.getLinkState(
            self._robot_id, self._handles.arm_base_link_index, physicsClientId=self._client
        )
        return state[0][0], state[0][1]

    def _get_ee_pos(self) -> tuple[float, float, float]:
        import pybullet as p

        state = p.getLinkState(
            self._robot_id, self._handles.ee_link_index, physicsClientId=self._client
        )
        return state[0]

    def _get_ee_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        import pybullet as p

        state = p.getLinkState(
            self._robot_id, self._handles.ee_link_index, physicsClientId=self._client
        )
        return state[0], state[1]

    def _object_pos(self) -> tuple[float, float, float]:
        import pybullet as p

        pos, _ = p.getBasePositionAndOrientation(self._object.body_id, physicsClientId=self._client)
        return pos

    def _ee_to_object_dist(self) -> float:
        ee = self._get_ee_pos()
        obj = self._object_pos()
        return math.dist(ee, obj)

    def _compute_reward_and_status(self):
        dist = self._ee_to_object_dist()
        approach_progress = self._prev_dist - dist
        self._prev_dist = dist
        obj_z = self._object_pos()[2]
        lift = obj_z - self._object_spawn_z

        reward = 2.0 * approach_progress
        terminated = False
        success = False
        grasped_this_step = False

        if self._grasp is None and self._gripper_closed and dist < GRASP_DIST_THRESHOLD:
            self._grasp = grasp_attach.attach(
                self._robot_id,
                self._handles.ee_link_index,
                self._object.body_id,
                physics_client=self._client,
            )
            reward += 10.0
            grasped_this_step = True
        elif self._grasp is not None and not self._gripper_closed:
            # Opened the gripper before lift succeeded -- drop penalty (spec: "drop penalty").
            grasp_attach.release(self._grasp)
            self._grasp = None
            if lift < LIFT_HEIGHT:
                reward -= 5.0
                terminated = True

        if self._grasp is not None:
            reward += 3.0 * max(lift, 0.0)
            if lift >= LIFT_HEIGHT:
                reward += 20.0
                success = True
                terminated = True

        info = {
            "success": success,
            "is_holding": self._grasp is not None,
            "grasped_this_step": grasped_this_step,
            "object_lift": lift,
            "ee_object_dist": dist,
        }
        return reward, terminated, info

    def _get_obs(self) -> np.ndarray:
        ee_pos, ee_orn = self._get_ee_pose()
        obj_pos = self._object_pos()
        rel = tuple(obj_pos[i] - ee_pos[i] for i in range(3))
        limits = self.robot_config.action_limits
        gripper_opening = (
            limits.gripper_close_width if self._gripper_closed else limits.gripper_open_width
        )
        obs = (
            list(ee_pos)
            + list(ee_orn)
            + [gripper_opening]
            + list(obj_pos)
            + list(rel)
            + [
                math.dist(ee_pos, obj_pos),
                self._object.mass_kg,
                1.0 if self._grasp is not None else 0.0,
            ]
        )
        return np.array(obs, dtype=np.float32)
