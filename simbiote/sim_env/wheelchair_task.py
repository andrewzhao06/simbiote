"""Wheelchair co-navigation task — Tier 2/3 stretch, spec §5.5.

Trains only the third FSM state, "3. Constrained Nav (Co-Nav Policy, Base +
Wheelchair)" — Approach/Align/Grasp and Release/Park are the FSM states
Step 4's `task_executor.py` sequences around this RL task (§6b.4), not part
of this environment. At `reset()`, the wheelchair is spawned already welded
to the robot's end-effector (via `grasp_attach.attach()`), simulating that
the preceding Align & Grasp phase already succeeded.

    obs = privileged state (robot pose/vel, wheelchair pose + tilt, goal)
    action = base velocity (same shape as nav_task.py)
    reward = progress toward destination, penalize tilt angle and sharp
             turns that would tip it (spec's exact framing)

Explicitly optional per §5.6 — only build/train this once nav+grasp (Tier 1)
are solid. Not wired into the core acceptance tests for that reason.
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

from simbiote import assets
from simbiote.robot.robot_config import STAND_IN_CONFIG, RobotConfig
from simbiote.sim_env import grasp_attach
from simbiote.sim_env import pybullet_scene as scene

WHEELCHAIR_URDF = str(assets.WHEELCHAIR_URDF)

# [robot_x, robot_y, cos(yaw), sin(yaw), robot_vx, robot_vy, robot_omega,
#  chair_x, chair_y, chair_roll, chair_pitch, tilt_mag,
#  goal_dx, goal_dy, goal_dist]
OBS_DIM = 15
ACT_DIM = 3  # vx, vy, omega -- identical shape to nav_task.py

TIP_OVER_TILT_RAD = 0.5  # ~28.6 deg -- treated as a failed co-nav episode


class WheelchairEnv(gym.Env if gym is not None else object):
    """Push/guide an already-attached wheelchair to a goal without tipping it."""

    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    def __init__(
        self,
        robot_config: RobotConfig = STAND_IN_CONFIG,
        room_size: float = 5.0,
        max_steps: int = 300,
        sim_steps_per_action: int = 8,
        goal_threshold: float = 0.35,
        tilt_penalty_weight: float = 4.0,
        turn_penalty_weight: float = 0.5,
        gui: bool = False,
        seed: int | None = None,
        goal_override: tuple[float, float] | None = None,
    ):
        if gym is None:
            raise ImportError("gymnasium is required for WheelchairEnv (pip install gymnasium)")

        self.robot_config = robot_config
        self.room_size = room_size
        self.max_steps = max_steps
        self.sim_steps_per_action = sim_steps_per_action
        self.goal_threshold = goal_threshold
        self.tilt_penalty_weight = tilt_penalty_weight
        self.turn_penalty_weight = turn_penalty_weight
        self.gui = gui
        self._rng = random.Random(seed)
        self.goal_override = goal_override

        limits = robot_config.action_limits
        act_high = np.array(
            [limits.max_linear_vel, limits.max_linear_vel, limits.max_angular_vel], dtype=np.float32
        )
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)
        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._client: int | None = None
        self._robot_id: int | None = None
        self._handles = None
        self._wheelchair_id: int | None = None
        self._grasp: grasp_attach.GraspConstraint | None = None
        self._goal_xy: tuple[float, float] = (0.0, 0.0)
        self._prev_dist = 0.0
        self._step_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        import pybullet as p

        if seed is not None:
            self._rng = random.Random(seed)
        if self._client is None:
            self._client = scene.connect(gui=self.gui)
            scene.load_ground_plane(self._client)
        else:
            p.resetSimulation(physicsClientId=self._client)
            p.setGravity(0, 0, -9.81, physicsClientId=self._client)
            scene.load_ground_plane(self._client)

        wall_half = self.room_size / 2.0 - 0.2
        scene.build_stand_in_arena(self._client, room_size=self.room_size)
        self._robot_id = scene.load_robot(self.robot_config, self._client)
        self._handles = scene.RobotHandles.build(self.robot_config, self._client, self._robot_id)

        # Spawn the wheelchair with its handle roughly at the robot's resting EE pose,
        # simulating "Align & Grasp" already succeeded.
        ee_pos = p.getLinkState(
            self._robot_id, self._handles.ee_link_index, physicsClientId=self._client
        )[0]
        chair_pos = (ee_pos[0] + 0.28, ee_pos[1], 0.18)
        self._wheelchair_id = p.loadURDF(
            WHEELCHAIR_URDF, basePosition=chair_pos, physicsClientId=self._client
        )
        self._grasp = grasp_attach.attach(
            self._robot_id,
            self._handles.ee_link_index,
            self._wheelchair_id,
            physics_client=self._client,
        )

        self._goal_xy = self.goal_override or (
            self._rng.uniform(-wall_half, wall_half),
            self._rng.uniform(-wall_half, wall_half),
        )
        self._step_count = 0
        self._prev_dist = self._goal_distance()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        import pybullet as p

        action = np.clip(
            np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high
        )
        vx, vy, omega = action.tolist()
        p.resetBaseVelocity(
            self._robot_id,
            linearVelocity=[vx, vy, 0],
            angularVelocity=[0, 0, omega],
            physicsClientId=self._client,
        )
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._client)

        self._step_count += 1
        dist = self._goal_distance()
        progress = self._prev_dist - dist
        self._prev_dist = dist

        roll, pitch = self._wheelchair_tilt()
        tilt_mag = math.hypot(roll, pitch)

        reward = (
            5.0 * progress
            - self.tilt_penalty_weight * tilt_mag
            - self.turn_penalty_weight * abs(omega)
        )

        terminated = False
        tipped = tilt_mag > TIP_OVER_TILT_RAD
        if tipped:
            reward -= 20.0
            terminated = True

        success = dist < self.goal_threshold and not tipped
        if success:
            reward += 30.0
            terminated = True

        truncated = self._step_count >= self.max_steps
        info = {"success": success, "tipped": tipped, "goal_distance": dist, "tilt": tilt_mag}
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self._client is not None:
            grasp_attach.release(self._grasp)
            scene.disconnect(self._client)
            self._client = None

    # -- Helpers --------------------------------------------------------

    def _robot_pose(self):
        import pybullet as p

        pos, orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return pos, yaw

    def _robot_velocity(self):
        import pybullet as p

        return p.getBaseVelocity(self._robot_id, physicsClientId=self._client)

    def _wheelchair_pose(self):
        import pybullet as p

        return p.getBasePositionAndOrientation(self._wheelchair_id, physicsClientId=self._client)

    def _wheelchair_tilt(self) -> tuple[float, float]:
        import pybullet as p

        _, orn = self._wheelchair_pose()
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        return roll, pitch

    def _goal_distance(self) -> float:
        pos, _ = self._robot_pose()
        return math.hypot(self._goal_xy[0] - pos[0], self._goal_xy[1] - pos[1])

    def _get_obs(self) -> np.ndarray:
        pos, yaw = self._robot_pose()
        lin, ang = self._robot_velocity()
        chair_pos, _ = self._wheelchair_pose()
        roll, pitch = self._wheelchair_tilt()
        tilt_mag = math.hypot(roll, pitch)
        gdx, gdy = self._goal_xy[0] - pos[0], self._goal_xy[1] - pos[1]
        obs = [
            pos[0], pos[1], math.cos(yaw), math.sin(yaw), lin[0], lin[1], ang[2],
            chair_pos[0], chair_pos[1], roll, pitch, tilt_mag,
            gdx, gdy, math.hypot(gdx, gdy),
        ]
        return np.array(obs, dtype=np.float32)
