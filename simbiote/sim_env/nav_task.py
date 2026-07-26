"""Navigation task — spec §5.4:

    obs = privileged state (pose, nearby obstacles, goal)
    action = base velocity
    reward = progress-to-goal, collision penalty, smoothness

PyBullet stand-in today; swaps to Isaac Lab's task API tomorrow (§5.6) —
the observation/action/reward *shape* below is exactly what should carry
over, only the physics backend underneath changes.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

from simbiote.robot.robot_config import STAND_IN_CONFIG, RobotConfig
from simbiote.sim_env import pybullet_scene as scene

NUM_TRACKED_OBSTACLES = 3
# [robot_x, robot_y, cos(yaw), sin(yaw), goal_dx, goal_dy, goal_dist,
#  robot_vx, robot_vy, robot_omega] + 3 * (obst_dx, obst_dy, obst_dist)
OBS_DIM = 10 + NUM_TRACKED_OBSTACLES * 3
ACT_DIM = 3  # vx, vy, omega


class NavEnv(gym.Env if gym is not None else object):
    """Collision-free point-to-point navigation for the stand-in mobile base."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        robot_config: RobotConfig = STAND_IN_CONFIG,
        room_size: float = 4.0,
        num_obstacles: int = 3,
        max_steps: int = 300,
        sim_steps_per_action: int = 8,
        goal_threshold: float = 0.25,
        gui: bool = False,
        seed: Optional[int] = None,
        goal_override: Optional[Tuple[float, float]] = None,
    ):
        if gym is None:
            raise ImportError("gymnasium is required for NavEnv (pip install gymnasium)")

        self.robot_config = robot_config
        self.room_size = room_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.sim_steps_per_action = sim_steps_per_action
        self.goal_threshold = goal_threshold
        self.gui = gui
        self._rng = random.Random(seed)
        # Set by robot_iface/skills.py's navigate_to() to target a real scene
        # location instead of a randomized training goal.
        self.goal_override = goal_override

        limits = robot_config.action_limits
        act_high = np.array([limits.max_linear_vel, limits.max_linear_vel, limits.max_angular_vel], dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._client: Optional[int] = None
        self._robot_id: Optional[int] = None
        self._obstacle_ids: List[int] = []
        self._wall_ids: List[int] = []
        self._goal_xy: Tuple[float, float] = (0.0, 0.0)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_dist = 0.0
        self._step_count = 0

    # -- Gymnasium API --------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = random.Random(seed)
        if self._client is None:
            self._client = scene.connect(gui=self.gui)
            scene.load_ground_plane(self._client)
        else:
            self._teardown_episode_bodies()

        wall_half = self.room_size / 2.0 - 0.2
        self._wall_ids = scene.build_stand_in_arena(self._client, room_size=self.room_size)
        self._robot_id = scene.load_robot(self.robot_config, self._client)

        self._obstacle_ids = []
        for _ in range(self.num_obstacles):
            pos = (self._rng.uniform(-wall_half, wall_half), self._rng.uniform(-wall_half, wall_half), 0.2)
            obj = scene.spawn_graspable_box(self._client, pos, half_extents=(0.15, 0.15, 0.2), mass_kg=0.0, rgba=(0.3, 0.3, 0.9, 1.0))
            self._obstacle_ids.append(obj.body_id)

        self._goal_xy = self.goal_override or (self._rng.uniform(-wall_half, wall_half), self._rng.uniform(-wall_half, wall_half))
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._step_count = 0
        self._prev_dist = self._goal_distance()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        import pybullet as p

        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        vx, vy, omega = action.tolist()
        p.resetBaseVelocity(self._robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, omega], physicsClientId=self._client)
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._client)

        self._step_count += 1
        dist = self._goal_distance()
        progress = self._prev_dist - dist
        self._prev_dist = dist

        collided = self._check_collision()
        smoothness_penalty = float(np.linalg.norm(action - self._prev_action)) * 0.01
        self._prev_action = action

        reward = 5.0 * progress - smoothness_penalty
        terminated = False
        if collided:
            reward -= 5.0
            terminated = True
        success = dist < self.goal_threshold
        if success:
            reward += 20.0
            terminated = True

        truncated = self._step_count >= self.max_steps
        info = {"success": success, "collided": collided, "goal_distance": dist}
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self._client is not None:
            scene.disconnect(self._client)
            self._client = None

    # -- Helpers ----------------------------------------------------------

    def _teardown_episode_bodies(self) -> None:
        import pybullet as p

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        scene.load_ground_plane(self._client)

    def _robot_pose(self) -> Tuple[float, float, float]:
        import pybullet as p

        pos, orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return pos[0], pos[1], yaw

    def _robot_velocity(self) -> Tuple[float, float, float]:
        import pybullet as p

        lin, ang = p.getBaseVelocity(self._robot_id, physicsClientId=self._client)
        return lin[0], lin[1], ang[2]

    def _goal_distance(self) -> float:
        x, y, _ = self._robot_pose()
        return math.hypot(self._goal_xy[0] - x, self._goal_xy[1] - y)

    def _check_collision(self) -> bool:
        import pybullet as p

        for obs_id in self._obstacle_ids + self._wall_ids:
            pts = p.getContactPoints(bodyA=self._robot_id, bodyB=obs_id, physicsClientId=self._client)
            if len(pts) > 0:
                return True
        return False

    def _get_obs(self) -> np.ndarray:
        import pybullet as p

        x, y, yaw = self._robot_pose()
        vx, vy, omega = self._robot_velocity()
        gdx, gdy = self._goal_xy[0] - x, self._goal_xy[1] - y
        gdist = math.hypot(gdx, gdy)

        obstacle_feats: List[float] = []
        dists = []
        for obs_id in self._obstacle_ids:
            opos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self._client)
            odx, ody = opos[0] - x, opos[1] - y
            dists.append((math.hypot(odx, ody), odx, ody))
        dists.sort(key=lambda t: t[0])
        for i in range(NUM_TRACKED_OBSTACLES):
            if i < len(dists):
                d, odx, ody = dists[i]
                obstacle_feats.extend([odx, ody, d])
            else:
                obstacle_feats.extend([0.0, 0.0, 0.0])

        obs = [x, y, math.cos(yaw), math.sin(yaw), gdx, gdy, gdist, vx, vy, omega] + obstacle_feats
        return np.array(obs, dtype=np.float32)
