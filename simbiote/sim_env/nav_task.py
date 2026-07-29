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
from typing import Any, ClassVar

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

    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    def __init__(
        self,
        robot_config: RobotConfig = STAND_IN_CONFIG,
        room_size: float = 4.0,
        num_obstacles: int = 3,
        max_steps: int = 300,
        sim_steps_per_action: int = 8,
        goal_threshold: float = 0.25,
        gui: bool = False,
        seed: int | None = None,
        goal_override: tuple[float, float] | None = None,
        spawn_clearance: float = 0.6,
        obstacle_spacing: float = 0.5,
        min_goal_distance: float = 1.0,
        goal_clearance: float = 0.4,
    ):
        if gym is None:
            raise ImportError("gymnasium is required for NavEnv (pip install gymnasium)")

        self.robot_config = robot_config
        self.room_size = room_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.sim_steps_per_action = sim_steps_per_action
        self.goal_threshold = goal_threshold
        # Episode-validity margins (metres). See reset() for why these exist.
        self.spawn_clearance = spawn_clearance
        self.obstacle_spacing = obstacle_spacing
        self.min_goal_distance = min_goal_distance
        self.goal_clearance = goal_clearance
        self.gui = gui
        self._rng = random.Random(seed)
        # Set by robot_iface/skills.py's navigate_to() to target a real scene
        # location instead of a randomized training goal.
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
        self._obstacle_ids: list[int] = []
        self._wall_ids: list[int] = []
        self._goal_xy: tuple[float, float] = (0.0, 0.0)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_dist = 0.0
        self._step_count = 0

    # -- Gymnasium API --------------------------------------------------

    def _sample_xy(self, wall_half: float, acceptable, attempts: int = 200) -> tuple[float, float]:
        """Uniform sample inside the arena that satisfies `acceptable`.

        Falls back to the last draw rather than looping forever: a small room
        with many obstacles can be genuinely over-constrained, and a degraded
        episode beats a hung training run.
        """
        candidate = (0.0, 0.0)
        for _ in range(attempts):
            candidate = (
                self._rng.uniform(-wall_half, wall_half),
                self._rng.uniform(-wall_half, wall_half),
            )
            if acceptable(candidate):
                return candidate
        return candidate

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
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

        # Obstacles and the goal are rejection-sampled rather than placed by a
        # bare uniform draw. Unconstrained, ~23% of episodes spawned an obstacle
        # on top of the robot (nearest seen: 0.04 m) and terminated with a
        # collision on step 1, and another ~3% put the goal inside the success
        # threshold for a free win -- so roughly a quarter of every success rate
        # measured was decided at reset, before the policy acted at all.
        spawn_xy = (self.robot_config.default_spawn_pose.position[0],
                    self.robot_config.default_spawn_pose.position[1])

        # A caller-supplied goal (skills.navigate_to targeting a named scene
        # location) is known up front, so obstacles must also keep clear of it.
        # Otherwise one can spawn right on the destination and the robot
        # collides a few tens of centimetres short -- which is what "go to the
        # supply room" looked like: reached 0.55 m out, then hit a box.
        fixed_goal = tuple(self.goal_override) if self.goal_override else None

        def _far_enough(candidate, placed) -> bool:
            if (
                math.hypot(candidate[0] - spawn_xy[0], candidate[1] - spawn_xy[1])
                < self.spawn_clearance
            ):
                return False
            if fixed_goal is not None and math.hypot(
                candidate[0] - fixed_goal[0], candidate[1] - fixed_goal[1]
            ) < self.goal_clearance:
                return False
            return all(
                math.hypot(candidate[0] - other[0], candidate[1] - other[1])
                >= self.obstacle_spacing
                for other in placed
            )

        self._obstacle_ids = []
        placed_xy: list[tuple[float, float]] = []
        for _ in range(self.num_obstacles):
            xy = self._sample_xy(wall_half, lambda c: _far_enough(c, placed_xy))
            placed_xy.append(xy)
            obj = scene.spawn_graspable_box(
                self._client, (xy[0], xy[1], 0.2), half_extents=(0.15, 0.15, 0.2),
                mass_kg=0.0, rgba=(0.3, 0.3, 0.9, 1.0),
            )
            self._obstacle_ids.append(obj.body_id)

        if self.goal_override:
            self._goal_xy = self.goal_override
        else:
            # Far enough from the spawn that reaching it is an actual traverse,
            # and clear of the obstacles so it is reachable at all.
            self._goal_xy = self._sample_xy(
                wall_half,
                lambda c: (
                    math.hypot(c[0] - spawn_xy[0], c[1] - spawn_xy[1]) >= self.min_goal_distance
                    and all(
                        math.hypot(c[0] - o[0], c[1] - o[1]) >= self.goal_clearance
                        for o in placed_xy
                    )
                ),
            )
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
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

    def _robot_pose(self) -> tuple[float, float, float]:
        import pybullet as p

        pos, orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return pos[0], pos[1], yaw

    def _robot_velocity(self) -> tuple[float, float, float]:
        import pybullet as p

        lin, ang = p.getBaseVelocity(self._robot_id, physicsClientId=self._client)
        return lin[0], lin[1], ang[2]

    def _goal_distance(self) -> float:
        x, y, _ = self._robot_pose()
        return math.hypot(self._goal_xy[0] - x, self._goal_xy[1] - y)

    def _check_collision(self) -> bool:
        import pybullet as p

        for obs_id in self._obstacle_ids + self._wall_ids:
            pts = p.getContactPoints(
                bodyA=self._robot_id, bodyB=obs_id, physicsClientId=self._client
            )
            if len(pts) > 0:
                return True
        return False

    def _get_obs(self) -> np.ndarray:
        import pybullet as p

        x, y, yaw = self._robot_pose()
        vx, vy, omega = self._robot_velocity()
        gdx, gdy = self._goal_xy[0] - x, self._goal_xy[1] - y
        gdist = math.hypot(gdx, gdy)

        obstacle_feats: list[float] = []
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

        obs = [x, y, math.cos(yaw), math.sin(yaw), gdx, gdy, gdist, vx, vy, omega, *obstacle_feats]
        return np.array(obs, dtype=np.float32)
