"""Drive the trained nav policy through hospital.usd under Isaac Sim / PhysX 5.

This is the Isaac counterpart to `robot_iface/skills.navigate_to()`. That one
spins up a throwaway 4 x 4 m PyBullet arena per call and shrinks the hospital
onto it; this one holds a persistent hospital and moves the Ridgeback + Franka
through the actual building.

How the 4 x 4 m policy drives a 76 x 42 m building
--------------------------------------------------
`NavEnv`'s observation is world-frame and absolute -- `[x, y, cos(yaw),
sin(yaw), goal_dx, goal_dy, goal_dist, vx, vy, omega]` plus three nearest
obstacles -- and the robot always resets to the origin with goals no more than
~2.4 m away. Feeding it a hospital pose (x = 17.26, goal 40 m off) is far
outside anything it saw in training, and it does not act sensibly there.

So the policy is used as what it actually is: a *local* steering controller.
Each control tick rebuilds an observation in a frame the policy recognises:

  * the robot sits at the local origin, so `x, y` are always ~0 as in training;
  * world axes are *not* rotated, because `NavEnv`'s action is a world-frame
    velocity -- rotating the frame would rotate the output back out of true;
  * the goal is a carrot on the A* path a fixed lookahead ahead, clipped to
    the radius the policy was trained against, so `goal_dx/dy/dist` stay in
    distribution regardless of how far the real destination is;
  * the three obstacle slots are filled from `HospitalMap`'s occupancy grid --
    real walls and props, in the same relative-offset encoding.

`HospitalMap` handles the long-range question (which way around the building);
the policy handles the short-range one (steer to a point without clipping it).

Driving the base
----------------
The Ridgeback's base is three dummy joints (prismatic x, prismatic y, revolute
z) driven by stiff position PD, not velocity. The policy emits velocities, so
they are integrated into position targets each tick. The mapping from joint
axes to world axes is *calibrated at startup* rather than assumed: the asset
authors its own orientation, so which way `dummy_base_prismatic_x_joint` points
in world terms depends on how the robot prim composed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from simbiote.sim_env.hospital_map import (
    DEFAULT_HOSPITAL_USD,
    HOSPITAL_LOCATIONS,
    SPAWN,
    HospitalMap,
)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

HOSPITAL_RELATIVE = "/Isaac/Environments/Hospital/hospital.usd"
ROBOT_RELATIVE = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
ASSET_ROOT_FALLBACK = "/home/dell/AI/assets/isaac-5.1"
ASSET_ROOT_SETTING = "/persistent/isaac/asset_root/default"

# NavEnv's action bounds (robot_config.STAND_IN_CONFIG.action_limits).
MAX_LINEAR_VEL = 1.0
MAX_ANGULAR_VEL = 1.5

# NavEnv steps 8 PyBullet ticks at 240 Hz per action -> 30 Hz control.
CONTROL_HZ = 30.0
PHYSICS_HZ = 60.0


@dataclass
class NavTuning:
    """Parameters governing how the local policy is fed and trusted.

    Defaults are the values measured to work in `tune_hospital_nav.py`; every
    one of them changes behaviour enough to be worth naming rather than
    burying as a literal.
    """

    # How far along the path to place the carrot. Below ~1.0 m the robot
    # tracks the path tightly but corners slowly; above ~2.5 m it cuts corners
    # into walls because the carrot is through geometry the policy cannot see.
    lookahead: float = 1.6
    # Clip on |goal_dx, goal_dy| handed to the policy. NavEnv's arena is 4 m
    # across with goals sampled at >=1.0 m, so ~1.8 m is the edge of what it
    # saw. Feeding a larger delta pushes the network out of distribution.
    max_goal_delta: float = 1.8
    # Radius over which occupancy cells are searched for the obstacle slots.
    obstacle_radius: float = 2.5
    # Arrival tolerance at the final waypoint.
    goal_threshold: float = 0.6
    # Waypoint is "reached" and the carrot advances at this distance.
    waypoint_threshold: float = 0.7
    # Multiplier on the policy's velocity output. The hospital is open enough
    # that full speed is safe on straights, but the policy was trained in a
    # 4 m box and its outputs saturate; scaling <1 trades speed for tracking.
    speed_scale: float = 1.0
    # Give up after this long per navigate_to call.
    timeout_s: float = 120.0
    # Abort if the robot makes less than this much progress over the stall
    # window -- a policy that has parked itself against a wall should fail
    # loudly rather than burn the timeout.
    stall_window_s: float = 4.0
    stall_progress: float = 0.15


@dataclass
class NavResult:
    success: bool
    location_id: str
    goal_xy: Tuple[float, float]
    final_xy: Tuple[float, float]
    goal_distance: float
    path_length: float
    travelled: float
    waypoints: int
    steps: int
    duration_s: float
    reason: str = ""
    collided: bool = False
    min_clearance: float = float("inf")
    trace: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skill": "navigate_to",
            "location_id": self.location_id,
            "success": self.success,
            "goal_xy": [round(v, 3) for v in self.goal_xy],
            "final_xy": [round(v, 3) for v in self.final_xy],
            "goal_distance": round(self.goal_distance, 3),
            "path_length_m": round(self.path_length, 2),
            "travelled_m": round(self.travelled, 2),
            "waypoints": self.waypoints,
            "steps": self.steps,
            "duration_s": round(self.duration_s, 2),
            "min_clearance_m": round(self.min_clearance, 2),
            "collided": self.collided,
            "reason": self.reason,
        }


def load_policy(checkpoint: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """`f(obs) -> action` for an ActorCriticMLP `.pt` checkpoint."""
    import torch

    from simbiote.training.policy_net import ActorCriticMLP

    policy = ActorCriticMLP.load(Path(checkpoint))
    policy.eval()

    def infer(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action, _, _ = policy.act(
                torch.as_tensor(obs[None, :], dtype=torch.float32), deterministic=True
            )
        return action.numpy()[0]

    return infer


class IsaacHospital:
    """A live hospital with a Ridgeback+Franka in it, held open across calls.

    Deliberately a persistent handle rather than a per-skill context manager:
    Isaac Sim takes minutes to boot and PhysX cooks the building's collision
    meshes on first play, so a compound plan ("go to the supply room, pick up
    the tray, take it to room 1") has to run against one instance or it would
    spend its entire budget restarting the simulator.
    """

    def __init__(
        self,
        headless: bool = True,
        checkpoint: str | Path = CHECKPOINT_DIR / "nav_bc.pt",
        spawn: Tuple[float, float] = SPAWN,
        locations: Optional[Dict[str, Tuple[float, float]]] = None,
        tuning: Optional[NavTuning] = None,
        assets_root: Optional[str] = None,
        controller: str = "policy",
        renders_per_control: int = 0,
    ):
        self.tuning = tuning or NavTuning()
        self.locations = dict(locations or HOSPITAL_LOCATIONS)
        self.controller = controller
        # Rendering costs ~10x the physics step. Headed runs need it; headless
        # measurement runs do not.
        self.renders_per_control = renders_per_control if not headless else 0
        self._spawn = (float(spawn[0]), float(spawn[1]))

        self.map = HospitalMap.load_or_build()
        self.infer = load_policy(checkpoint) if controller == "policy" else None
        self.checkpoint = str(checkpoint)

        self._boot(headless=headless, assets_root=assets_root)

    # -- bring-up -----------------------------------------------------------

    def _boot(self, headless: bool, assets_root: Optional[str]) -> None:
        from isaacsim import SimulationApp

        self._app = SimulationApp({"headless": headless, "width": 1600, "height": 1000})

        import carb
        import omni.usd
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        settings = carb.settings.get_settings()
        if assets_root:
            settings.set(ASSET_ROOT_SETTING, assets_root)
        configured = settings.get(ASSET_ROOT_SETTING)
        # A remote root streams every asset over the network, which the event
        # is air-gapped against; fall back to the local pack.
        if configured and configured.startswith(("http", "omniverse://")):
            settings.set(ASSET_ROOT_SETTING, ASSET_ROOT_FALLBACK)

        root = get_assets_root_path(skip_check=True)
        hospital_usd = f"{root}{HOSPITAL_RELATIVE}"
        robot_usd = f"{root}{ROBOT_RELATIVE}"

        context = omni.usd.get_context()
        context.open_stage(hospital_usd)
        stage = context.get_stage()
        self.stage = stage

        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        physx_scene.CreateSolverTypeAttr("TGS")
        physx_scene.CreateEnableGPUDynamicsAttr(True)

        self.robot_path = "/World_Robot"
        robot_prim = stage.DefinePrim(self.robot_path, "Xform")
        robot_prim.GetReferences().AddReference(robot_usd)
        # The asset authors its own translate op; AddTranslateOp would raise.
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(robot_prim).GetOrderedXformOps()}
        spawn3 = Gf.Vec3d(self._spawn[0], self._spawn[1], 0.05)
        if "xformOp:translate" in ops:
            ops["xformOp:translate"].Set(spawn3)
        else:
            UsdGeom.Xformable(robot_prim).AddTranslateOp().Set(spawn3)
        self._robot_prim = robot_prim

        # Without this the articulation is floating-base: driving the base
        # joints slides the anchor backwards and base_link never moves.
        anchor = UsdPhysics.FixedJoint.Define(stage, "/base_anchor")
        anchor.CreateBody1Rel().SetTargets([f"{self.robot_path}/world"])

        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim

        self.sim = SimulationContext(
            physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ
        )
        self.sim.initialize_physics()
        self.sim.play()

        self.robot = SingleArticulation(self.robot_path)
        self.robot.initialize(self.sim.physics_sim_view)
        self.base = SingleRigidPrim(f"{self.robot_path}/base_link")
        self.base.initialize(self.sim.physics_sim_view)

        # Adding the anchor reorders the DOFs, so never hardcode indices.
        self.dof_x = self.robot.get_dof_index("dummy_base_prismatic_x_joint")
        self.dof_y = self.robot.get_dof_index("dummy_base_prismatic_y_joint")
        self.dof_yaw = self.robot.get_dof_index("dummy_base_revolute_z_joint")

        self._fold_arm()
        for _ in range(90):  # let the base settle before calibrating
            self.sim.step(render=False)

        self._calibrate_base_axes()

    def _fold_arm(self) -> None:
        """Park the Franka in a compact pose.

        The arm's default configuration sticks out far enough to catch door
        frames the occupancy grid (built for a 0.45 m base radius) says are
        clear, which reads as the nav policy steering badly.
        """
        from isaacsim.core.utils.types import ArticulationAction

        names = list(self.robot.dof_names)
        arm = [n for n in names if n.startswith("panda_joint")]
        folded = np.array([0.0, -1.6, 0.0, -2.8, 0.0, 1.6, 0.785])
        targets = self.robot.get_joint_positions().copy()
        for name, value in zip(arm, folded):
            targets[self.robot.get_dof_index(name)] = value
        self.robot.apply_action(ArticulationAction(joint_positions=targets))
        for _ in range(180):
            self.sim.step(render=False)

    def _calibrate_base_axes(self) -> None:
        """Measure how joint x/y map onto world x/y.

        `ridgeback_franka.usd` authors its own orientation on the default prim,
        so the prismatic joints are not guaranteed to be world-axis aligned.
        Rather than assume, nudge each joint and watch base_link: the result is
        a 2x2 world<-joint matrix, inverted once here and reused per tick.
        """
        from isaacsim.core.utils.types import ArticulationAction

        def nudge(dof: int, delta: float) -> np.ndarray:
            start = self.base_xy()
            targets = self.robot.get_joint_positions().copy()
            targets[dof] += delta
            self.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(60):
                self.sim.step(render=False)
            end = self.base_xy()
            targets[dof] -= delta
            self.robot.apply_action(ArticulationAction(joint_positions=targets))
            for _ in range(60):
                self.sim.step(render=False)
            return (np.asarray(end) - np.asarray(start)) / delta

        col_x = nudge(self.dof_x, 0.5)
        col_y = nudge(self.dof_y, 0.5)
        world_from_joint = np.column_stack([col_x, col_y])
        determinant = float(np.linalg.det(world_from_joint))
        if abs(determinant) < 1e-3:
            # Degenerate: the base did not move for one of the joints, which
            # usually means the anchor is missing or the robot spawned inside
            # a prop and contact jammed the joints.
            raise RuntimeError(
                "base axis calibration failed -- joint motion did not move "
                f"base_link (matrix {world_from_joint.tolist()}). Check the "
                "fixed-joint anchor and that the spawn point is clear."
            )
        self.world_from_joint = world_from_joint
        self.joint_from_world = np.linalg.inv(world_from_joint)

    # -- state --------------------------------------------------------------

    def base_xy(self) -> Tuple[float, float]:
        # PhysX writes simulated poses to Fabric; the authored xformOp still
        # reads the spawn pose, so this must come from the rigid prim.
        position, _ = self.base.get_world_pose()
        return (float(position[0]), float(position[1]))

    def base_yaw(self) -> float:
        _, orientation = self.base.get_world_pose()
        w, x, y, z = (float(v) for v in orientation)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def base_velocity(self) -> Tuple[float, float, float]:
        linear = self.base.get_linear_velocity()
        angular = self.base.get_angular_velocity()
        return (float(linear[0]), float(linear[1]), float(angular[2]))

    # -- observation --------------------------------------------------------

    def _nearest_obstacles(self, x: float, y: float, count: int = 3) -> List[Tuple[float, float, float]]:
        """`count` nearest occupied cells as (dx, dy, distance), robot-relative.

        Matches `NavEnv._get_obs()`'s encoding: world-frame offsets, sorted
        nearest-first, zero-padded. The grid's raw (uninflated) occupancy is
        used so the distances mean the same thing they did in training --
        centre-to-obstacle, not centre-to-inflated-boundary.
        """
        spec = self.map.spec
        radius = int(self.tuning.obstacle_radius / spec.resolution)
        cx, cy = spec.to_cell(x, y)
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, spec.width)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, spec.height)
        window = self.map.raw[x0:x1, y0:y1]
        hits = np.argwhere(window)
        if hits.size == 0:
            return []
        wx = spec.origin_x + (hits[:, 0] + x0 + 0.5) * spec.resolution
        wy = spec.origin_y + (hits[:, 1] + y0 + 0.5) * spec.resolution
        dx, dy = wx - x, wy - y
        distance = np.hypot(dx, dy)
        order = np.argsort(distance)[: count * 40]
        out: List[Tuple[float, float, float]] = []
        # Nearest cells cluster on one wall face; spread the three slots over
        # distinct obstacles so the policy sees the corridor, not one point.
        for index in order:
            candidate = (float(dx[index]), float(dy[index]), float(distance[index]))
            if all(math.hypot(candidate[0] - p[0], candidate[1] - p[1]) > 0.8 for p in out):
                out.append(candidate)
            if len(out) == count:
                break
        return out

    def _observation(self, goal: Tuple[float, float]) -> np.ndarray:
        x, y = self.base_xy()
        yaw = self.base_yaw()
        vx, vy, omega = self.base_velocity()

        gdx, gdy = goal[0] - x, goal[1] - y
        distance = math.hypot(gdx, gdy)
        # Clip into the arena-scale the policy trained on, keeping the bearing.
        limit = self.tuning.max_goal_delta
        if distance > limit:
            gdx, gdy = gdx * limit / distance, gdy * limit / distance
            distance = limit

        features: List[float] = []
        for odx, ody, od in self._nearest_obstacles(x, y):
            features.extend([odx, ody, od])
        while len(features) < 9:
            features.extend([0.0, 0.0, 0.0])

        # The robot sits at the local origin: NavEnv always reset to (0, 0),
        # so passing the hospital's absolute coordinates here would be the
        # single largest distribution shift in the observation.
        return np.array(
            [0.0, 0.0, math.cos(yaw), math.sin(yaw), gdx, gdy, distance, vx, vy, omega]
            + features,
            dtype=np.float32,
        )

    # -- control ------------------------------------------------------------

    def _apply_velocity(self, vx: float, vy: float, omega: float, dt: float) -> None:
        """Integrate a world-frame velocity into the base's position drives."""
        from isaacsim.core.utils.types import ArticulationAction

        joint_delta = self.joint_from_world @ np.array([vx * dt, vy * dt])
        targets = self.robot.get_joint_positions().copy()
        targets[self.dof_x] += float(joint_delta[0])
        targets[self.dof_y] += float(joint_delta[1])
        targets[self.dof_yaw] += float(omega * dt)
        self.robot.apply_action(ArticulationAction(joint_positions=targets))

    def _carrot(self, path: Sequence[Tuple[float, float]], index: int, xy: Tuple[float, float]) -> Tuple[Tuple[float, float], int]:
        """Advance along `path` and return the point to steer at.

        Pure-pursuit style: skip waypoints already within `waypoint_threshold`,
        then interpolate `lookahead` metres along the remaining path so the
        goal handed to the policy is a smoothly moving target rather than a
        waypoint that snaps.
        """
        while index < len(path) - 1 and math.dist(xy, path[index]) < self.tuning.waypoint_threshold:
            index += 1

        remaining = self.tuning.lookahead
        point = path[index]
        cursor = xy
        cursor_index = index
        while cursor_index < len(path):
            segment = math.dist(cursor, path[cursor_index])
            if segment >= remaining:
                t = remaining / segment if segment > 1e-6 else 0.0
                point = (
                    cursor[0] + (path[cursor_index][0] - cursor[0]) * t,
                    cursor[1] + (path[cursor_index][1] - cursor[1]) * t,
                )
                break
            remaining -= segment
            cursor = path[cursor_index]
            point = cursor
            cursor_index += 1
        return point, index

    def navigate_to(
        self,
        location_id: str,
        goal_xy: Optional[Tuple[float, float]] = None,
        trace: bool = False,
    ) -> NavResult:
        """Drive to a named scene-graph location through the real building."""
        if goal_xy is None:
            if location_id not in self.locations:
                raise KeyError(
                    f"navigate_to: unknown location_id '{location_id}'. "
                    f"Known: {sorted(self.locations)}"
                )
            goal_xy = self.locations[location_id]

        started = time.time()
        start_xy = self.base_xy()
        path = self.map.plan(start_xy, goal_xy)
        if not path:
            return NavResult(
                False, location_id, goal_xy, start_xy,
                math.dist(start_xy, goal_xy), 0.0, 0.0, 0, 0,
                time.time() - started, reason="no path found",
            )
        path_length = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))

        dt = 1.0 / CONTROL_HZ
        physics_steps = max(int(PHYSICS_HZ / CONTROL_HZ), 1)
        max_steps = int(self.tuning.timeout_s * CONTROL_HZ)
        stall_steps = int(self.tuning.stall_window_s * CONTROL_HZ)

        index = 0
        travelled = 0.0
        previous = start_xy
        min_clearance = float("inf")
        history: List[Tuple[float, float]] = [start_xy]
        recent: List[Tuple[float, float]] = []
        reason = "timeout"
        success = False

        for step in range(max_steps):
            xy = self.base_xy()
            travelled += math.dist(xy, previous)
            previous = xy
            if trace:
                history.append(xy)
            min_clearance = min(min_clearance, self.map.clearance(xy[0], xy[1], limit=3.0))

            if math.dist(xy, goal_xy) < self.tuning.goal_threshold:
                success, reason = True, "reached goal"
                break

            target, index = self._carrot(path, index, xy)

            if self.controller == "policy":
                action = self.infer(self._observation(target))
                vx = float(np.clip(action[0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL))
                vy = float(np.clip(action[1], -MAX_LINEAR_VEL, MAX_LINEAR_VEL))
                omega = float(np.clip(action[2], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL))
            else:
                # Reference controller: drive straight at the carrot at the
                # policy's own speed limit. Exists so a policy result can be
                # read against something, rather than just "it moved".
                dx, dy = target[0] - xy[0], target[1] - xy[1]
                norm = math.hypot(dx, dy) or 1.0
                vx, vy = dx / norm * MAX_LINEAR_VEL, dy / norm * MAX_LINEAR_VEL
                omega = 0.0

            scale = self.tuning.speed_scale
            self._apply_velocity(vx * scale, vy * scale, omega * scale, dt)
            for _ in range(physics_steps):
                self.sim.step(render=False)
            if self.renders_per_control:
                self.sim.render()

            recent.append(xy)
            if len(recent) > stall_steps:
                recent.pop(0)
                if math.dist(recent[0], xy) < self.tuning.stall_progress:
                    reason = "stalled"
                    break

        duration = time.time() - started
        final = self.base_xy()
        return NavResult(
            success=success,
            location_id=location_id,
            goal_xy=tuple(goal_xy),
            final_xy=final,
            goal_distance=math.dist(final, goal_xy),
            path_length=path_length,
            travelled=travelled,
            waypoints=len(path),
            steps=step + 1,
            duration_s=duration,
            reason=reason,
            min_clearance=min_clearance,
            trace=history if trace else [],
        )

    def teleport(self, xy: Tuple[float, float]) -> None:
        """Reset the base to a pose without simulating the drive there.

        Used to set up an evaluation run from a chosen start; not a skill.

        Sets joint *state* rather than driving to a position target. Commanding
        the target and stepping cannot work: a 40 m jump inside the settle
        window is 16 m/s, the base never arrives, and the next run silently
        starts from wherever it stopped -- which shows up as a 73 m traversal
        reporting a 2.9 m path. The drive targets are rewritten to match, or
        the PD controller immediately hauls the robot back to the old target.
        """
        from isaacsim.core.utils.types import ArticulationAction

        current = self.base_xy()
        joint_delta = self.joint_from_world @ np.array([xy[0] - current[0], xy[1] - current[1]])
        positions = self.robot.get_joint_positions().copy()
        positions[self.dof_x] += float(joint_delta[0])
        positions[self.dof_y] += float(joint_delta[1])

        self.robot.set_joint_positions(positions)
        self.robot.set_joint_velocities(np.zeros_like(positions))
        self.robot.apply_action(ArticulationAction(joint_positions=positions))
        for _ in range(120):
            self.sim.step(render=False)

        landed = self.base_xy()
        if math.dist(landed, xy) > 1.0:
            raise RuntimeError(
                f"teleport to {tuple(round(v, 2) for v in xy)} landed at "
                f"{tuple(round(v, 2) for v in landed)}"
            )

    def spin(self) -> None:
        """Pump the app once -- keeps a headed window responsive while idle."""
        self._app.update()

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:  # noqa: BLE001 - shutting down regardless
            pass
        self._app.close()
