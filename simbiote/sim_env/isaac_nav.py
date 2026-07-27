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


class MainThreadBridge:
    """Run callables on the main thread on behalf of worker threads.

    Omniverse Kit is not thread-safe: `SimulationApp.update()` and
    `SimulationContext.step()` must be called from the thread that created the
    app, and calling them from anywhere else aborts the process outright -- no
    Python traceback, just a dead simulator.

    That collides with `task_executor.execute()`, which deliberately runs each
    skill in a `ThreadPoolExecutor` so a wedged skill can be abandoned on
    timeout. The executor's design is right and worth keeping, so instead of
    unpicking it, skills hand their simulator work to this bridge: the worker
    thread blocks, the main loop picks the callable up and runs it, and the
    result (or exception) goes back to the caller.

    Called from the main thread itself, `call()` just runs the function -- so
    direct use without a server around it behaves exactly as before.
    """

    def __init__(self) -> None:
        import queue as _queue
        import threading

        self._queue: "_queue.Queue" = _queue.Queue()
        self._threading = threading

    def call(self, function: Callable[[], object]) -> object:
        threading = self._threading
        if threading.current_thread() is threading.main_thread():
            return function()

        box: Dict[str, object] = {}
        done = threading.Event()
        self._queue.put((function, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box.get("value")

    def pump(self) -> bool:
        """Run one queued callable if there is one. Main thread only."""
        import queue as _queue

        try:
            function, box, done = self._queue.get_nowait()
        except _queue.Empty:
            return False
        try:
            box["value"] = function()
        except BaseException as error:  # noqa: BLE001 - relayed to the caller
            box["error"] = error
        finally:
            done.set()
        return True

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
    # Arrival tolerance at the final waypoint. The base is 0.96 m long, so
    # anything under ~0.5 m is asking it to park on a point; 0.6 m had runs
    # coasting to a halt 0.61 m out and reporting failure, and 0.8 m still lost
    # one that settled 0.86 m out.
    goal_threshold: float = 1.0
    # Weight on the cross-track correction blended into the policy's velocity.
    #
    # The policy was trained against three *point* obstacles scattered in an
    # open 4 m box; it has no behaviour for a continuous surface, so in a
    # corridor it drifts into the wall and grinds along it rather than
    # standing off. The planned path is the line where clearance is actually
    # guaranteed (>= ROBOT_RADIUS by construction), so instead of asking the
    # policy for something it never learned, this pulls the base back onto
    # that line. The policy still chooses the heading and the speed.
    #
    # 0 reproduces the raw policy; ~1.0 tracks the path tightly.
    cross_track_gain: float = 0.9
    # Cross-track error beyond which the correction saturates.
    cross_track_limit: float = 0.8
    # Below this clearance the base slows toward `min_speed_scale`, on the
    # theory that grinding in the ~0.70 m corridor to room_1 was an overshoot
    # problem. Measured: it is not -- 15/20 with the slowdown against 16/20
    # without, so it is off by default. Left in because it is the obvious
    # thing to reach for and the measurement is worth keeping.
    slow_clearance: float = 1.2
    min_speed_scale: float = 1.0
    # Waypoint is "reached" and the carrot advances at this distance.
    waypoint_threshold: float = 0.7
    # Multiplier on the policy's velocity output. The hospital is open enough
    # that full speed is safe on straights, but the policy was trained in a
    # 4 m box and its outputs saturate; scaling <1 trades speed for tracking.
    speed_scale: float = 1.0
    # How far the position target is allowed to lead the base's actual joint
    # position. This is the whole force budget: the drives are position PD, so
    # commanded-minus-actual *is* the push. Integrating the target off the
    # measured position instead leaves a one-tick error (~0.03 m), which is not
    # enough force to round a door frame -- the base creeps and stops. Too
    # large and the 1e7 stiffness bulldozes through walls (the 10 m test in
    # check_isaac_hospital.py). 0.35 m pushes firmly and still gets stopped by
    # geometry.
    max_target_lead: float = 0.35
    # Give up after this long per navigate_to call. The base tops out near
    # 0.65 m/s under the lead clamp and the longest route in the building is
    # 75 m, so 120 s failed the long traversals on the clock alone.
    timeout_s: float = 300.0
    # Abort if the robot makes less than this much progress over the stall
    # window -- a policy that has parked itself against a wall should fail
    # loudly rather than burn the timeout. Tuned generously: brushing a
    # corridor wall pauses the base for ~1 s at a time while it works past,
    # and a 4 s / 0.15 m rule called those genuine recoveries a stall.
    stall_window_s: float = 8.0
    stall_progress: float = 0.25


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
        width: int = 1600,
        height: int = 1000,
        low_graphics: bool = False,
        hide_roof: bool = False,
        minimal_scene: bool = False,
        pace: float = 0.0,
        render_interval: float = 0.25,
        progress_every: int = 0,
        flat: bool = False,
    ):
        self.tuning = tuning or NavTuning()
        self.locations = dict(locations or HOSPITAL_LOCATIONS)
        self.controller = controller
        # Rendering costs ~10x the physics step, so headless measurement runs
        # skip it entirely. A headed run has to render or the window shows a
        # frozen robot while the traversal happens invisibly -- default it on
        # rather than making every caller remember to ask.
        if headless:
            self.renders_per_control = 0
        else:
            self.renders_per_control = renders_per_control or 3
        self._spawn = (float(spawn[0]), float(spawn[1]))
        self._width, self._height = width, height
        self._low_graphics = low_graphics
        self._hide_roof = hide_roof
        self._minimal_scene = minimal_scene
        self._pace = pace
        self._render_interval = render_interval
        self._last_render = 0.0
        self._progress_every = progress_every
        self._flat = flat

        self.infer = load_policy(checkpoint) if controller == "policy" else None
        self.checkpoint = str(checkpoint)
        self.map: Optional[HospitalMap] = None

        self._boot(headless=headless, assets_root=assets_root)
        # After _boot: building the grid needs `pxr`, and under Isaac Sim's own
        # interpreter that only imports once SimulationApp has started. Loading
        # a cached grid would have worked either way, which is exactly why this
        # ordering has to be deliberate -- it fails only on a cold cache.
        if self.map is None:
            self.map = HospitalMap.load_or_build()

    # -- bring-up -----------------------------------------------------------

    def _boot(self, headless: bool, assets_root: Optional[str]) -> None:
        from isaacsim import SimulationApp

        self._app = SimulationApp(
            {"headless": headless, "width": self._width, "height": self._height}
        )

        import carb
        import omni.usd
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        settings = carb.settings.get_settings()

        if self._low_graphics and not headless:
            # The hospital is a heavy RTX scene: 1064 prop instances, glossy
            # floors, and 104 point lights. Turning off the effects that sample
            # the scene repeatedly per pixel is where the cost is -- geometry
            # alone is cheap by comparison. Physics is untouched.
            for key, value in {
                "/rtx/reflections/enabled": False,
                "/rtx/translucency/enabled": False,
                "/rtx/ambientOcclusion/enabled": False,
                "/rtx/indirectDiffuse/enabled": False,
                "/rtx/directLighting/sampledLighting/enabled": False,
                "/rtx/post/dlss/execMode": 0,
                "/rtx/post/aa/op": 0,
                "/rtx/sceneDb/ambientLightIntensity": 1.0,
            }.items():
                settings.set(key, value)
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
        if self._flat:
            # Flat mode: same robot, same map, same planned route -- but drawn
            # on an empty floor instead of hospital.usd.
            #
            # The hospital is a 1064-instance ray-traced scene, and on this box
            # the first frames after physics starts moving block long enough
            # that the window stops answering the compositor. Rendering a
            # ground plane and a handful of markers cannot do that.
            #
            # Navigation is genuinely unchanged: `HospitalMap` is still built
            # from hospital.usd, so A* still routes around the building's real
            # walls and the robot still drives to the real coordinates. What is
            # lost is contact -- there is no geometry here to bump into, so
            # this shows the intended route rather than proving collision
            # avoidance. The headless hospital runs are what prove that.
            context.new_stage()
            stage = context.get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            # Needed here rather than after _boot: the floor and the markers
            # are sized and placed from the grid.
            self.map = HospitalMap.load_or_build()
            self._build_flat_scene(stage)
        else:
            context.open_stage(hospital_usd)
            stage = context.get_stage()
        self.stage = stage

        if self._hide_roof or self._minimal_scene:
            self._declutter(stage, minimal=self._minimal_scene)

        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        physx_scene.CreateSolverTypeAttr("TGS")
        # GPU dynamics allocates its buffers lazily, when bodies actually start
        # moving -- so the cost lands on the first command rather than at boot,
        # competing with the renderer for the same unified memory at exactly
        # the wrong moment. There is one articulation here; CPU PhysX carries
        # it comfortably.
        physx_scene.CreateEnableGPUDynamicsAttr(not self._low_graphics)

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
        self._settle(90)  # let the base settle before calibrating

        self._reset_drive_target()
        self._calibrate_base_axes()
        if not headless:
            self._setup_chase_camera()

    def _build_flat_scene(self, stage) -> None:
        """A floor, a light, and a marker per destination.

        Deliberately trivial to draw: an unlit-ish distant light and a handful
        of cubes. The point is that the renderer can never be the reason the
        robot stops moving.
        """
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        light.CreateIntensityAttr(3000.0)
        light.CreateAngleAttr(1.0)

        # Floor spanning the hospital's own footprint, so the markers sit at
        # the coordinates they really have.
        spec = self.map.spec
        width = spec.width * spec.resolution
        depth = spec.height * spec.resolution
        centre = (spec.origin_x + width / 2.0, spec.origin_y + depth / 2.0)
        floor = UsdGeom.Cube.Define(stage, "/World/Floor")
        floor.CreateSizeAttr(1.0)
        floor_xform = UsdGeom.Xformable(floor)
        floor_xform.AddTranslateOp().Set(Gf.Vec3d(centre[0], centre[1], -0.05))
        floor_xform.AddScaleOp().Set(Gf.Vec3f(float(width), float(depth), 0.1))
        floor.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.24, 0.28)])

        # One marker per named destination, so the route reads as going
        # somewhere rather than wandering.
        for index, (name, (x, y)) in enumerate(sorted(self.locations.items())):
            marker = UsdGeom.Cube.Define(stage, Sdf.Path(f"/World/Marker_{index}"))
            marker.CreateSizeAttr(1.0)
            marker_xform = UsdGeom.Xformable(marker)
            marker_xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.6))
            marker_xform.AddScaleOp().Set(Gf.Vec3f(0.6, 0.6, 1.2))
            marker.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.55, 0.85)])
        self._marker_count = len(self.locations)

    def show_route(self, path: Sequence[Tuple[float, float]]) -> None:
        """Draw the planned route as a strip of pucks (flat mode only)."""
        if not self._flat or not path:
            return
        from pxr import Gf, Sdf, UsdGeom

        for prim in list(self.stage.GetPrimAtPath("/World").GetChildren()):
            if prim.GetName().startswith("Route_"):
                self.stage.RemovePrim(prim.GetPath())

        # Sample along the path rather than one puck per waypoint, so long
        # straight segments still read as a line.
        points: List[Tuple[float, float]] = []
        for index in range(1, len(path)):
            start, end = path[index - 1], path[index]
            span = math.dist(start, end)
            for k in range(max(int(span / 1.0), 1)):
                t = k / max(int(span / 1.0), 1)
                points.append((start[0] + (end[0] - start[0]) * t,
                               start[1] + (end[1] - start[1]) * t))
        points.append(path[-1])

        for index, (x, y) in enumerate(points[:400]):
            puck = UsdGeom.Cube.Define(self.stage, Sdf.Path(f"/World/Route_{index}"))
            puck.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(puck)
            xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.02))
            xform.AddScaleOp().Set(Gf.Vec3f(0.22, 0.22, 0.04))
            puck.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.75, 0.15)])

    @staticmethod
    def _declutter(stage, minimal: bool = False) -> int:
        """Hide the roof, the ceiling lights, and optionally every loose prop.

        Two things this buys: the scene renders far cheaper (104 ray-traced
        point lights is the single biggest cost in the hospital), and with the
        roof off you can see the robot moving through the floor plan from above
        instead of staring at a ceiling.

        `minimal` goes further and leaves only the building shell -- walls,
        floors, doors, stairs. That is what "just the rooms" means visually.

        All of this sets *visibility*, which in USD is independent of
        collision. Every prop keeps its collider, so the robot navigates an
        identical building to the one the 16/20 measurements were taken in --
        it just does not have to be drawn. Anything else would invalidate the
        results, which is why nothing here deactivates or deletes a prim.
        """
        from pxr import UsdGeom

        # The shell: what has to stay visible for the place to read as rooms.
        keep = ("Wall", "Floor", "Door", "Stair", "Column", "Window")

        hidden = 0
        for prim in stage.Traverse():
            name = prim.GetName()
            drop = name.startswith("Geo_Roof") or "Ceiling" in name or name.startswith("Light_test")
            if minimal and not drop and name.startswith("Geo_"):
                # A prop is anything that is not part of the shell.
                drop = not any(token in name for token in keep)
            if not drop:
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden += 1
        print(
            f"  hid {hidden} prims "
            f"({'shell only' if minimal else 'roof + ceiling lights'}); "
            "colliders unaffected, so navigation is unchanged"
        )
        return hidden

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
        self._settle(180)

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
            self._settle(60)
            end = self.base_xy()
            targets[dof] -= delta
            self.robot.apply_action(ArticulationAction(joint_positions=targets))
            self._settle(60)
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

    def _setup_chase_camera(self, height: float = 14.0, back: float = 7.0) -> None:
        """A camera looking down at the robot, re-aimed as it drives.

        Framing the middle of a 76 m building leaves the robot a few pixels
        across. This sits above and behind it instead, which with the roof
        hidden gives a clean overhead view of the route.
        """
        from pxr import Gf, UsdGeom

        self._camera = UsdGeom.Camera.Define(self.stage, "/ChaseCamera")
        self._camera.CreateFocalLengthAttr(18.0)
        self._camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 5000.0))
        self._camera_op = UsdGeom.Xformable(self._camera).AddTransformOp()
        self._camera_offset = (back, height)
        self._aim_camera()
        try:
            import omni.kit.viewport.utility as viewport_utils

            viewport_utils.get_active_viewport().camera_path = "/ChaseCamera"
        except Exception as exc:  # noqa: BLE001 - framing is a convenience
            print(f"  (could not bind the chase camera: {exc})")

    def _aim_camera(self) -> None:
        if getattr(self, "_camera_op", None) is None:
            return
        from pxr import Gf

        x, y = self.base_xy()
        back, height = self._camera_offset
        target = Gf.Vec3d(x, y, 0.4)
        eye = Gf.Vec3d(x - back, y - back, height)
        self._camera_op.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())

    def _advance(self, physics_steps: int, tick: int = 0) -> None:
        """Step physics, and keep a headed window alive while doing it.

        `SimulationContext.render()` draws the viewport but does not pump the
        application's event loop. A traversal is a tight 20-60 s loop, so with
        render() alone the window stops answering the compositor and the
        desktop marks it "not responding" -- and a session killed at that point
        looks exactly like an out-of-memory crash. `SimulationApp.update()`
        both draws and pumps, so headed runs go through it instead.
        """
        for _ in range(physics_steps):
            self.sim.step(render=False)
        if not self.renders_per_control:
            return
        # Throttle on the wall clock, not on a tick count. A frame in this
        # scene can take far longer than the 33 ms control period, and
        # "every Nth tick" then means the loop spends nearly all its time
        # rendering: physics barely advances, so the robot looks frozen while
        # the window stops answering the compositor. Gating on elapsed time is
        # self-limiting -- if frames are slow we simply draw fewer of them, and
        # the robot keeps moving. Jittery, but moving.
        now = time.time()
        if now - self._last_render < self._render_interval:
            return
        self._last_render = now
        self._aim_camera()
        self._app.update()
        # Hand the compositor a moment. Without a yield the control loop goes
        # straight back into physics and the window manager can miss enough
        # ping/expose events in a row to declare the app hung -- which is what
        # "Isaac Sim is not responding" was, not a crash.
        if self._pace:
            time.sleep(self._pace)

    def _settle(self, steps: int) -> None:
        """Step for a while during setup, pumping a headed window as we go.

        Boot does several hundred steps (folding the arm, settling, calibrating
        the base axes) before the first command is even accepted. Without
        pumping, the window is born unresponsive.
        """
        for index in range(steps):
            self.sim.step(render=False)
            if self.renders_per_control and index % 10 == 0:
                self._app.update()

    def _reset_drive_target(self) -> None:
        """Re-seat the integrated target on the base's actual joint state."""
        self._target = self.robot.get_joint_positions().copy()

    def _apply_velocity(self, vx: float, vy: float, omega: float, dt: float) -> None:
        """Integrate a world-frame velocity into the base's position drives.

        The target is held in `self._target` and integrated forward, *not*
        re-read from the encoder each tick. Re-reading makes the commanded
        position chase the actual one, so the tracking error -- and therefore
        the drive force -- never exceeds a single tick's motion and the base
        stalls against any contact at all.

        The lead is clamped instead, which bounds the force directly: enough to
        push past a door frame, not enough to drive through a wall.
        """
        from isaacsim.core.utils.types import ArticulationAction

        joint_delta = self.joint_from_world @ np.array([vx * dt, vy * dt])
        self._target[self.dof_x] += float(joint_delta[0])
        self._target[self.dof_y] += float(joint_delta[1])
        self._target[self.dof_yaw] += float(omega * dt)

        actual = self.robot.get_joint_positions()
        lead = self.tuning.max_target_lead
        for dof in (self.dof_x, self.dof_y):
            self._target[dof] = float(
                np.clip(self._target[dof], actual[dof] - lead, actual[dof] + lead)
            )
        self._target[self.dof_yaw] = float(
            np.clip(self._target[self.dof_yaw], actual[self.dof_yaw] - 0.5, actual[self.dof_yaw] + 0.5)
        )
        self.robot.apply_action(ArticulationAction(joint_positions=self._target))

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

    def _cross_track(
        self, path: Sequence[Tuple[float, float]], index: int, xy: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Velocity correction pulling the base back onto the planned segment.

        Returns a world-frame (vx, vy) perpendicular to the active segment,
        proportional to how far off it the base has drifted. Zero when the base
        is on the line, so on a straight open run this contributes nothing and
        the policy is driving alone.
        """
        if index >= len(path):
            return (0.0, 0.0)
        start = path[max(index - 1, 0)]
        end = path[index]
        segment_x, segment_y = end[0] - start[0], end[1] - start[1]
        length = math.hypot(segment_x, segment_y)
        if length < 1e-6:
            return (0.0, 0.0)
        ux, uy = segment_x / length, segment_y / length
        # Left-hand normal of the segment direction.
        px, py = -uy, ux
        # Signed distance from the line, positive when the base is off to the
        # left of travel. Drive back along -offset * normal.
        dx, dy = xy[0] - start[0], xy[1] - start[1]
        offset = dx * px + dy * py
        limit = self.tuning.cross_track_limit
        offset = max(-limit, min(limit, offset))
        return (-offset * px, -offset * py)

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
        self.show_route(path)
        # Start each traversal with the drive target on the current state, so a
        # lead built up at the end of the previous call cannot launch this one.
        self._reset_drive_target()

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
            clearance = self.map.clearance(xy[0], xy[1], limit=3.0)
            min_clearance = min(min_clearance, clearance)

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

            if self.tuning.cross_track_gain:
                cx, cy = self._cross_track(path, index, xy)
                vx += cx * self.tuning.cross_track_gain
                vy += cy * self.tuning.cross_track_gain
                # Re-clip: the correction can push the sum past the bound the
                # policy's own action space is limited to.
                speed = math.hypot(vx, vy)
                if speed > MAX_LINEAR_VEL:
                    vx, vy = vx * MAX_LINEAR_VEL / speed, vy * MAX_LINEAR_VEL / speed

            scale = self.tuning.speed_scale
            # Ease off in tight geometry, using the clearance already measured
            # at the top of this tick.
            if clearance < self.tuning.slow_clearance:
                fraction = max(clearance, 0.0) / self.tuning.slow_clearance
                scale *= self.tuning.min_speed_scale + (
                    1.0 - self.tuning.min_speed_scale
                ) * fraction

            self._apply_velocity(vx * scale, vy * scale, omega * scale, dt)
            # Draws every Nth control tick rather than every one: a frame costs
            # ~10x a physics step, so drawing at the full 30 Hz control rate
            # makes a 40 m traversal take minutes of wall clock. Every 3rd tick
            # is 10 fps -- smooth enough to watch, and often enough that the
            # window keeps answering the compositor.
            self._advance(physics_steps, tick=step)

            # Heartbeat: if the window is janky, this is how you tell the
            # difference between "rendering slowly" and "not moving".
            if self._progress_every and step % self._progress_every == 0:
                print(
                    f"    ... {math.dist(xy, goal_xy):6.1f} m to go "
                    f"({xy[0]:7.2f}, {xy[1]:7.2f})",
                    flush=True,
                )

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
        self._target = positions.copy()
        self._settle(120)
        self._reset_drive_target()

        landed = self.base_xy()
        if math.dist(landed, xy) > 1.0:
            raise RuntimeError(
                f"teleport to {tuple(round(v, 2) for v in xy)} landed at "
                f"{tuple(round(v, 2) for v in landed)}"
            )

    def spin(self) -> None:
        """Pump the app once -- keeps a headed window responsive while idle."""
        self._app.update()

    def is_running(self) -> bool:
        """False once the operator closes the window."""
        return bool(self._app.is_running())

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:  # noqa: BLE001 - shutting down regardless
            pass
        self._app.close()
