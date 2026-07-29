"""Step 2 (§5.1-5.4) bring-up check: robot spawns and drives in hospital.usd.

This is the Isaac Sim / PhysX 5 counterpart to the PyBullet work in
`simbiote/sim_env/`. It does not train anything -- it proves the substrate the
nav and grasp tasks will sit on:

  * hospital.usd loads with its props and colliders actually resolved
  * the Ridgeback + Franka articulation initialises with the expected DOFs
  * the robot rests on the floor under PhysX 5 instead of sinking or toppling
  * the 3-DOF base and the 7-DOF arm both track commands

Run:
    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        scripts/gb10/isaac/check_isaac_hospital.py            # add --gui to watch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Isaac Sim's boot log goes straight to the fd; without this the script's own
# output sits in Python's block buffer until exit, so a run that stalls looks
# like it produced nothing.
sys.stdout.reconfigure(line_buffering=True)

# Both assets are resolved through Isaac Sim's own asset root rather than
# hardcoded, so they come from wherever /persistent/isaac/asset_root/default
# points. On this box that is the local pack at /home/dell/AI/assets/isaac-5.1;
# out of the box it is an S3 URL, which would pull ~GB over the network and
# breaks the air-gap requirement. See ASSET_ROOT_FALLBACK below.
HOSPITAL_RELATIVE = "/Isaac/Environments/Hospital/hospital.usd"
ROBOT_RELATIVE = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
ASSET_ROOT_FALLBACK = "/home/dell/AI/assets/isaac-5.1"
# Clearest point on the hospital floor: 2.75 m to the nearest prim occupying
# the robot's height band (z 0.15..1.6), found by intersecting floor-tile
# centres against the bounds of all 785 such prims. Picking a tile centre
# without that clearance test lands the robot inside a prop, where contact
# jams the base joints and nothing responds to commands.
SPAWN = (7.81, 8.25, 0.05)

parser = argparse.ArgumentParser()
parser.add_argument("--assets-root", default=None,
                    help="Override Isaac's asset root for this run")
parser.add_argument("--hospital", default=None, help="Override the hospital USD")
parser.add_argument("--robot", default=None, help="Override the robot USD")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--spawn", type=float, nargs=3, default=list(SPAWN))
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f' -- {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


import carb  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

ASSET_ROOT_SETTING = "/persistent/isaac/asset_root/default"
settings = carb.settings.get_settings()
if args.assets_root:
    settings.set(ASSET_ROOT_SETTING, args.assets_root)

print("Asset root")
configured = settings.get(ASSET_ROOT_SETTING)
if configured and configured.startswith(("http", "omniverse://")):
    # A cloud root means every asset streams over the network, which the event
    # is explicitly air-gapped against. Fall back to the local pack.
    print(f"  asset root is remote ({configured}); falling back to the local pack")
    settings.set(ASSET_ROOT_SETTING, ASSET_ROOT_FALLBACK)

assets_root = get_assets_root_path(skip_check=True)
check("asset root is local (air-gapped)",
      bool(assets_root) and not assets_root.startswith(("http", "omniverse://")),
      assets_root)

hospital_usd = args.hospital or f"{assets_root}{HOSPITAL_RELATIVE}"
robot_usd = args.robot or f"{assets_root}{ROBOT_RELATIVE}"
check("hospital asset present in the pack", Path(hospital_usd).is_file(), hospital_usd)
check("robot asset present in the pack", Path(robot_usd).is_file(), robot_usd)
args.robot = robot_usd

print(f"\nHospital: {hospital_usd}\nRobot:    {robot_usd}\n")
context = omni.usd.get_context()
context.open_stage(hospital_usd)
stage = context.get_stage()

print("Environment")
check("hospital stage opens", stage is not None)
if stage is None:
    simulation_app.close()
    raise SystemExit(1)

up_axis = UsdGeom.GetStageUpAxis(stage)
check("hospital is Z-up (PhysX default gravity)", up_axis == "Z", up_axis)
check("metersPerUnit == 1", UsdGeom.GetStageMetersPerUnit(stage) == 1.0)

root = stage.GetDefaultPrim()
bound = UsdGeom.Imageable(root).ComputeWorldBound(
    Usd.TimeCode.Default(), UsdGeom.Tokens.default_
).ComputeAlignedBox()
size = bound.GetSize()
check("hospital has real extent", not bound.IsEmpty() and size[0] > 10,
      f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m")

# If the hospital's relative ./Props references do not resolve you get an empty
# shell with no walls and the robot drives through the building, so count the
# geometry rather than trusting that the stage "opened".
props = [p for p in stage.Traverse() if p.GetName().startswith("Geo_")]
check("prop/wall geometry resolved", len(props) > 100, f"{len(props)} Geo_ prims")
colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
check("environment has colliders", len(colliders) > 0, f"{len(colliders)} prims")

# The shipped pack was a partial download missing the /NVIDIA tree, so the dome
# light's sky HDR silently failed to load. Catch any such gap by resolving every
# asset-valued attribute on the stage.
unresolved: list[str] = []
resolved = 0
for prim in stage.Traverse():
    for attribute in prim.GetAttributes():
        if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
            continue
        value = attribute.Get()
        if not value:
            continue
        if value.resolvedPath and Path(value.resolvedPath).exists():
            resolved += 1
        else:
            unresolved.append(value.path)
check("every referenced asset resolves", not unresolved,
      f"{resolved} resolved" + (f", MISSING {sorted(set(unresolved))}" if unresolved else ""))

print("\nPhysX 5 configuration (spec §5.2)")
scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
physx_scene.CreateSolverTypeAttr("TGS")
physx_scene.CreateEnableGPUDynamicsAttr(True)

material = UsdPhysics.MaterialAPI.Apply(
    UsdGeom.Scope.Define(stage, "/physicsMaterial").GetPrim()
)
material.CreateStaticFrictionAttr(0.8)
material.CreateDynamicFrictionAttr(0.65)
material.CreateRestitutionAttr(0.0)
check("TGS solver + friction material authored",
      physx_scene.GetSolverTypeAttr().Get() == "TGS",
      "static 0.8 / dynamic 0.65 / restitution 0.0")

print("\nRobot spawn")
robot_path = "/World_Robot"
robot_prim = stage.DefinePrim(robot_path, "Xform")
robot_prim.GetReferences().AddReference(args.robot)

# ridgeback_franka.usd already authors translate/orient/scale on its default
# prim, and those ops compose through onto this prim -- AddTranslateOp() would
# raise "already exists in xformOpOrder". Reuse the composed op instead.
xformable = UsdGeom.Xformable(robot_prim)
existing = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
if "xformOp:translate" in existing:
    existing["xformOp:translate"].Set(Gf.Vec3d(*args.spawn))
else:
    xformable.AddTranslateOp().Set(Gf.Vec3d(*args.spawn))
check("robot reference composes", robot_prim.IsValid() and bool(robot_prim.GetChildren()))

# ridgeback_franka.usd models its mobile base as three dummy joints hanging off
# a `world` link, and ships no joint anchoring that link to the static frame.
# Left as-is the articulation is floating-base, and driving dummy_base_x slides
# the *anchor* backwards while base_link and the arm stay exactly where they
# are -- the robot appears to accept every command and never goes anywhere.
# Pinning `world` makes the chain behave like a mobile base.
anchor = UsdPhysics.FixedJoint.Define(stage, "/base_anchor")
anchor.CreateBody1Rel().SetTargets([f"{robot_path}/world"])
check("articulation anchored to the static frame", bool(anchor.GetPrim()))

from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402

simulation_context = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60)
simulation_context.initialize_physics()
# PhysicsContext re-derives gravity from the stage up axis on init; the hospital
# is Z-up so the default is already correct, but assert it rather than assume.
direction, magnitude = simulation_context.get_physics_context().get_gravity()
check("gravity is -Z at 9.81", list(direction) == [0.0, 0.0, -1.0]
      and abs(magnitude - 9.81) < 1e-3, f"{list(direction)} x {magnitude:.2f}")

simulation_context.play()
robot = SingleArticulation(robot_path)
robot.initialize(simulation_context.physics_sim_view)

print("\nArticulation")
names = list(robot.dof_names)
check("articulation initialises", robot.num_dof > 0, f"{robot.num_dof} DOF")
base_dofs = [n for n in names if n.startswith("dummy_base")]
arm_dofs = [n for n in names if n.startswith("panda_joint")]
finger_dofs = [n for n in names if "finger" in n]
check("3-DOF planar base present", len(base_dofs) == 3, ", ".join(base_dofs))
check("7-DOF Franka arm present", len(arm_dofs) == 7, ", ".join(arm_dofs))
check("gripper present", len(finger_dofs) == 2, ", ".join(finger_dofs))

from isaacsim.core.prims import SingleRigidPrim  # noqa: E402

# The articulation root (/panda_mobile -> the `world` link) is a FIXED base;
# the chain is world -> dummy_base_x -> dummy_base_y -> base_link, so the part
# that actually moves through the hospital is base_link. Reading the root's
# pose reports the anchor and never changes.
base = SingleRigidPrim(f"{robot_path}/base_link")
base.initialize(simulation_context.physics_sim_view)


def base_pose() -> tuple[np.ndarray, float]:
    """World position of base_link, and its tilt from the spawn orientation."""
    position, orientation = base.get_world_pose()
    _w, x, y, _z = (float(v) for v in orientation)
    # R[2][2] = cos(angle between the body Z axis and world +Z)
    upright = max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))
    return np.asarray([float(v) for v in position]), float(np.degrees(np.arccos(upright)))


print("\nSettling under PhysX")
simulation_context.step(render=False)
spawn_position, spawn_tilt = base_pose()
for _ in range(120):  # 2 s
    simulation_context.step(render=False)
position, tilt = base_pose()
drop = float(spawn_position[2] - position[2])
check("base holds its height (does not sink or launch)", abs(drop) < 0.05,
      f"z {spawn_position[2]:+.3f} -> {position[2]:+.3f} m")
# Compare against the spawn orientation: the asset authors its own orient, so
# an absolute "is w~1" test would flag a correctly-standing robot.
check("robot stays upright", abs(tilt - spawn_tilt) < 15.0,
      f"tilt {spawn_tilt:.1f} -> {tilt:.1f} deg")

from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

# Report the drive gains: a stiffness of 0 means position targets are ignored
# and the RL action space has to be effort or velocity instead.
properties = robot.dof_properties
stiffness = {name: float(properties["stiffness"][index])
             for index, name in enumerate(names)}
damping = {name: float(properties["damping"][index])
           for index, name in enumerate(names)}
notes.append(
    "drive stiffness -- base "
    f"{stiffness.get('dummy_base_prismatic_x_joint', 0):.0f}, arm "
    f"{stiffness.get('panda_joint1', 0):.0f}, gripper "
    f"{stiffness.get('panda_finger_joint1', 0):.0f}"
)

print("\nBase drive (holonomic, 3 DOF, via PD drives)")
start = base_pose()[0]
x_index = robot.get_dof_index("dummy_base_prismatic_x_joint")
targets = robot.get_joint_positions().copy()
commanded = targets[x_index] + 1.0
targets[x_index] = commanded
robot.apply_action(ArticulationAction(joint_positions=targets))
for _ in range(180):  # 3 s
    simulation_context.step(render=False)
moved = float(robot.get_joint_positions()[x_index])
travelled = float(np.linalg.norm(base_pose()[0][:2] - start[:2]))
check(
    "base X drive tracks a 1 m position target",
    abs(moved - commanded) < 0.05,
    f"commanded {commanded:+.2f}, reached {moved:+.2f} "
    f"(stiffness {stiffness.get('dummy_base_prismatic_x_joint', 0):.0f})",
)
check("base_link translates in world", travelled > 0.5, f"{travelled:.2f} m")

# The planar joints are position-driven at ~1e7 stiffness. That is stiff enough
# to overpower contact, so the base can bulldoze through walls instead of being
# stopped by them. Nav has to learn to avoid geometry -- the physics will not
# enforce it for free. Command a target well outside the room and see.
print("\nObstacle response (does the base stop at walls?)")
wall_target = targets.copy()
wall_target[x_index] = commanded + 10.0
robot.apply_action(ArticulationAction(joint_positions=wall_target))
before = base_pose()[0]
for _ in range(300):  # 5 s
    simulation_context.step(render=False)
after = base_pose()[0]
pushed = float(np.linalg.norm(after[:2] - before[:2]))
blocked = pushed < 9.0
notes.append(
    f"commanded the base 10 m through the building: it moved {pushed:.2f} m -- "
    + ("contact stopped it, so walls do constrain the base"
       if blocked else
       "it passed through walls; the position drive overpowers contact, so "
       "collision avoidance must come from the nav reward, or lower the base "
       "drive stiffness / switch to velocity control")
)

print("\nArm drive (7 DOF, via PD drives)")
arm_indices = [robot.get_dof_index(name) for name in arm_dofs]
# A standard Franka ready pose; well inside limits.
ready = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
targets = robot.get_joint_positions().copy()
targets[arm_indices] = ready
robot.apply_action(ArticulationAction(joint_positions=targets))
for _ in range(300):  # 5 s to converge and hold against gravity
    simulation_context.step(render=False)
reached = robot.get_joint_positions()[arm_indices]
error = float(np.abs(reached - ready).max())
check(
    "arm drives to and holds the ready pose",
    error < 0.1,
    f"max joint error {error:.4f} rad (stiffness "
    f"{stiffness.get('panda_joint1', 0):.0f}, damping "
    f"{damping.get('panda_joint1', 0):.0f})",
)

print("\nNotes")
notes.append(
    f"spawned at {tuple(round(v, 2) for v in args.spawn)}, "
    f"base_link rests at z={position[2]:+.3f}"
)
notes.append(
    "base is a 3-DOF dummy planar joint (prismatic x/y + revolute z), not four "
    "mecanum wheels -- nav actions map to base velocities directly, no wheel IK"
)
notes.append(
    "the `world` link must be pinned with a FixedJoint (this script adds one). "
    "Unanchored, the articulation is floating-base and base commands slide the "
    "anchor instead of the robot"
)
notes.append(
    "read the robot's world pose from base_link, not from the articulation "
    "root -- the root is the pinned anchor and never moves"
)
notes.append(
    "resolve DOF indices with get_dof_index(name); adding the anchor reordered "
    "them (dummy_base_prismatic_x went 9 -> 0), so hardcoded indices break"
)
for note in notes:
    print(f"  - {note}")

print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL CHECKS PASSED'}\n")
simulation_app.close()
raise SystemExit(1 if failures else 0)
