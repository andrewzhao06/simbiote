"""Open the hospital scene with the Ridgeback+Franka spawned, and leave it up.

check_isaac_hospital.py asserts the contract and exits; this is the one you
watch. Same asset resolution, same fixed-base anchor, same clear spawn point,
so what you see is what the checks ran against.

    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        scripts/gb10/view_isaac_hospital.py            # --play to run physics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

HOSPITAL_RELATIVE = "/Isaac/Environments/Hospital/hospital.usd"
ROBOT_RELATIVE = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
ASSET_ROOT_FALLBACK = "/home/dell/AI/assets/isaac-5.1"
SPAWN = (7.81, 8.25, 0.05)

parser = argparse.ArgumentParser()
parser.add_argument("--assets-root", default=None)
parser.add_argument("--spawn", type=float, nargs=3, default=list(SPAWN))
parser.add_argument("--play", action="store_true", help="Start physics running")
parser.add_argument("--no-robot", action="store_true", help="Environment only")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": False, "width": 1600, "height": 1000})

import carb  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

ASSET_ROOT_SETTING = "/persistent/isaac/asset_root/default"
settings = carb.settings.get_settings()
if args.assets_root:
    settings.set(ASSET_ROOT_SETTING, args.assets_root)
configured = settings.get(ASSET_ROOT_SETTING)
if configured and configured.startswith(("http", "omniverse://")):
    print(f"  asset root is remote ({configured}); falling back to the local pack")
    settings.set(ASSET_ROOT_SETTING, ASSET_ROOT_FALLBACK)

assets_root = get_assets_root_path(skip_check=True)
hospital_usd = f"{assets_root}{HOSPITAL_RELATIVE}"
robot_usd = f"{assets_root}{ROBOT_RELATIVE}"
print(f"\nAsset root: {assets_root}\nHospital:   {hospital_usd}\n")

context = omni.usd.get_context()
context.open_stage(hospital_usd)
stage = context.get_stage()

props = sum(1 for p in stage.Traverse() if p.GetName().startswith("Geo_"))
bound = UsdGeom.Imageable(stage.GetDefaultPrim()).ComputeWorldBound(
    Usd.TimeCode.Default(), UsdGeom.Tokens.default_
).ComputeAlignedBox()
size = bound.GetSize()
print(f"  {props} Geo_ prims, {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m")

scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)

robot_path = "/World_Robot"
if not args.no_robot:
    robot_prim = stage.DefinePrim(robot_path, "Xform")
    robot_prim.GetReferences().AddReference(robot_usd)
    # The asset already authors translate/orient/scale; reuse the composed op.
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(robot_prim).GetOrderedXformOps()}
    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(Gf.Vec3d(*args.spawn))
    else:
        UsdGeom.Xformable(robot_prim).AddTranslateOp().Set(Gf.Vec3d(*args.spawn))
    # Without this the articulation is floating-base and the base commands
    # slide the anchor instead of the robot.
    anchor = UsdPhysics.FixedJoint.Define(stage, "/base_anchor")
    anchor.CreateBody1Rel().SetTargets([f"{robot_path}/world"])
    print(f"  robot spawned at {tuple(round(v, 2) for v in args.spawn)}")

# Frame the viewport on the robot rather than the middle of a 76 m building.
# Keep the eye inside the spawn's ~2.75 m clearance radius, or the camera sits
# on the far side of a wall and stares at plasterboard.
if args.no_robot:
    target = bound.GetMidpoint()
    eye = Gf.Vec3d(target[0] - 12.0, target[1] - 12.0, target[2] + 8.0)
else:
    robot_bound = UsdGeom.Imageable(robot_prim).ComputeWorldBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_
    ).ComputeAlignedBox()
    centre = robot_bound.GetMidpoint()
    extent = robot_bound.GetSize()
    print(
        f"  robot bound centre ({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}) "
        f"size {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m"
    )
    target = Gf.Vec3d(centre[0], centre[1], centre[2])
    eye = Gf.Vec3d(target[0] - 2.2, target[1] - 2.2, target[2] + 1.6)
camera = UsdGeom.Camera.Define(stage, "/ViewerCamera")
transform = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
camera.AddTransformOp().Set(transform.GetInverse())
camera.CreateFocalLengthAttr(20.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
try:
    import omni.kit.viewport.utility as viewport_utils

    viewport_utils.get_active_viewport().camera_path = "/ViewerCamera"
except Exception as exc:  # noqa: BLE001 - framing is a convenience
    print(f"  (could not bind the framing camera: {exc})")

if args.play:
    from isaacsim.core.api import SimulationContext

    simulation_context = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60)
    simulation_context.initialize_physics()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    simulation_context.get_physics_context().set_gravity(-9.81)
    simulation_context.play()
    print("  physics running")

print(
    "\n  The robot is under /World_Robot in the Stage tree.\n"
    "  Close the window to exit.\n"
)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
