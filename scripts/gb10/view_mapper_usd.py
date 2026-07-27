"""Open a mapper .usd in the Isaac Sim GUI and leave it up for inspection.

check_mapper_usd.py asserts the contract and exits; this just shows you the
scene. Physics is left stopped so the point cloud stays where the scan put it.

    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        scripts/gb10/view_mapper_usd.py artifacts/mapper/out/gb10_walkthrough.usda
"""

from __future__ import annotations

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("usd", type=Path)
parser.add_argument(
    "--point-size",
    type=float,
    default=0.03,
    help="Rendered LiDAR point width in metres; bump it if the cloud looks sparse",
)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": False, "width": 1600, "height": 1000})

import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

usd_path = args.usd.resolve()
print(f"\nOpening {usd_path} in the Isaac Sim viewport\n")
context = omni.usd.get_context()
context.open_stage(str(usd_path))
stage = context.get_stage()

geometry = stage.GetPrimAtPath("/FactoryFlowScene/ReconstructedGeometry")
points = UsdGeom.Points(geometry)
count = len(points.GetPointsAttr().Get() or [])

# Fatter points read better on screen than the 2 cm the exporter authors.
widths = points.GetWidthsAttr()
if not widths:
    widths = points.CreateWidthsAttr()
widths.Set([args.point_size])
points.SetWidthsInterpolation(UsdGeom.Tokens.constant)

# The scan has no lighting of its own, and an unlit point cloud renders black.
if not stage.GetPrimAtPath("/FactoryFlowScene/ViewerLight"):
    light = UsdLux.DomeLight.Define(stage, "/FactoryFlowScene/ViewerLight")
    light.CreateIntensityAttr(1000.0)

bound = UsdGeom.Imageable(geometry).ComputeWorldBound(
    Usd.TimeCode.Default(), UsdGeom.Tokens.default_
)
box = bound.ComputeAlignedBox()
centre, size = box.GetMidpoint(), box.GetSize()
print(f"  {count:,} LiDAR points")
print(f"  room {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m, centred at "
      f"({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f})")

# Drop a camera outside the cloud looking back at its centre, so the viewport
# opens on the room instead of inside a wall.
camera = UsdGeom.Camera.Define(stage, "/FactoryFlowScene/ViewerCamera")
reach = max(size[0], size[2]) * 0.9
eye = Gf.Vec3d(centre[0] + reach, centre[1] + size[1] * 0.8, centre[2] + reach)
transform = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(*centre), Gf.Vec3d(0, 1, 0))
camera.AddTransformOp().Set(transform.GetInverse())
camera.CreateFocalLengthAttr(18.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

try:
    import omni.kit.viewport.utility as viewport_utils

    viewport_utils.get_active_viewport().camera_path = (
        "/FactoryFlowScene/ViewerCamera"
    )
except Exception as exc:  # noqa: BLE001 - viewport is a convenience, not a contract
    print(f"  (could not bind the framing camera: {exc})")

labelled = [
    child.GetName()
    for child in stage.GetPrimAtPath("/FactoryFlowScene").GetChildren()
    if child.GetAttribute("factoryflow:kind")
]
print(
    f"\n  Navigate: {len(labelled)} labelled prim(s) sit in the Stage tree under\n"
    f"  /FactoryFlowScene -- {', '.join(labelled)}.\n"
    "  Close the window to exit.\n"
)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
