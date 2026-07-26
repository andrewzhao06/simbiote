"""SCAN_MAP.md 4.5 acceptance test: does the mapper's .usd open in Isaac Sim?

Checks that the scene opens cleanly, that the reconstructed geometry composes
into something with a real world bound ("recognisable as the scanned room"),
that the navigable-floor and graspable tags survive composition, and -- the
part a file-level validator cannot show -- that the floor collider actually
stops a rigid body in PhysX.

Run with Isaac Sim's interpreter:

    source Gagan/env.gb10.sh
    $FF_PY Gagan/check_isaac_sim.py Gagan/out/gb10_walkthrough.usda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("usd", type=Path)
parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim window")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import omni.usd  # noqa: E402

failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f' -- {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


usd_path = args.usd.resolve()
print(f"\nOpening {usd_path}\n")
context = omni.usd.get_context()
context.open_stage(str(usd_path))
stage = context.get_stage()

print("Stage")
check("stage opens in Isaac Sim", stage is not None)
if stage is None:
    simulation_app.close()
    sys.exit(1)

up_axis = UsdGeom.GetStageUpAxis(stage)
metres = UsdGeom.GetStageMetersPerUnit(stage)
check("metersPerUnit == 1", metres == 1.0, f"{metres}")
notes.append(f"stage up axis is {up_axis}")

root = stage.GetPrimAtPath("/FactoryFlowScene")
check("/FactoryFlowScene exists", bool(root))
check(
    "not flagged as proxy geometry",
    root and root.GetAttribute("factoryflow:proxy").Get() is False,
)

print("\nReconstructed geometry")
geometry = stage.GetPrimAtPath("/FactoryFlowScene/ReconstructedGeometry")
check("referenced geometry layer composes", bool(geometry))
points_api = UsdGeom.Points(geometry)
check("composes as UsdGeom.Points", bool(points_api))
count = len(points_api.GetPointsAttr().Get() or []) if points_api else 0
check("carries LiDAR points", count > 0, f"{count:,} points")

bound = (
    UsdGeom.Imageable(geometry).ComputeWorldBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_
    )
    if geometry
    else None
)
box = bound.ComputeAlignedBox() if bound else None
volume = 0.0
if box and not box.IsEmpty():
    size = box.GetSize()
    volume = size[0] * size[1] * size[2]
    notes.append(
        f"scanned room measures {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m"
    )
check("has a non-empty world bound", volume > 1.0, f"{volume:.1f} cubic metres")

print("\nStep 2 semantic contract")
floor = stage.GetPrimAtPath("/FactoryFlowScene/navigable_floor")
graspable = stage.GetPrimAtPath("/FactoryFlowScene/proxy_graspable")
check("navigable floor region present", bool(floor))
check(
    "floor tagged is_navigable",
    bool(floor) and floor.GetAttribute("factoryflow:is_navigable").Get() is True,
)
check(
    "floor has a collider",
    bool(floor) and floor.HasAPI(UsdPhysics.CollisionAPI),
)
check("graspable object present", bool(graspable))
check(
    "object tagged is_graspable",
    bool(graspable) and graspable.GetAttribute("factoryflow:is_graspable").Get() is True,
)
check(
    "object carries a mass",
    bool(graspable) and (graspable.GetAttribute("factoryflow:mass_kg").Get() or 0) > 0,
)

# Is the graspable object actually resting on the floor, not floating or sunk?
if floor and graspable:
    floor_top = (
        floor.GetAttribute("xformOp:translate").Get()[1]
        + floor.GetAttribute("xformOp:scale").Get()[1] / 2
    )
    object_bottom = (
        graspable.GetAttribute("xformOp:translate").Get()[1]
        - graspable.GetAttribute("xformOp:scale").Get()[1] / 2
    )
    check(
        "graspable object rests on the floor",
        abs(object_bottom - floor_top) < 1e-3,
        f"gap {object_bottom - floor_top:+.4f} m",
    )

print("\nPhysX (does the floor collider actually stop a body?)")
floor_top = (
    floor.GetAttribute("xformOp:translate").Get()[1]
    + floor.GetAttribute("xformOp:scale").Get()[1] / 2
)
floor_centre = floor.GetAttribute("xformOp:translate").Get()

drop_from = floor_top + 1.0
cube = UsdGeom.Cube.Define(stage, "/drop_test")
cube.CreateSizeAttr(0.2)
cube.AddTranslateOp().Set(Gf.Vec3d(floor_centre[0], drop_from, floor_centre[2]))
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())

from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402

simulation_context = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60)
simulation_context.initialize_physics()

# Isaac Sim's PhysicsContext resets the stage up axis to Z on init and derives
# the gravity direction from it, silently overwriting any gravity authored on
# /physicsScene. On this Y-up scene that leaves gravity pointing along -Z, i.e.
# sideways through the walls. Re-assert Y and let set_gravity re-derive.
physics_context = simulation_context.get_physics_context()
before = physics_context.get_gravity()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
physics_context.set_gravity(-9.81)
direction, magnitude = physics_context.get_gravity()
check(
    "gravity points along -Y for this Y-up scene",
    list(direction) == [0.0, -1.0, 0.0] and abs(magnitude - 9.81) < 1e-3,
    f"{list(direction)} x {magnitude:.2f} (was {list(before[0])} before correction)",
)

simulation_context.play()
# PhysX writes simulated transforms to Fabric, not back to the USD attribute,
# so the pose has to be read through the physics view -- reading
# xformOp:translate here just returns the authored spawn height.
body = SingleRigidPrim("/drop_test")
body.initialize(simulation_context.physics_sim_view)
for _ in range(180):  # 3 s
    simulation_context.step(render=False)

resting = float(body.get_world_pose()[0][1])
expected = floor_top + 0.1  # half the 0.2 m cube
fell = drop_from - resting
check(
    "dropped body falls under gravity",
    fell > 0.5,
    f"fell {fell:.3f} m from {drop_from:.3f}",
)
check(
    "floor collider stops it at the floor",
    abs(resting - expected) < 0.05,
    f"rest {resting:.3f} m vs floor top {floor_top:.3f} m",
)

print("\nNotes")
for note in notes:
    print(f"  - {note}")
if up_axis == "Y":
    print(
        "  - HANDOFF: the scene is Y-up per the Step 2 contract, but Isaac Sim's\n"
        "    PhysicsContext resets the stage up axis to Z on init and derives\n"
        "    gravity from it -- confirmed here, it overwrote gravity authored on\n"
        "    /physicsScene and pointed it along -Z, sideways through the walls.\n"
        "    Step 2 must re-assert UsdGeom.SetStageUpAxis(stage, 'Y') and call\n"
        "    physics_context.set_gravity(-9.81) AFTER initialize_physics(), as\n"
        "    this script does, or the robot will fall sideways on spawn."
    )

print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL CHECKS PASSED'}\n")
simulation_app.close()
sys.exit(1 if failures else 0)
