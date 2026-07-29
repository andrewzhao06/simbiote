"""Open a mapper-produced .usd/.usda in Isaac Sim headlessly and check the Step 2 contract.

This is Teammate 1's §4.5 acceptance test and the thing Step 2's `validate_map()`
is coded against: the stage opens in Isaac Sim, the scale is right, there is at
least one navigable-floor region, and at least one graspable object.

Run it through `scripts/gb10/mapper/validate_usd_isaac.sh`, which enforces the memory
budget first. Booting this alongside a resident LLM is what took the box down on
2026-07-26 — GB10 has no separate VRAM, so an Isaac Sim stage and a vLLM server
draw from the same 128 GB pool.

    scripts/gb10/mapper/validate_usd_isaac.sh /home/dell/factoryflow/stage/proxy-demo.usda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RELEASE = "/home/dell/IsaacSim/_build/linux-aarch64/release"
# The lightweight headless experience. Deliberately NOT isaacsim.exp.full.kit --
# "Full" boots the RTX renderer and the whole GUI extension set.
EXPERIENCE = str(Path(RELEASE) / "apps" / "isaacsim.exp.base.python.kit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", help="Path to the .usd/.usda produced by the mapper")
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Write the machine-readable report here as well as stdout",
    )
    return parser.parse_args()


def check_stage(stage) -> dict:
    """Collect the Step 2 contract facts from an opened stage."""
    from pxr import UsdGeom, UsdPhysics

    report: dict = {"errors": [], "warnings": []}

    default_prim = stage.GetDefaultPrim()
    report["default_prim"] = default_prim.GetPath().pathString if default_prim else None
    if not default_prim:
        report["errors"].append("stage has no defaultPrim; Step 2 references it by path")

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    report["meters_per_unit"] = meters_per_unit
    if abs(meters_per_unit - 1.0) > 1e-6:
        report["errors"].append(
            f"metersPerUnit is {meters_per_unit}, expected 1.0 (scene must be in meters)"
        )

    report["up_axis"] = UsdGeom.GetStageUpAxis(stage)

    navigable, graspable, collidable = [], [], []
    prim_count = 0

    for prim in stage.Traverse():
        prim_count += 1
        path = prim.GetPath().pathString

        nav_attr = prim.GetAttribute("factoryflow:is_navigable")
        if nav_attr and nav_attr.Get():
            navigable.append(path)

        grasp_attr = prim.GetAttribute("factoryflow:is_graspable")
        if grasp_attr and grasp_attr.Get():
            graspable.append(path)

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collidable.append(path)

    report["prim_count"] = prim_count
    report["navigable_floor_prims"] = navigable
    report["graspable_prims"] = graspable
    report["collision_prims"] = collidable

    if not navigable:
        report["errors"].append(
            "no prim carries factoryflow:is_navigable=true -- Step 2 has no nav costmap source"
        )
    if not graspable:
        report["errors"].append(
            "no prim carries factoryflow:is_graspable=true -- Step 2 has no grasp target"
        )

    # A floor with no collider is navigable in name only; the robot falls through it.
    for path in navigable:
        if path not in collidable:
            report["errors"].append(f"navigable prim {path} has no PhysicsCollisionAPI")
    for path in graspable:
        if path not in collidable:
            report["warnings"].append(f"graspable prim {path} has no PhysicsCollisionAPI")

    schema_attr = default_prim.GetAttribute("factoryflow:schemaVersion") if default_prim else None
    report["schema_version"] = schema_attr.Get() if schema_attr else None

    proxy_attr = default_prim.GetAttribute("factoryflow:proxy") if default_prim else None
    report["proxy"] = bool(proxy_attr.Get()) if proxy_attr else False
    if report["proxy"]:
        report["warnings"].append(
            "stage is a PROXY scene (no reconstructed geometry) -- valid for contract "
            "testing, not the demo path"
        )

    return report


def main() -> int:
    args = parse_args()
    usd_path = str(Path(args.usd).resolve())
    if not Path(usd_path).is_file():
        print(f"FAIL: no such file: {usd_path}", file=sys.stderr)
        return 2

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True, "experience": EXPERIENCE})

    import omni.usd

    context = omni.usd.get_context()
    # Kit 6.0 returns a bare bool here; older Kit returns (ok, error). Handle both.
    result = context.open_stage(usd_path)
    if isinstance(result, tuple):
        ok, error = result[0], result[1]
    else:
        ok, error = bool(result), "open_stage returned False"
    if not ok:
        print(f"FAIL: Isaac Sim could not open the stage: {error}", file=sys.stderr)
        sim_app.close()
        return 1

    stage = context.get_stage()
    report = check_stage(stage)
    report["usd_path"] = usd_path

    # Everything below must run BEFORE sim_app.close(): the app is launched with
    # --/app/fastShutdown=True, so close() tears the process down without returning.
    print("\n=== Isaac Sim USD validation ===")
    print(f"stage            : {report['usd_path']}")
    print(f"defaultPrim      : {report['default_prim']}")
    print(f"schemaVersion    : {report['schema_version']}")
    print(f"metersPerUnit    : {report['meters_per_unit']}")
    print(f"upAxis           : {report['up_axis']}")
    print(f"prims            : {report['prim_count']}")
    print(
        f"navigable floor  : {len(report['navigable_floor_prims'])} "
        f"{report['navigable_floor_prims']}"
    )
    print(f"graspable objects: {len(report['graspable_prims'])} {report['graspable_prims']}")
    print(f"colliders        : {len(report['collision_prims'])}")

    for warning in report["warnings"]:
        print(f"WARN  {warning}")
    for error in report["errors"]:
        print(f"ERROR {error}")

    if args.json_out:
        with Path(args.json_out).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nreport written to {args.json_out}")

    if report["errors"]:
        print(f"\nRESULT: FAIL ({len(report['errors'])} error(s))", flush=True)
        code = 1
    else:
        print("\nRESULT: PASS -- stage satisfies the Step 2 load contract", flush=True)
        code = 0

    sim_app.close()
    # Only reached if close() returned; otherwise the exit code comes from Kit and
    # the wrapper falls back to the RESULT line / JSON report.
    return code


if __name__ == "__main__":
    sys.exit(main())
