"""Measure whether the nav policy actually traverses hospital.usd.

Runs every ordered pair of scene-graph locations and reports success, path
efficiency, and the closest the base came to real geometry. `--controller
pursuit` swaps the policy for a straight-at-the-carrot reference so the policy
number can be read against something.

    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        Suraj/eval_hospital_nav.py --checkpoint checkpoints/nav_bc.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/nav_bc.pt")
parser.add_argument("--controller", default="policy", choices=["policy", "pursuit"])
parser.add_argument("--pairs", type=int, default=0, help="Limit to the first N pairs")
parser.add_argument("--lookahead", type=float, default=None)
parser.add_argument("--speed-scale", type=float, default=None)
parser.add_argument("--goal-threshold", type=float, default=None)
parser.add_argument("--timeout", type=float, default=None)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--out", default=None)
args = parser.parse_args()

from simbiote.sim_env.hospital_map import HOSPITAL_LOCATIONS  # noqa: E402
from simbiote.sim_env.isaac_nav import IsaacHospital, NavTuning  # noqa: E402

tuning = NavTuning()
for name in ("lookahead", "speed_scale", "goal_threshold", "timeout"):
    value = getattr(args, name if name != "timeout" else "timeout")
    if value is not None:
        setattr(tuning, "timeout_s" if name == "timeout" else name, value)

print(f"controller={args.controller} checkpoint={args.checkpoint}")
print(f"tuning: lookahead={tuning.lookahead} speed_scale={tuning.speed_scale} "
      f"goal_threshold={tuning.goal_threshold} timeout={tuning.timeout_s}")

hospital = IsaacHospital(
    headless=not args.gui,
    checkpoint=args.checkpoint,
    tuning=tuning,
    controller=args.controller,
)
print(f"\nbase axes world<-joint:\n{hospital.world_from_joint.round(3)}")
print(f"spawned at {tuple(round(v, 2) for v in hospital.base_xy())}\n")

pairs = list(itertools.permutations(sorted(HOSPITAL_LOCATIONS), 2))
if args.pairs:
    pairs = pairs[: args.pairs]

results = []
for start, goal in pairs:
    hospital.teleport(HOSPITAL_LOCATIONS[start])
    settled = hospital.base_xy()
    result = hospital.navigate_to(goal)
    row = result.to_dict()
    row["start"] = start
    row["start_xy"] = [round(v, 2) for v in settled]
    results.append(row)
    efficiency = result.travelled / result.path_length if result.path_length else 0.0
    print(
        f"  {start:14s} -> {goal:14s} "
        f"{'OK  ' if result.success else 'FAIL'} "
        f"dist {result.goal_distance:5.2f} m  path {result.path_length:5.1f} m  "
        f"travelled {result.travelled:5.1f} m ({efficiency:4.2f}x)  "
        f"clr {result.min_clearance:4.2f} m  {result.steps:4d} steps  "
        f"{result.duration_s:5.1f}s  {result.reason}"
    )

wins = sum(1 for r in results if r["success"])
print(f"\n{wins}/{len(results)} traversals succeeded")
if wins:
    ok = [r for r in results if r["success"]]
    print(f"  mean efficiency {sum(r['travelled_m'] / max(r['path_length_m'], 0.1) for r in ok) / len(ok):.2f}x")
    print(f"  min clearance over successes {min(r['min_clearance_m'] for r in ok):.2f} m")
for row in results:
    if not row["success"]:
        print(f"  FAILED {row['start']} -> {row['location_id']}: {row['reason']} "
              f"(stopped {row['goal_distance']} m out)")

if args.out:
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")

hospital.close()
raise SystemExit(0 if wins == len(results) else 1)
