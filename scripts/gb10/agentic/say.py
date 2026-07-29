"""Send a natural-language command to the running hospital sim.

`scripts/gb10/agentic/hospital_server.py` has to already be up in another shell. This is a
plain-CPython client -- it never imports Isaac Sim or torch, so it runs under
the system interpreter and returns as soon as the robot finishes.

    python3 scripts/gb10/agentic/say.py "go to the nurse station"
    python3 scripts/gb10/agentic/say.py --status
    python3 scripts/gb10/agentic/say.py "go to room two" "then the supply room"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# These scripts are run by path (often under Isaac Sim's bundled interpreter,
# where the package isn't installed), so put the repo on sys.path. Located by
# walking up to pyproject.toml rather than by a fixed parent count, which
# silently breaks the moment the file moves between script subdirectories.
REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from simbiote.agentic.control_queue import ControlQueue, default_control_root  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("instructions", nargs="*")
parser.add_argument("--status", action="store_true", help="show sim state and exit")
parser.add_argument("--timeout", type=float, default=900.0)
parser.add_argument("--json", action="store_true", help="print the raw result")
parser.add_argument("--control-root", default=None)
args = parser.parse_args()

queue = ControlQueue(Path(args.control_root) if args.control_root else default_control_root())
status = queue.status()

if args.status or not args.instructions:
    if status is None:
        print(
            "sim is not running (no status file). "
            "Start scripts/gb10/agentic/hospital_server.py"
        )
        raise SystemExit(1)
    print(json.dumps(status, indent=2))
    raise SystemExit(0)

if status is None or not status.get("ready"):
    print(
        "sim is not ready. In a terminal with a display, run:\n"
        "  /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh "
        "scripts/gb10/agentic/hospital_server.py",
        file=sys.stderr,
    )
    raise SystemExit(1)

failures = 0
for instruction in args.instructions:
    seq = queue.submit(instruction)
    print(f'> "{instruction}"  (queued as {seq}, waiting for the robot ...)')
    result = queue.await_result(seq, timeout_s=args.timeout)

    if result is None:
        print(f"  timed out after {args.timeout:.0f}s -- is the sim still up?")
        failures += 1
        continue

    if args.json:
        print(json.dumps(result, indent=2))
    elif result.get("error"):
        print(f"  could not plan: {result['error']}")
    else:
        plan = " -> ".join(
            f"{step['tool']}({', '.join(str(v) for v in step['args'].values())})"
            for step in result.get("plan", [])
        )
        print(f"  plan: {plan}")
        if result.get("degraded"):
            print("  (rule-based fallback -- the model server was unreachable)")
        for step in result.get("steps", []):
            print(f"    {step['tool']}: {step['status']} ({step['duration_s']}s)")
        print(
            f"  robot {tuple(result.get('robot_from', []))} -> "
            f"{tuple(result.get('robot_to', []))}   "
            f"{'OK' if result.get('ok') else 'FAILED'}  [{result.get('elapsed_s')}s]"
        )

    if not result.get("ok"):
        failures += 1

raise SystemExit(1 if failures else 0)
