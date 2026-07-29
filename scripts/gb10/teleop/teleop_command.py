"""Send a one-shot command to a running teleop_hospital session.

Hand teleop drives; this hands the robot to the trained nav policy for a
long trip across the building, then gives control back.

    ./.venv/bin/python scripts/gb10/teleop/teleop_command.py --goto nurse_station
    ./.venv/bin/python scripts/gb10/teleop/teleop_command.py --list

Runs in the repo .venv (or any interpreter) -- it only needs stdlib plus
`simbiote.teleop.action_bridge`, and never touches Isaac directly.
"""

from __future__ import annotations

import argparse
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

from simbiote.sim_env.hospital_map import HOSPITAL_LOCATIONS  # noqa: E402
from simbiote.teleop.action_bridge import DEFAULT_COMMAND_PORT, send_command  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goto", metavar="LOCATION", help="Drive to a named scene-graph location.")
    parser.add_argument("--stop", action="store_true", help="Ask the simulator to stop autonomy.")
    parser.add_argument("--list", action="store_true", help="List known locations and exit.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_COMMAND_PORT)
    args = parser.parse_args()

    if args.list or not (args.goto or args.stop):
        print("known locations:")
        for name, xy in sorted(HOSPITAL_LOCATIONS.items()):
            print(f"  {name:<14} {xy}")
        return 0 if args.list else 2

    if args.goto:
        if args.goto not in HOSPITAL_LOCATIONS:
            print(f"unknown location {args.goto!r}. Known: {', '.join(sorted(HOSPITAL_LOCATIONS))}",
                  file=sys.stderr)
            return 2
        send_command("goto", host=args.host, port=args.port, location=args.goto)
        print(f"sent: goto {args.goto} {HOSPITAL_LOCATIONS[args.goto]}")
        print("watch the Isaac Sim window / its log for progress")
    if args.stop:
        send_command("stop", host=args.host, port=args.port)
        print("sent: stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
