"""Step 4 end to end: natural language -> plan -> the robot drives the hospital.

An operator types "take the tray to room one"; a local LLM turns that into
validated tool calls against the scene graph; the task executor runs them one
skill at a time; each nav skill drives the Ridgeback + Franka through
hospital.usd under PhysX with Step 2's trained policy at the wheel.

The simulator is booted once and handed to every skill, so a multi-step
instruction actually crosses the building -- step two starts where step one
finished.

    # watch it, one instruction
    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        Andrew/hospital_demo.py --gui "go to the supply room"

    # several in sequence, then leave the window up
    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        Andrew/hospital_demo.py --gui --hold \
        "go to the supply room" "now take it to room one"

    # type them yourself
    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        Andrew/hospital_demo.py --gui --interactive

Defaults to the local OpenAI-compatible server (vLLM on :8000). Add
`--llm fake` to run the deterministic rule-based planner with no model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("instructions", nargs="*", help="natural-language commands")
parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
parser.add_argument("--interactive", action="store_true", help="read instructions from stdin")
parser.add_argument("--hold", action="store_true", help="keep the window up after the last command")
parser.add_argument("--llm", default="openai-compat", choices=["openai-compat", "fake"])
parser.add_argument(
    "--profile",
    default=None,
    help="LLM profile name (default: SIMBIOTE_LLM_PROFILE, else qwen3-8b)",
)
parser.add_argument("--nav-checkpoint", default=str(REPO / "checkpoints" / "nav_bc.pt"))
parser.add_argument("--scene", default=str(REPO / "simbiote" / "fixtures" / "hospital_scene_graph.json"))
parser.add_argument("--stage", default=str(REPO / "stage"))
args = parser.parse_args()

if not args.instructions and not args.interactive:
    parser.error("give at least one instruction, or pass --interactive")

# Point at the vLLM container on :8000 unless the operator has already chosen
# something. The profile table's default is Ollama on :11434, which is not what
# is running on this box -- left alone the planner silently degrades to the
# rule-based fallback and the demo looks like it used the model when it did not.
#
# The timeout matters as much as the URL: qwen3 is a thinking model and emits
# its reasoning before the tool call, so at ~15 tok/s a three-step plan takes
# ~45 s. The profile's 60 s is marginal and the default 180 s profile is not
# the one selected here.
import os  # noqa: E402

os.environ.setdefault("SIMBIOTE_LLM_URL", "http://localhost:8000/v1")
os.environ.setdefault("SIMBIOTE_LLM_MODEL", "qwen3-8b")
os.environ.setdefault("SIMBIOTE_LLM_TIMEOUT", "300")

# Boot the simulator before anything else touches Isaac's modules.
from simbiote.sim_env.isaac_nav import IsaacHospital  # noqa: E402

print("booting Isaac Sim and loading hospital.usd ...")
hospital = IsaacHospital(headless=not args.gui, checkpoint=args.nav_checkpoint)
print(f"  robot at {tuple(round(v, 2) for v in hospital.base_xy())}")
print(f"  destinations: {', '.join(sorted(hospital.locations))}\n")

from simbiote.agentic.agentic_session import run_session  # noqa: E402
from simbiote.agentic.llm_backend import describe_backend, make_backend  # noqa: E402
from simbiote.agentic.robot_tools import IsaacBackend  # noqa: E402
from simbiote.agentic.scene_query import load_scene  # noqa: E402

scene = load_scene(args.scene)
robot = IsaacBackend(
    nav_checkpoint=args.nav_checkpoint,
    grasp_checkpoint=str(REPO / "checkpoints" / "grasp_bc.pt"),
    hospital=hospital,
)

llm = make_backend(args.llm, scene, profile=args.profile)
print(f"planner: {describe_backend(llm)}\n")


def run(instruction: str) -> bool:
    print(f'> "{instruction}"')
    before = hospital.base_xy()
    result = run_session(instruction, scene, llm, robot, stage=args.stage)

    if result.error:
        print(f"  could not plan: {result.error}\n")
        return False

    plan = " -> ".join(
        f"{call.tool}({', '.join(str(v) for v in call.args.values())})"
        for call in result.calls
    )
    print(f"  plan: {plan}")
    if result.degraded:
        print("  (planned by the rule-based fallback -- the model was unreachable)")

    for step in result.report.steps:
        print(f"    {step.tool}: {step.status.value} ({step.duration_s:.1f}s)")

    after = hospital.base_xy()
    print(
        f"  robot {tuple(round(v, 2) for v in before)} -> "
        f"{tuple(round(v, 2) for v in after)}   "
        f"{'OK' if result.ok else 'FAILED'}\n"
    )
    return result.ok


ok = 0
total = 0
for instruction in args.instructions:
    total += 1
    ok += run(instruction)

if args.interactive:
    print("Type an instruction, or 'quit'.\n")
    while True:
        try:
            line = input("instruction> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break
        total += 1
        ok += run(line)

print(f"{ok}/{total} instructions completed")

if args.hold and args.gui:
    print("holding the window open -- close it to exit")
    while hospital._app.is_running():
        hospital.spin()

hospital.close()
raise SystemExit(0 if ok == total else 1)
