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
        scripts/gb10/agentic/hospital_demo.py --gui "go to the supply room"

    # several in sequence, then leave the window up
    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        scripts/gb10/agentic/hospital_demo.py --gui --hold \
        "go to the supply room" "now take it to room one"

    # pick from a menu of canned commands, over and over
    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        scripts/gb10/agentic/hospital_demo.py --gui --interactive

Defaults to the local OpenAI-compatible server (vLLM on :8000). Add
`--llm fake` to run the deterministic rule-based planner with no model.
"""

from __future__ import annotations

import argparse
import select
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO = Path(__file__).resolve().parents[3]

#: The canned commands `--interactive` offers. Edit this list to change the
#: menu -- it is the only place the demo's instructions are written down.
#:
#: They are phrased the way an operator would say them, not as tool calls: the
#: whole point of the demo is that the LLM turns the sentence into
#: `navigate_to(<id>)`, so pre-resolving them here would skip the step being
#: shown. Both phrases resolve through the scene graph's aliases.
CANNED_COMMANDS = (
    "go to the nurse station",
    "go to the supply room",
)

parser = argparse.ArgumentParser()
parser.add_argument("instructions", nargs="*", help="natural-language commands")
parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
parser.add_argument(
    "--interactive",
    action="store_true",
    help="pick canned commands from a menu, redisplayed after each run",
)
parser.add_argument("--hold", action="store_true", help="keep the window up after the last command")
parser.add_argument("--llm", default="openai-compat", choices=["openai-compat", "fake"])
parser.add_argument(
    "--profile",
    default=None,
    help="LLM profile name (default: SIMBIOTE_LLM_PROFILE, else qwen3-8b)",
)
parser.add_argument(
    "--light",
    action="store_true",
    help="low-graphics mode: cheap RTX, roof and ceiling lights hidden. "
         "Colliders are untouched, so the robot drives identically.",
)
parser.add_argument(
    "--minimal",
    action="store_true",
    help="draw only the building shell -- every prop is hidden. Colliders "
         "stay, so navigation is identical.",
)
parser.add_argument(
    "--pace",
    type=float,
    default=0.02,
    help="seconds to yield after each drawn frame. This is what keeps the "
         "window answering the desktop; at 0 it renders flat out and the WM "
         "reports it as not responding (default 0.02)",
)
parser.add_argument(
    "--render-interval",
    type=float,
    default=0.25,
    help="min seconds between drawn frames. Raise it (0.5, 1.0) to trade a "
         "jerkier picture for a robot that keeps moving (default 0.25 = 4 fps)",
)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
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

print("booting Isaac Sim and loading hospital.usd (first run cooks colliders) ...")
hospital = IsaacHospital(
    headless=not args.gui,
    checkpoint=args.nav_checkpoint,
    width=args.width,
    height=args.height,
    low_graphics=args.light or args.minimal,
    hide_roof=args.light or args.minimal,
    minimal_scene=args.minimal,
    pace=args.pace,
    render_interval=args.render_interval,
    # Print progress while a traversal runs, so a long drive is visibly
    # working rather than indistinguishable from a hang.
    progress_every=30,
)
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


def run_session_pumped(instruction: str):
    """Run one instruction with the main thread left free to serve the sim.

    task_executor puts each skill on a worker thread (so a wedged skill can be
    abandoned on timeout), and IsaacBackend routes every simulator call back to
    the main thread through MainThreadBridge, because Kit aborts the process if
    stepped from anywhere else.

    Calling run_session() straight from the main thread therefore deadlocks:
    the worker blocks in bridge.call() waiting for the main thread to pump the
    queue, while the main thread is blocked inside run_session waiting on the
    worker's future. Nothing moves until the executor's timeout fires. The
    window keeps its last frame, so it looks exactly like a freeze.

    So: run the session on a worker of our own, and spend the main thread
    pumping the bridge and the app -- the same shape hospital_server.py uses.
    """

    box: dict = {}

    def target() -> None:
        try:
            box["result"] = run_session(instruction, scene, llm, robot, stage=args.stage)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            box["error"] = exc

    worker = threading.Thread(target=target, name="session", daemon=True)
    worker.start()
    while worker.is_alive():
        # pump() returns False when the queue is empty; only then is it worth
        # spending time on a frame, so simulator work never waits behind one.
        if not robot.bridge.pump():
            hospital.spin()
            time.sleep(0.01)
    worker.join()
    # Drain anything the worker queued as it finished.
    while robot.bridge.pump():
        pass

    if "error" in box:
        raise box["error"]
    return box["result"]


def run(instruction: str) -> bool:
    print(f'> "{instruction}"')
    before = hospital.base_xy()
    result = run_session_pumped(instruction)

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


def prompt_with_spin(prompt: str) -> str:
    """Read a line of input while keeping the Isaac Sim window alive.

    input() blocks the calling thread, and the main thread is the only one
    allowed to pump Kit. Since this program spends nearly all of its time
    sitting at the menu waiting for a keystroke, using input() means the window
    stops updating for minutes at a stretch -- so the desktop decides the app
    has hung and offers to force quit it. Nothing has actually crashed; the
    render loop is just never given a turn.

    Polling stdin instead lets the gaps between keystrokes go to the renderer.
    select() on stdin is fine for a terminal or a pipe, which is all this is
    ever driven by.
    """

    sys.stdout.write(prompt)
    sys.stdout.flush()
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if ready:
            line = sys.stdin.readline()
            if not line:  # EOF -- treated by the caller as "quit"
                raise EOFError
            return line.strip()
        if args.gui:
            hospital.spin()


def show_menu() -> None:
    # No leading blank line: both callers already end their output with one.
    for number, command in enumerate(CANNED_COMMANDS, start=1):
        print(f"  [{number}] {command}")
    print("  [q] quit")
    print()


if args.interactive:
    while True:
        # Reprinted every pass. A run takes tens of seconds and scrolls the
        # menu off the screen, so an operator who has just watched the robot
        # park needs the choices in front of them again, not scrollback.
        show_menu()
        try:
            choice = prompt_with_spin("choose> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not choice:
            continue
        if choice.lower() in ("q", "quit", "exit"):
            break

        if choice.isdigit():
            # A number is always a menu pick. An out-of-range one is a
            # mis-key, so say so rather than sending "9" to the planner as an
            # instruction and letting it come back as a baffling plan failure.
            index = int(choice)
            if not 1 <= index <= len(CANNED_COMMANDS):
                print(f"  no option {choice} -- pick 1-{len(CANNED_COMMANDS)}, or q to quit\n")
                continue
            instruction = CANNED_COMMANDS[index - 1]
        else:
            # Anything else is still sent through as a free-text instruction,
            # so the demo does not lose the ability to try a phrasing that is
            # not on the list.
            instruction = choice

        total += 1
        ok += run(instruction)

print(f"{ok}/{total} instructions completed")

if args.hold and args.gui:
    print("holding the window open -- close it to exit")
    while hospital._app.is_running():
        hospital.spin()

hospital.close()
raise SystemExit(0 if ok == total else 1)
