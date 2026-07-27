"""Open the hospital sim and take natural-language commands from another shell.

Run this in a terminal with a display. It boots Isaac Sim, spawns the
Ridgeback + Franka in hospital.usd, and then sits in its main loop watching a
command queue. Anything sent with `Andrew/say.py` gets parsed by the local LLM
into tool calls and executed against the live robot -- you watch it drive.

    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        Andrew/hospital_server.py

Then, from any other shell:

    python3 Andrew/say.py "go to the nurse station"
    python3 Andrew/say.py "take the tray to room one"

Add `--headless` to run it without a window (useful for checking the wiring
over ssh). Ctrl-C here, or closing the window, shuts it down.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="no window")
parser.add_argument("--llm", default="openai-compat", choices=["openai-compat", "fake"])
parser.add_argument("--profile", default=None, help="LLM profile name")
parser.add_argument("--nav-checkpoint", default=str(REPO / "checkpoints" / "nav_bc.pt"))
parser.add_argument("--scene", default=str(REPO / "simbiote" / "fixtures" / "hospital_scene_graph.json"))
parser.add_argument("--stage", default=str(REPO / "stage"))
parser.add_argument("--control-root", default=None, help="command queue directory")
parser.add_argument(
    "--light",
    action="store_true",
    help="low-graphics mode: smaller window, cheap RTX, roof and ceiling lights "
         "hidden. Colliders are untouched, so the robot drives identically.",
)
parser.add_argument(
    "--minimal",
    action="store_true",
    help="draw only the building shell (walls/floors/doors) -- every prop is "
         "hidden. Colliders stay, so navigation is identical.",
)
parser.add_argument(
    "--pace",
    type=float,
    default=0.02,
    help="seconds to yield after each drawn frame; raise if the window feels "
         "unresponsive (default 0.02)",
)
parser.add_argument(
    "--render-interval",
    type=float,
    default=0.25,
    help="min seconds between drawn frames. Raise it (0.5, 1.0) to trade a "
         "jittery picture for a robot that keeps moving (default 0.25 = 4 fps)",
)
parser.add_argument(
    "--flat",
    action="store_true",
    help="drive the same planned route on an empty floor with destination "
         "markers instead of inside hospital.usd. Routing is identical (the "
         "map is still built from the hospital); there is just no geometry to "
         "collide with.",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="scale on the policy's commanded velocity (0.3 = slow and steady)",
)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

# The profile table defaults to Ollama on :11434; what runs on this box is vLLM
# on :8000. Left alone the planner silently degrades to the rule-based fallback
# and the demo looks like it used the model when it did not. qwen3 is a
# thinking model at ~15 tok/s, so a three-step plan needs well over 60 s.
import os  # noqa: E402

os.environ.setdefault("SIMBIOTE_LLM_URL", "http://localhost:8000/v1")
os.environ.setdefault("SIMBIOTE_LLM_MODEL", "qwen3-8b")
# 90 s, not longer: OpenAICompatBackend retries transport failures with
# backoff, so a 300 s timeout turns one unreachable model into a ~15 minute
# hang with the robot sitting still and the operator with no idea why. Failing
# fast degrades to the rule-based planner, which produces the same plans.
os.environ.setdefault("SIMBIOTE_LLM_TIMEOUT", "90")

from simbiote.sim_env.isaac_nav import IsaacHospital  # noqa: E402

print("booting Isaac Sim and loading hospital.usd (first run cooks colliders) ...")
from simbiote.sim_env.isaac_nav import NavTuning  # noqa: E402

tuning = NavTuning()
tuning.speed_scale = args.speed

hospital = IsaacHospital(
    headless=args.headless,
    tuning=tuning,
    checkpoint=args.nav_checkpoint,
    width=args.width,
    height=args.height,
    low_graphics=args.light or args.minimal,
    hide_roof=args.light or args.minimal,
    minimal_scene=args.minimal,
    pace=args.pace,
    render_interval=args.render_interval,
    progress_every=30,
    flat=args.flat,
)

from simbiote.agentic.agentic_session import run_session  # noqa: E402
from simbiote.agentic.control_queue import DEFAULT_ROOT, ControlQueue  # noqa: E402
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

queue = ControlQueue(Path(args.control_root) if args.control_root else DEFAULT_ROOT)
queue.reset()


def announce(busy: bool, instruction: str = "") -> None:
    x, y = hospital.base_xy()
    queue.publish_status(
        ready=True,
        busy=busy,
        instruction=instruction,
        robot_xy=[round(x, 2), round(y, 2)],
        locations=sorted(hospital.locations),
        planner=describe_backend(llm),
        headless=args.headless,
    )


announce(busy=False)
print(f"\n  robot at {tuple(round(v, 2) for v in hospital.base_xy())}")
print(f"  destinations: {', '.join(sorted(hospital.locations))}")
print(f"  planner: {describe_backend(llm)}")
print(f"  queue: {queue.root}")
print('\nReady. From another shell:  python3 Andrew/say.py "go to the nurse station"\n')


def handle(seq: str, instruction: str) -> None:
    print(f'> [{seq}] "{instruction}"')
    announce(busy=True, instruction=instruction)
    before = hospital.base_xy()
    started = time.time()

    try:
        result = run_session(instruction, scene, llm, robot, stage=args.stage)
    except Exception as exc:  # noqa: BLE001 - one bad command must not kill the sim
        print(f"  error: {exc}\n")
        queue.publish_result(seq, {"ok": False, "error": str(exc), "instruction": instruction})
        announce(busy=False)
        return

    after = hospital.base_xy()
    payload = {
        "ok": bool(result.ok),
        "instruction": instruction,
        "session_id": result.session_id,
        "error": result.error,
        "degraded": result.degraded,
        "planner": result.llm,
        "plan": [{"tool": c.tool, "args": c.args} for c in result.calls],
        "steps": [
            {"tool": s.tool, "status": s.status.value, "duration_s": round(s.duration_s, 2),
             "detail": s.detail}
            for s in (result.report.steps if result.report else [])
        ],
        "robot_from": [round(v, 2) for v in before],
        "robot_to": [round(v, 2) for v in after],
        "elapsed_s": round(time.time() - started, 1),
    }
    queue.publish_result(seq, payload)

    if result.error:
        print(f"  could not plan: {result.error}\n")
    else:
        plan = " -> ".join(f"{c.tool}({', '.join(str(v) for v in c.args.values())})" for c in result.calls)
        print(f"  plan: {plan}")
        for step in payload["steps"]:
            print(f"    {step['tool']}: {step['status']} ({step['duration_s']}s)")
        print(
            f"  robot {tuple(payload['robot_from'])} -> {tuple(payload['robot_to'])}   "
            f"{'OK' if result.ok else 'FAILED'}\n"
        )
    announce(busy=False)


try:
    while hospital.is_running():
        pending = queue.next_instruction()
        if pending is None:
            # Idle: keep pumping the app so the window stays interactive, and
            # sleep a little so polling an empty queue is not a busy loop.
            hospital.spin()
            time.sleep(0.05)
            continue

        # The instruction runs on a worker thread, because task_executor puts
        # each skill on one of its own and the simulator work has to come back
        # here to be executed. This thread stays free to service that bridge
        # and to keep the window alive -- which is the whole reason the process
        # used to die the instant a command arrived.
        worker = threading.Thread(target=handle, args=pending, daemon=True)
        worker.start()
        while worker.is_alive():
            if not robot.bridge.pump():
                hospital.spin()
                time.sleep(0.01)
        worker.join()
        # Drain anything queued as the worker finished.
        while robot.bridge.pump():
            pass
except KeyboardInterrupt:
    print("\nshutting down")
finally:
    queue.publish_status(ready=False, busy=False)
    hospital.close()
