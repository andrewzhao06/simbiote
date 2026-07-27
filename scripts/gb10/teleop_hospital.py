"""Hand-teleoperate the Ridgeback+Franka through hospital.usd, live in Isaac Sim.

Runs under Isaac Sim's bundled interpreter and receives `RobotAction`s over UDP
from the teleop process (which runs in the repo `.venv` because it needs OpenCV
and WiLoR). See `simbiote/teleop/action_bridge.py` for why it's split.

    # terminal 1 -- the simulator
    /home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
        scripts/gb10/teleop_hospital.py

    # terminal 2 -- camera + hand tracking + the preview window
    ./.venv/bin/python scripts/teleop/run_demo.py --sink udp \
        --camera-url http://<iphone-ip>:4747/video/640x480

Or use `scripts/gb10/run_teleop_hospital.sh`, which starts both.

What the hand controls
----------------------
Pinching switches modes, because one hand can't steer a base and pose an arm
at once without the two fighting (see `ik_bridge.ControlMode`).

* Open hand -> DRIVE. Hand off-centre steers the base: up/down drives
  forward/back, left/right turns. The arm holds its pose.
* Pinched -> MANIPULATE. The base parks, hand height moves the claw up and
  down, and the pincer closes.

Vertical claw motion uses `sim_env/arm_lift.py`, which measures dz/dq for each
arm joint at startup and solves the 1-DOF task with the pseudo-inverse of that
Jacobian row. Full 6-DOF arm posing still isn't wired -- that wants cuRobo
(staged in /home/dell/AI/repos/curobo) or a Lula descriptor this asset pack
doesn't ship.

Safety
------
If the teleop process stops sending -- phone backgrounded, camera unplugged,
window closed -- `ActionReceiver.latest()` starts returning None and the base
is commanded to a full stop rather than continuing on its last velocity.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint", default="checkpoints/nav_hospital.pt",
    help="Trained nav policy to load. Teleop drives the base directly and does not "
         "consult it, but loading it keeps the trained robot in the scene and lets "
         "--controller policy hand back to autonomy later.",
)
parser.add_argument("--headless", action="store_true", help="No Isaac Sim window (for testing).")
parser.add_argument("--port", type=int, default=47800)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--speed-scale", type=float, default=1.0, help="Scale hand-commanded base velocity.")
parser.add_argument(
    "--frame", choices=["body", "world"], default="body",
    help="Interpret the hand's velocity in the robot's own frame (default) or the world's. "
         "`IsaacHospital._apply_velocity` is world-frame because NavEnv's policy emits "
         "world-frame velocities; for hand driving that's wrong -- after turning 90 degrees, "
         "'forward' would still mean world +X. 'body' rotates the command by the base yaw so "
         "forward means wherever the robot is facing.",
)
parser.add_argument("--max-seconds", type=float, default=0.0, help="Auto-exit after N seconds (0 = run forever).")
parser.add_argument(
    "--spawn", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the spawn position in hospital world coordinates.",
)
args = parser.parse_args()

# Imports below deliberately follow argparse: pulling in isaac_nav starts the
# chain that boots SimulationApp, and `--help` shouldn't cost a simulator boot.
from simbiote.robot_iface.actions import GripperState  # noqa: E402
from simbiote.sim_env.hospital_map import SPAWN  # noqa: E402
from simbiote.sim_env.arm_lift import ArmLift  # noqa: E402
from simbiote.sim_env.isaac_nav import CONTROL_HZ, PHYSICS_HZ, IsaacHospital  # noqa: E402
from simbiote.teleop.ik_bridge import WORKSPACE_Z_MAX, WORKSPACE_Z_MIN  # noqa: E402
from simbiote.teleop.action_bridge import ActionReceiver, CommandReceiver  # noqa: E402


def main() -> int:
    print(f"[isaac] booting hospital.usd (headless={args.headless}) -- this takes a few minutes")
    hospital = IsaacHospital(
        headless=args.headless,
        checkpoint=args.checkpoint,
        spawn=tuple(args.spawn) if args.spawn else SPAWN,
        # Render every control tick: this is a headed, human-in-the-loop session,
        # so a smooth window matters more than physics throughput.
        renders_per_control=1,
    )
    print(f"[isaac] spawned at {tuple(round(v, 2) for v in hospital.base_xy())}")

    # Must come after _reset_drive_target so the arm's home pose is read from a
    # settled target vector, not a half-initialised one.
    hospital._reset_drive_target()
    print("[isaac] calibrating arm lift (nudging each joint to measure dz/dq)...")
    arm = ArmLift(hospital)
    print(
        f"[isaac] arm dofs={arm.arm_dofs} fingers={arm.finger_dofs} "
        f"home_z={arm.home_z:.3f} m  dz/dq={arm.sensitivity.round(3).tolist()}"
    )

    receiver = ActionReceiver(host=args.host, port=args.port)
    commands = CommandReceiver(host=args.host, port=args.port + 1)
    print(f"[isaac] listening for teleop on udp://{args.host}:{args.port}")
    print(f"[isaac] listening for commands on udp://{args.host}:{args.port + 1}")
    print(f"[isaac] known locations: {', '.join(sorted(hospital.locations))}")
    print("[isaac] start the teleop process now; Ctrl+C here to stop.\n")

    dt = 1.0 / CONTROL_HZ
    # `_apply_velocity` integrates the drive target forward by `dt` of *sim*
    # time, so the loop has to advance physics by the same amount or the target
    # outruns the robot and pins against `max_target_lead` -- which reads as
    # "commands arrive but the base barely moves". navigate_to() gets this
    # right; matching it here.
    physics_steps = max(int(PHYSICS_HZ / CONTROL_HZ), 1)
    # Clear any target lead built up during boot before taking commands.
    hospital._reset_drive_target()
    started = time.time()
    last_report = 0.0
    was_live = False
    gripper_closed = None
    ticks = 0
    last_cmd = (0.0, 0.0, 0.0)
    mode = "DRIVE"

    try:
        while True:
            for message in commands.poll():
                if message["command"] == "goto":
                    where = message.get("location", "")
                    if where not in hospital.locations:
                        print(f"[isaac] unknown location {where!r}; "
                              f"known: {', '.join(sorted(hospital.locations))}")
                        continue
                    print(f"[isaac] AUTONOMY: navigating to {where} "
                          f"{hospital.locations[where]} -- hand control paused")
                    result = hospital.navigate_to(where)
                    print(f"[isaac] AUTONOMY: {'reached' if result.success else 'FAILED'} "
                          f"{where} ({result.reason}) in {result.duration_s:.1f}s")
                    # Hand frames piled up while navigate_to blocked; drop them
                    # so control resumes on the operator's *current* pose
                    # rather than replaying a stale one.
                    receiver.latest()
                    hospital._reset_drive_target()
                elif message["command"] == "stop":
                    print("[isaac] AUTONOMY: stop")

            action = receiver.latest()

            if action is None:
                # No fresh command: stop the base rather than coast blind.
                hospital._apply_velocity(0.0, 0.0, 0.0, dt)
                last_cmd = (0.0, 0.0, 0.0)
                if was_live:
                    print("[isaac] teleop stream went quiet -- base stopped")
                    was_live = False
            else:
                if not was_live:
                    print("[isaac] teleop stream live -- hand is driving the base")
                    was_live = True
                vx, vy, omega = action.base_velocity
                vx *= args.speed_scale
                vy *= args.speed_scale
                if args.frame == "body":
                    yaw = hospital.base_yaw()
                    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
                    vx, vy = vx * cos_yaw - vy * sin_yaw, vx * sin_yaw + vy * cos_yaw
                hospital._apply_velocity(vx, vy, omega * args.speed_scale, dt)
                last_cmd = (vx, vy, omega * args.speed_scale)

                closed = action.gripper_state == GripperState.CLOSED
                arm.set_gripper(closed)

                # Manipulate mode is implied by the action, not a separate
                # field: zero base velocity + closed gripper. See ik_bridge's
                # ControlMode. In drive mode the arm targets are simply left
                # alone, so it holds its pose while the robot moves.
                manipulating = closed and max(abs(v) for v in action.base_velocity) < 1e-6
                if manipulating and action.arm_target_pose is not None:
                    arm.servo_to(
                        arm.height_for(
                            action.arm_target_pose.position[2],
                            WORKSPACE_Z_MIN,
                            WORKSPACE_Z_MAX,
                        )
                    )
                if closed != gripper_closed:
                    print(f"[isaac] gripper {'CLOSED' if closed else 'OPEN'}")
                    gripper_closed = closed
                mode = "MANIP" if manipulating else "DRIVE"

            for _ in range(physics_steps):
                hospital.sim.step(render=False)
            hospital.sim.render()
            ticks += 1

            now = time.time()
            if now - last_report >= 2.0:
                x, y = hospital.base_xy()
                state = "live" if receiver.is_live else "idle"
                # Report the command as well as the pose: "actions arriving but
                # base not moving" and "actions arriving that say zero" look
                # identical from position alone, and they have different causes.
                print(
                    f"[isaac] {state} | base=({x:6.2f}, {y:6.2f}) yaw={hospital.base_yaw():+.2f} "
                    f"| {mode} cmd=({last_cmd[0]:+.2f},{last_cmd[1]:+.2f},{last_cmd[2]:+.2f}) "
                    f"claw_z={arm.ee_z():.2f} "
                    f"| {receiver.received} rx, {receiver.dropped} dropped "
                    f"| {ticks / max(now - started, 1e-6):.1f} tick/s"
                )
                last_report = now

            if args.max_seconds and now - started > args.max_seconds:
                print(f"[isaac] reached --max-seconds {args.max_seconds}, exiting")
                break
    except KeyboardInterrupt:
        print("\n[isaac] interrupted")
    finally:
        receiver.close()
        commands.close()
        hospital.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
