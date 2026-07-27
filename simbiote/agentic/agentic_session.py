"""One agentic command, end to end — spec §6b.5.

parse -> execute -> log. Every executed session is logged as a demonstration,
exactly like a Step 3 teleop session, so Step 2's fine-tune loop ingests both
through the same reader.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from simbiote import demo_logger
from simbiote.agentic import task_executor
from simbiote.agentic.command_parser import ParseError, parse_instruction
from simbiote.agentic.llm_backend import (
    PROFILES,
    LLMBackend,
    describe_backend,
    make_backend,
    primary_of,
    resolve_profile,
)
from simbiote.agentic.robot_tools import (
    CheckpointBackend,
    IsaacBackend,
    RobotBackend,
    RobotTools,
    StubBackend,
)
from simbiote.agentic.scene_query import SceneGraph, load_scene
from simbiote.agentic.task_executor import ExecutionReport, StepReport, StepStatus
from simbiote.agentic.tool_schema import ToolCall
from simbiote.robot_iface.actions import RobotAction

__all__ = ["SessionResult", "run_session", "main"]

#: Best-measured nav policy for hospital traversal. `nav_bc.pt` (behavioural
#: cloning) beats both PPO variants here -- 16/20 location pairs against
#: nav_ppo's fewer, matching Suraj's finding that PPO on top of BC degrades it.
DEFAULT_NAV_CHECKPOINT = (
    Path(__file__).resolve().parent.parent.parent / "checkpoints" / "nav_bc.pt"
)


@dataclass
class SessionResult:
    session_id: str
    instruction: str
    calls: list[ToolCall] = field(default_factory=list)
    report: ExecutionReport | None = None
    trajectory: demo_logger.Trajectory | None = None
    error: str | None = None
    #: Which backend was configured and which one actually served the parse.
    llm: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.report is not None and self.report.ok

    @property
    def degraded(self) -> bool:
        """True if the plan came from the fallback rather than the real model."""
        return bool(self.llm.get("degraded"))


def run_session(
    instruction: str,
    scene: SceneGraph,
    llm: LLMBackend,
    robot: RobotBackend,
    *,
    session_id: str | None = None,
    stage: str | os.PathLike[str] | None = None,
    timeout_s: float = task_executor.DEFAULT_SKILL_TIMEOUT_S,
    on_step: "object | None" = None,
) -> SessionResult:
    """Parse, execute, and log one natural-language instruction."""
    session_id = session_id or demo_logger.new_session_id("agentic")
    result = SessionResult(session_id=session_id, instruction=instruction)

    try:
        result.calls = parse_instruction(instruction, scene, llm)
    except ParseError as exc:
        result.error = str(exc)
        result.llm = describe_backend(llm)
        return result
    result.llm = describe_backend(llm)

    tools = RobotTools(scene, robot)

    def sink(action: RobotAction, skill: str) -> None:
        # Logged as each action is emitted, not batched at the end, so a run
        # that dies mid-skill still leaves usable demonstration data.
        demo_logger.log_action(
            action, "agentic", session_id=session_id, skill=skill, stage=stage
        )

    result.report = task_executor.execute(
        result.calls,
        tools,
        on_action=sink,
        timeout_s=timeout_s,
        on_step=on_step,  # type: ignore[arg-type]
    )

    demo_logger.write_report(
        session_id,
        {
            "session_id": session_id,
            "instruction": instruction,
            "llm": result.llm,
            "plan": [c.to_dict() for c in result.calls],
            **result.report.to_dict(),
        },
        stage=stage,
    )
    try:
        result.trajectory = demo_logger.export_trajectory(session_id, stage=stage)
    except KeyError:
        # A plan that fails on its first skill never emits an action, so no
        # session file exists to export. That is a failed run, not a broken
        # one -- surface it through result.report like any other failure
        # instead of crashing the CLI with a traceback mid-demo.
        result.trajectory = None
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MARK = {
    StepStatus.SUCCEEDED: "ok  ",
    StepStatus.FAILED: "FAIL",
    StepStatus.TIMED_OUT: "TIME",
    StepStatus.BLOCKED: "BLOCK",
    StepStatus.SKIPPED: "skip",
}


def _print_step(step: StepReport) -> None:
    mark = _MARK.get(step.status, str(step.status))
    args = ", ".join(f"{k}={v}" for k, v in step.args.items())
    print(f"  [{mark}] {step.index + 1}. {step.tool}({args})  {step.detail}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simbiote-agentic",
        description="Issue a natural-language instruction to the robot (Step 4).",
    )
    parser.add_argument(
        "instruction",
        nargs="?",
        help='e.g. "pick up the tray in the supply room"; optional with --preflight',
    )
    parser.add_argument(
        "--scene", default=None, help="scene-graph JSON (default: the hospital fixture)"
    )
    parser.add_argument(
        "--llm",
        default="fake",
        choices=["fake", "openai-compat"],
        help="instruction-parsing backend (default: fake, no model needed)",
    )
    parser.add_argument(
        "--llm-profile",
        default=None,
        choices=sorted(PROFILES),
        help="model target for --llm openai-compat (default: $SIMBIOTE_LLM_PROFILE, else qwen3-8b)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            "fail instead of degrading to the rule-based backend when the model "
            "server is unreachable — use this during rehearsals"
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "wait for the model server to answer before parsing. Run this the "
            "moment training ends, while Nemotron Super is loading back in"
        ),
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=None,
        help="seconds to wait in --preflight (default: the profile's ready_timeout_s)",
    )
    parser.add_argument(
        "--robot",
        default="stub",
        choices=["stub", "checkpoint", "isaac"],
        help=(
            "skill-execution backend (default: stub). 'checkpoint' runs Step 2's "
            "exports in a throwaway PyBullet arena; 'isaac' drives the real robot "
            "through hospital.usd and keeps its pose between skills"
        ),
    )
    parser.add_argument(
        "--isaac-gui",
        action="store_true",
        help="show the Isaac Sim window when --robot isaac (default: headless)",
    )
    parser.add_argument("--nav-checkpoint", default="", help="path for --robot checkpoint")
    parser.add_argument("--grasp-checkpoint", default="", help="path for --robot checkpoint")
    parser.add_argument(
        "--stage",
        default=None,
        help="writable staging dir for logs (default: $SIMBIOTE_STAGE)",
    )
    parser.add_argument(
        "--fail-skill",
        action="append",
        default=[],
        metavar="NAME",
        help="stub only: force this skill to fail, to exercise the failure path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=task_executor.DEFAULT_SKILL_TIMEOUT_S,
        help="per-skill timeout in seconds",
    )
    return parser


def _run_preflight(llm: LLMBackend, args: argparse.Namespace) -> bool:
    """Warm the model server and report whether it answered.

    Separated from the instruction path because on the day these happen at
    different moments: preflight as soon as Isaac Sim releases memory, the
    instruction once an operator is standing at the machine.
    """
    backend = primary_of(llm)
    wait_ready = getattr(backend, "wait_ready", None)
    if wait_ready is None:
        print("preflight: nothing to warm (rule-based backend is always ready)")
        return True

    profile = resolve_profile(args.llm_profile)
    timeout = args.preflight_timeout if args.preflight_timeout is not None else profile.ready_timeout_s
    print(
        f"preflight: waiting up to {timeout:.0f}s for {profile.model} "
        f"(~{profile.footprint_gb:.0f} GB, resident={profile.resident}) at {profile.base_url}",
        flush=True,
    )
    if wait_ready(timeout):
        print("preflight: model server ready")
        return True
    print(
        f"preflight: model server did not answer within {timeout:.0f}s",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.instruction is None and not args.preflight:
        parser.error("an instruction is required unless --preflight is given")

    scene = load_scene(Path(args.scene) if args.scene else None)
    llm = make_backend(
        args.llm,
        scene,
        profile=args.llm_profile,
        fallback=not args.no_fallback,
        on_degrade=lambda exc: print(
            f"\n!! model server unreachable, falling back to the rule-based backend:\n"
            f"   {exc}\n"
            f"   This plan was NOT produced by the model. Re-run with --no-fallback "
            f"to fail instead.\n",
            file=sys.stderr,
            flush=True,
        ),
    )
    if args.robot == "stub":
        robot: RobotBackend = StubBackend(fail_skills=tuple(args.fail_skill))
    elif args.robot == "isaac":
        # Booting Isaac Sim and cooking the hospital's colliders takes a while,
        # so this is constructed only when actually selected.
        robot = IsaacBackend(
            nav_checkpoint=args.nav_checkpoint or str(DEFAULT_NAV_CHECKPOINT),
            grasp_checkpoint=args.grasp_checkpoint or None,
            headless=not args.isaac_gui,
        )
    else:
        robot = CheckpointBackend(args.nav_checkpoint, args.grasp_checkpoint)

    if args.preflight:
        ready = _run_preflight(llm, args)
        if args.instruction is None:
            return 0 if ready else 1
        if not ready and args.no_fallback:
            return 1

    llm_label = args.llm
    if args.llm != "fake":
        llm_label = f"{args.llm}/{resolve_profile(args.llm_profile).name}"
    print(f'instruction: "{args.instruction}"')
    print(f"scene: {scene.scene_id}  llm: {llm_label}  robot: {args.robot}")

    result = run_session(
        args.instruction,
        scene,
        llm,
        robot,
        stage=args.stage,
        timeout_s=args.timeout,
        on_step=_print_step,
    )

    if result.error:
        print(f"\nparse failed: {result.error}", file=sys.stderr)
        return 2

    print("\nplan:")
    for i, call in enumerate(result.calls, start=1):
        print(f"  {i}. {call}")

    print("\nexecution:")
    # Steps already streamed via on_step above; print the summary.
    assert result.report is not None
    print(f"  actions logged: {result.report.action_count}")
    if result.report.compensated:
        print("  compensating detach ran after failure")

    traj_path = demo_logger.session_path(result.session_id, args.stage)
    print(f"\nsession:    {result.session_id}")
    print(f"planned by: {result.llm.get('served_by')}" + ("  (DEGRADED)" if result.degraded else ""))
    print(f"trajectory: {traj_path}  ({len(result.trajectory or [])} actions)")

    if result.ok:
        print("\nresult: SUCCESS")
        return 0
    print(f"\nresult: FAILED — {result.report.failure_reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
