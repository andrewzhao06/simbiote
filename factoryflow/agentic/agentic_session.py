"""One agentic command, end to end — master doc 6b.5.

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

from factoryflow import demo_logger
from factoryflow.agentic import task_executor
from factoryflow.agentic.command_parser import ParseError, parse_instruction
from factoryflow.agentic.llm_backend import LLMBackend, make_backend
from factoryflow.agentic.robot_tools import (
    CheckpointBackend,
    RobotBackend,
    RobotTools,
    StubBackend,
)
from factoryflow.agentic.scene_query import SceneGraph, load_scene
from factoryflow.agentic.task_executor import ExecutionReport, StepReport, StepStatus
from factoryflow.agentic.tool_schema import ToolCall
from factoryflow.robot_iface.actions import RobotAction

__all__ = ["SessionResult", "run_session", "main"]


@dataclass
class SessionResult:
    session_id: str
    instruction: str
    calls: list[ToolCall] = field(default_factory=list)
    report: ExecutionReport | None = None
    trajectory: demo_logger.Trajectory | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.report is not None and self.report.ok


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
        return result

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
            "plan": [c.to_dict() for c in result.calls],
            **result.report.to_dict(),
        },
        stage=stage,
    )
    result.trajectory = demo_logger.export_trajectory(session_id, stage=stage)
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
        prog="factoryflow-agentic",
        description="Issue a natural-language instruction to the robot (Step 4).",
    )
    parser.add_argument("instruction", help='e.g. "pick up the tray in the supply room"')
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
        "--robot",
        default="stub",
        choices=["stub", "checkpoint"],
        help="skill-execution backend (default: stub; 'checkpoint' needs Step 2's exports)",
    )
    parser.add_argument("--nav-checkpoint", default="", help="path for --robot checkpoint")
    parser.add_argument("--grasp-checkpoint", default="", help="path for --robot checkpoint")
    parser.add_argument(
        "--stage",
        default=None,
        help="writable staging dir for logs (default: $FACTORYFLOW_STAGE)",
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    scene = load_scene(Path(args.scene) if args.scene else None)
    llm = make_backend(args.llm, scene)
    robot: RobotBackend = (
        StubBackend(fail_skills=tuple(args.fail_skill))
        if args.robot == "stub"
        else CheckpointBackend(args.nav_checkpoint, args.grasp_checkpoint)
    )

    print(f'instruction: "{args.instruction}"')
    print(f"scene: {scene.scene_id}  llm: {args.llm}  robot: {args.robot}")

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
    print(f"trajectory: {traj_path}  ({len(result.trajectory or [])} actions)")

    if result.ok:
        print("\nresult: SUCCESS")
        return 0
    print(f"\nresult: FAILED — {result.report.failure_reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
