"""The task hierarchy — master doc 6b.4.

The piece that makes this more than one opaque tool call. The LLM produces a
plan once; from there this FSM runs each atomic skill to completion or failure,
checks its precondition before entering it, and decides whether it is safe to
advance. No LLM call happens per step.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from factoryflow.agentic.robot_tools import ActionSink, RobotTools, SkillResult
from factoryflow.agentic.tool_schema import ToolCall

__all__ = [
    "StepStatus",
    "StepReport",
    "ExecutionReport",
    "execute",
    "DEFAULT_SKILL_TIMEOUT_S",
]

#: A policy that never converges must fail its step rather than hang the demo.
DEFAULT_SKILL_TIMEOUT_S = 120.0


class StepStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"  # precondition not met, skill never ran
    SKIPPED = "skipped"  # an earlier step failed

    def __str__(self) -> str:
        return self.value


@dataclass
class StepReport:
    index: int
    tool: str
    args: dict[str, Any]
    status: StepStatus
    detail: str = ""
    duration_s: float = 0.0
    action_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "args": dict(self.args),
            "status": self.status.value,
            "detail": self.detail,
            "duration_s": round(self.duration_s, 4),
            "action_count": self.action_count,
        }


@dataclass
class ExecutionReport:
    steps: list[StepReport] = field(default_factory=list)
    ok: bool = True
    failed_step: int | None = None
    failure_reason: str = ""
    #: True if a held constraint had to be released after a failure.
    compensated: bool = False

    @property
    def action_count(self) -> int:
        return sum(s.action_count for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failed_step": self.failed_step,
            "failure_reason": self.failure_reason,
            "compensated": self.compensated,
            "action_count": self.action_count,
            "steps": [s.to_dict() for s in self.steps],
        }


def _precondition_error(call: ToolCall, tools: RobotTools) -> str | None:
    """Why this skill must not start yet, or None if it may.

    Preconditions live here rather than inside the skills so the executor —
    not the model, and not the backend — owns the decision to advance.
    """
    held = tools.attached_to

    if call.tool == "attach_handle":
        if held is not None:
            return f"already attached to {held!r}; detach before attaching again"
    elif call.tool == "nav_with_payload":
        if held is None:
            return "no payload is attached; attach_handle must succeed first"
    elif call.tool == "detach":
        if held is None:
            return "nothing is attached; detach would be a no-op"
    elif call.tool in ("navigate_to", "pick_up"):
        if held is not None:
            return (
                f"a payload ({held!r}) is still attached; use nav_with_payload "
                "or detach first"
            )
    return None


def execute(
    calls: list[ToolCall],
    tools: RobotTools,
    *,
    on_action: ActionSink | None = None,
    timeout_s: float = DEFAULT_SKILL_TIMEOUT_S,
    on_step: Callable[[StepReport], None] | None = None,
) -> ExecutionReport:
    """Run a validated plan, one skill at a time.

    Aborts on the first failure and, if a constraint is still held at that
    point, releases it — leaving the robot welded to a wheelchair mid-failure
    is the worst way for this to end on stage.
    """
    report = ExecutionReport()
    aborted = False

    # A worker thread per run so a wedged skill can be abandoned on timeout.
    # The abandoned thread is not killed (Python cannot); it is left detached
    # and the run aborts, which is the honest behaviour for a hung policy.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skill")
    try:
        for index, call in enumerate(calls):
            if aborted:
                step = StepReport(index, call.tool, call.args, StepStatus.SKIPPED,
                                  "an earlier step failed")
                report.steps.append(step)
                if on_step:
                    on_step(step)
                continue

            blocked = _precondition_error(call, tools)
            if blocked is not None:
                step = StepReport(index, call.tool, call.args, StepStatus.BLOCKED, blocked)
                report.steps.append(step)
                if on_step:
                    on_step(step)
                report.ok = False
                report.failed_step = index
                report.failure_reason = f"{call.tool}: {blocked}"
                aborted = True
                continue

            started = time.monotonic()
            future = pool.submit(tools.call, call, on_action)
            try:
                result: SkillResult = future.result(timeout=timeout_s)
            except FutureTimeout:
                elapsed = time.monotonic() - started
                step = StepReport(
                    index, call.tool, call.args, StepStatus.TIMED_OUT,
                    f"skill did not finish within {timeout_s:g}s", elapsed,
                )
                report.steps.append(step)
                if on_step:
                    on_step(step)
                report.ok = False
                report.failed_step = index
                report.failure_reason = f"{call.tool}: timed out after {timeout_s:g}s"
                aborted = True
                continue
            except Exception as exc:  # a backend blew up rather than returning
                elapsed = time.monotonic() - started
                step = StepReport(
                    index, call.tool, call.args, StepStatus.FAILED,
                    f"{type(exc).__name__}: {exc}", elapsed,
                )
                report.steps.append(step)
                if on_step:
                    on_step(step)
                report.ok = False
                report.failed_step = index
                report.failure_reason = f"{call.tool}: {type(exc).__name__}: {exc}"
                aborted = True
                continue

            elapsed = time.monotonic() - started
            step = StepReport(
                index,
                call.tool,
                call.args,
                StepStatus.SUCCEEDED if result.ok else StepStatus.FAILED,
                result.detail,
                elapsed,
                len(result.actions),
            )
            report.steps.append(step)
            if on_step:
                on_step(step)

            if not result.ok:
                report.ok = False
                report.failed_step = index
                report.failure_reason = f"{call.tool}: {result.detail}"
                aborted = True

        # Compensating release, so a failed run does not leave the arm welded
        # to a payload.
        if not report.ok and tools.attached_to is not None:
            try:
                release = tools.detach(on_action)
                report.compensated = release.ok
            except Exception:  # nothing useful left to do if cleanup itself fails
                report.compensated = False
    finally:
        # wait=False: a timed-out skill's thread must not block the return.
        pool.shutdown(wait=False)

    return report
