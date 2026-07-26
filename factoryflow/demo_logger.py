"""Demonstration logging — master doc Part 6.5 / 6b.5.

PROPOSAL, not yet ratified. Shared with Teammate 3 (teleop) and consumed by
Teammate 2's ``ingest_demo()``. Per 6b.7 this schema is a *group* decision, not
a bilateral one — see SCHEMAS_PROPOSAL.md.

Format is JSON Lines, one action per line, rather than a single JSON array: a
session that dies mid-run still leaves a readable partial trajectory. That
matters when the plan is ten rehearsal loops under time pressure.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from factoryflow.robot_iface.actions import RobotAction

__all__ = [
    "Source",
    "TrajectoryStep",
    "Trajectory",
    "new_session_id",
    "stage_dir",
    "session_path",
    "log_action",
    "export_trajectory",
]

Source = Literal["teleop", "agentic"]

#: Master doc Part 3 gives the sandbox exactly one writable path on the GB10.
#: Overridable for laptop work and tests via ``FACTORYFLOW_STAGE``.
DEFAULT_STAGE = "/var/factoryflow/stage"


def stage_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the writable staging directory.

    Precedence: explicit argument, then ``$FACTORYFLOW_STAGE``, then the
    sandbox default. On a dev laptop the default is not writable, so
    ``FACTORYFLOW_STAGE`` is expected to be set — see the README.
    """
    if override is not None:
        return Path(override)
    return Path(os.environ.get("FACTORYFLOW_STAGE", DEFAULT_STAGE))


def new_session_id(source: Source = "agentic") -> str:
    """A sortable, collision-resistant session id: ``agentic-20260725-231502-4f1a``."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    suffix = f"{int(time.time() * 1e6) % 0xFFFF:04x}"
    return f"{source}-{stamp}-{suffix}"


def session_path(session_id: str, stage: str | os.PathLike[str] | None = None) -> Path:
    return stage_dir(stage) / "sessions" / f"{session_id}.jsonl"


@dataclass(frozen=True)
class TrajectoryStep:
    """One logged action.

    ``skill`` is Step 4's addition to the shared record: the atomic skill that
    produced this action (``navigate_to``, ``pick_up``, ...). Teleop leaves it
    ``None``. It is additive, so a teleop-written file still reads cleanly here,
    and it lets Step 2 slice a trajectory by skill when fine-tuning.

    ``ok`` means "this action was issued to the robot without error". Whether the
    *skill* ultimately succeeded lives in the execution report sidecar, because
    that is not known at the moment an action is emitted.
    """

    t: float
    source: Source
    action: RobotAction
    skill: str | None = None
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "source": self.source,
            "skill": self.skill,
            "action": self.action.to_dict(),
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryStep":
        return cls(
            t=float(d["t"]),
            source=d["source"],
            action=RobotAction.from_dict(d["action"]),
            skill=d.get("skill"),
            ok=bool(d.get("ok", True)),
        )


@dataclass
class Trajectory:
    """An ordered set of logged actions for one session."""

    session_id: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[TrajectoryStep]:
        return iter(self.steps)

    @property
    def actions(self) -> list[RobotAction]:
        return [s.action for s in self.steps]

    @property
    def duration_s(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return self.steps[-1].t - self.steps[0].t

    def skills(self) -> list[str]:
        """Skills in first-appearance order, for a quick sanity read of a run."""
        seen: list[str] = []
        for s in self.steps:
            if s.skill and s.skill not in seen:
                seen.append(s.skill)
        return seen


def log_action(
    action: RobotAction,
    source: Source,
    *,
    session_id: str,
    skill: str | None = None,
    t: float | None = None,
    ok: bool = True,
    stage: str | os.PathLike[str] | None = None,
) -> TrajectoryStep:
    """Append one action to the session's trajectory file.

    Opened and closed per call rather than held open: a held handle loses
    buffered lines if the process is killed, which is the exact case JSONL is
    here to survive.
    """
    step = TrajectoryStep(
        t=time.time() if t is None else t,
        source=source,
        action=action,
        skill=skill,
        ok=ok,
    )
    path = session_path(session_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(step.to_dict()) + "\n")
    return step


def export_trajectory(
    session_id: str, stage: str | os.PathLike[str] | None = None
) -> Trajectory:
    """Read a session back. This is the call Step 2's ``ingest_demo()`` wraps."""
    path = session_path(session_id, stage)
    if not path.exists():
        raise FileNotFoundError(f"no trajectory logged for session {session_id!r}: {path}")

    steps: list[TrajectoryStep] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(TrajectoryStep.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                # A torn final line is expected if the session was killed
                # mid-write; anything else is a real schema problem.
                raise ValueError(
                    f"{path}:{lineno}: malformed trajectory record: {exc}"
                ) from exc
    return Trajectory(session_id=session_id, steps=steps)


def write_report(
    session_id: str,
    report: dict[str, Any],
    stage: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the execution-report sidecar next to the trajectory.

    Per-skill success/failure lives here rather than on each action record,
    since it is only known once a skill has finished.
    """
    path = session_path(session_id, stage).with_suffix(".report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
