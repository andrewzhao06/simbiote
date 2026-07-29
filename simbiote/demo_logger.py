"""Shared demo logger — spec §6.5/§6b.5, jointly owned by Teammates 3 & 4:

    simbiote/demo_logger.py  -- log_action(action, source: "teleop"|"agentic")
                                    export_trajectory(session_id) -> Trajectory

This is the merge of all three teammates' independent implementations, once
Sky's teleop chain and Andrew's agentic chain both needed to actually call it:

- `start_session()` / `log_action()` / `export_trajectory()` / `end_session()`
  module-level API, with in-memory sessions keyed by `session_id` (Suraj's) --
  this is what `training/retrain.py`'s `ingest_demo()` / `finetune_policy()`
  and the test suite drive directly.
- `log_action()` no longer *requires* a prior `start_session()` call -- if a
  caller (Sky's/Andrew's original code, both wrote it this way) logs without
  one, a session is created implicitly using `new_session_id(source)`.
  Explicit `start_session()` still works exactly as before for callers that
  want to name/configure a session up front.
- Durable JSONL persistence per action, one line per step, opened and closed
  per call rather than held open (Andrew's) -- a session that dies mid-run
  still leaves a readable partial trajectory on disk, which an in-memory-only
  logger can't offer. Every `log_action()` call both updates the in-memory
  session *and* appends to `<stage>/sessions/<session_id>.jsonl`.
- `stage_dir()` / `session_path()` / `new_session_id()` / `write_report()`
  utilities (Andrew's) -- `agentic_session.py` calls these directly for the
  execution-report sidecar and to report where a trajectory landed on disk.
- Falls back to a local `./stage` directory when the sandboxed
  `/var/simbiote/stage` path doesn't exist (Sky's laptop-friendly default),
  overridable via `$SIMBIOTE_STAGE` (Andrew's env var -- see spec Part 3,
  security architecture).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from simbiote.robot_iface.actions import RobotAction
from simbiote.robot_iface.trajectory import DemoSource, Trajectory, TrajectoryStep

__all__ = [
    "end_session",
    "export_trajectory",
    "log_action",
    "new_session_id",
    "session_path",
    "stage_dir",
    "start_session",
    "write_report",
]

_active_sessions: dict[str, Trajectory] = {}
_current_session_id: str | None = None

#: Per Part 3 (security architecture), the process only has write access to
#: /var/simbiote/stage on the real box. On a dev laptop that path won't
#: exist, so the default falls back to a local ./stage directory.
_SANDBOX_STAGE = "/var/simbiote/stage"


def stage_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the writable staging directory.

    Precedence: explicit argument, then `$SIMBIOTE_STAGE`, then the sandbox
    path if it exists, else a local `./stage` (laptop dev default).
    """
    if override is not None:
        return Path(override)
    env = os.environ.get("SIMBIOTE_STAGE")
    if env:
        return Path(env)
    sandbox = Path(_SANDBOX_STAGE)
    return sandbox if sandbox.exists() else Path("./stage")


def new_session_id(source: DemoSource = "teleop") -> str:
    """A sortable, collision-resistant session id: `agentic-20260725-231502-4f1a`."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    suffix = f"{int(time.time() * 1e6) % 0xFFFF:04x}"
    return f"{source}-{stamp}-{suffix}"


def session_path(session_id: str, stage: str | os.PathLike[str] | None = None) -> Path:
    return stage_dir(stage) / "sessions" / f"{session_id}.jsonl"


def start_session(
    session_id: str | None = None,
    source: DemoSource = "teleop",
    task: str = "nav",
) -> str:
    """Begin logging a new teleop/agentic session. Subsequent `log_action()`
    calls with no explicit `session_id` append to this one. Returns the
    session id (generated via `new_session_id()` if not given)."""
    global _current_session_id
    session_id = session_id or new_session_id(source)
    _active_sessions[session_id] = Trajectory(session_id=session_id, source=source, task=task)
    _current_session_id = session_id
    return session_id


def log_action(
    action: RobotAction,
    source: DemoSource,
    observation: list[float] | None = None,
    reward: float = 0.0,
    session_id: str | None = None,
    skill: str | None = None,
    ok: bool = True,
    t: float | None = None,
    stage: str | os.PathLike[str] | None = None,
) -> TrajectoryStep:
    """spec: `log_action(action, source: "teleop"|"agentic")`. `observation`
    is optional but should be supplied whenever available -- `bc_pretrain.py`
    needs it, an action alone isn't a trainable (obs, action) pair.

    If `session_id` names a session that hasn't been `start_session()`-ed
    (or no session is active at all), one is created implicitly -- teleop and
    agentic call sites don't always start a session explicitly.
    """
    global _current_session_id
    sid = session_id or _current_session_id
    if sid is None:
        sid = start_session(source=source, task="nav")
    elif sid not in _active_sessions:
        _active_sessions[sid] = Trajectory(session_id=sid, source=source, task="nav")
        _current_session_id = sid

    traj = _active_sessions[sid]
    if traj.source != source:
        raise ValueError(
            f"log_action: source '{source}' doesn't match session '{sid}''s source '{traj.source}'"
        )

    timestamp = time.time() if t is None else t
    step = TrajectoryStep(
        timestamp=timestamp,
        observation=observation or [],
        action=action,
        source=source,
        reward=reward,
        skill=skill,
        ok=ok,
    )
    traj.steps.append(step)

    path = session_path(sid, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(step.to_dict()) + "\n")

    return step


def export_trajectory(
    session_id: str | None = None, stage: str | os.PathLike[str] | None = None
) -> Trajectory:
    """spec: `export_trajectory(session_id) -> Trajectory`. Read-only —
    doesn't end the session, so a caller can export mid-session for a live
    preview (e.g. an audit trail, §3).

    Prefers the in-memory session (fast path, current process); falls back to
    reading the on-disk JSONL if the session isn't in memory (e.g. a
    previous, crashed process) -- durability from Andrew's design.
    """
    sid = session_id or _current_session_id
    if sid is not None and sid in _active_sessions:
        return _active_sessions[sid]

    if sid is None:
        raise KeyError("demo_logger.export_trajectory: no active session and no session_id given")

    path = session_path(sid, stage)
    if not path.exists():
        raise KeyError(f"demo_logger.export_trajectory: no session '{sid}'")

    steps: list[TrajectoryStep] = []
    source: DemoSource = "teleop"
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                steps.append(TrajectoryStep.from_dict(record))
                source = record.get("source", source)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                # A torn final line is expected if the session was killed
                # mid-write; anything else is a real schema problem.
                raise ValueError(f"{path}:{lineno}: malformed trajectory record: {exc}") from exc
    return Trajectory(session_id=sid, source=source, task="nav", steps=steps)


def end_session(session_id: str | None = None) -> Trajectory:
    """Export and drop a session from memory (call this once a teleop/agentic
    run completes, before handing the Trajectory to `ingest_demo()`)."""
    global _current_session_id
    trajectory = export_trajectory(session_id)
    sid = session_id or _current_session_id
    _active_sessions.pop(sid, None)
    if _current_session_id == sid:
        _current_session_id = None
    return trajectory


def write_report(
    session_id: str,
    report: dict[str, Any],
    stage: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the execution-report sidecar next to the trajectory.

    Per-skill success/failure lives here rather than on each action record,
    since it is only known once a skill has finished (Andrew's design, used
    by `agentic/agentic_session.py`).
    """
    path = session_path(session_id, stage).with_suffix(".report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
