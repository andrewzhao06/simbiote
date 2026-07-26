"""Shared demo logger — spec §6.5/§6b.5, jointly owned by Teammates 3 & 4:

    simbiote/demo_logger.py  -- log_action(action, source: "teleop"|"agentic")
                                    export_trajectory(session_id) -> Trajectory

This is Teammate 2's reference implementation, built so `training/retrain.py`
(`ingest_demo()` -> `finetune_policy()`) has a real producer to test against
tonight, ahead of the actual group conversation on this schema (§5.7/§6.7/
§6b.7 all flag it as "the one interface all [three/four] of you touch").

The spec's one-line signature (`log_action(action, source)`) implies a
single "current session" being logged to; `start_session()`/`end_session()`
make that session explicit (and support >1 concurrent session for testing)
without changing the two call sites Steps 3 & 4 actually use.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from simbiote.robot_iface.actions import RobotAction
from simbiote.robot_iface.trajectory import DemoSource, Trajectory, TrajectoryStep

_active_sessions: Dict[str, Trajectory] = {}
_current_session_id: Optional[str] = None


def start_session(session_id: str, source: DemoSource, task: str) -> None:
    """Begin logging a new teleop/agentic session. Subsequent `log_action()`
    calls with no explicit `session_id` append to this one."""
    global _current_session_id
    _active_sessions[session_id] = Trajectory(session_id=session_id, source=source, task=task)
    _current_session_id = session_id


def log_action(
    action: RobotAction,
    source: DemoSource,
    observation: Optional[List[float]] = None,
    reward: float = 0.0,
    session_id: Optional[str] = None,
) -> None:
    """spec: `log_action(action, source: "teleop"|"agentic")`. `observation`
    is optional but should be supplied whenever available -- `bc_pretrain.py`
    needs it, an action alone isn't a trainable (obs, action) pair."""
    sid = session_id or _current_session_id
    if sid is None or sid not in _active_sessions:
        raise RuntimeError("demo_logger.log_action: no active session -- call start_session() first")
    traj = _active_sessions[sid]
    if traj.source != source:
        raise ValueError(f"log_action: source '{source}' doesn't match session '{sid}''s source '{traj.source}'")
    timestamp = float(len(traj.steps))
    traj.steps.append(TrajectoryStep(timestamp=timestamp, observation=observation or [], action=action, reward=reward))


def export_trajectory(session_id: Optional[str] = None) -> Trajectory:
    """spec: `export_trajectory(session_id) -> Trajectory`. Read-only —
    doesn't end the session, so a caller can export mid-session for a live
    preview (e.g. OpenClaw's audit trail, §3)."""
    sid = session_id or _current_session_id
    if sid is None or sid not in _active_sessions:
        raise KeyError(f"demo_logger.export_trajectory: no session '{sid}'")
    return _active_sessions[sid]


def end_session(session_id: Optional[str] = None) -> Trajectory:
    """Export and drop a session from memory (call this once a teleop/agentic
    run completes, before handing the Trajectory to `ingest_demo()`)."""
    global _current_session_id
    trajectory = export_trajectory(session_id)
    sid = session_id or _current_session_id
    del _active_sessions[sid]
    if _current_session_id == sid:
        _current_session_id = None
    return trajectory
