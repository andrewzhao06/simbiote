"""Demonstration logging, shared by Step 3 (teleop) and Step 4 (agentic).

Both control paths produce the same RobotAction stream (see
robot_iface/actions.py); this module records that stream per-session and
exports it in the schema Step 2's ingest_demo() / finetune_policy() consumes.

Per Part 3 (security architecture), the process only has write access to
/var/factoryflow/stage/ on the real box. On a dev laptop that path won't
exist, so the default falls back to a local ./stage directory — pass
log_dir explicitly on the actual GB10/DGX deployment.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from factoryflow.robot_iface.actions import RobotAction

Source = Literal["teleop", "agentic"]

_STAGE_DIR = Path("/var/factoryflow/stage")
DEFAULT_LOG_DIR = _STAGE_DIR / "teleop_logs" if _STAGE_DIR.exists() else Path("./stage/teleop_logs")


@dataclass
class TrajectoryStep:
    action: RobotAction
    source: Source

    def to_dict(self) -> dict:
        return {"action": self.action.to_dict(), "source": self.source}

    @staticmethod
    def from_dict(d: dict) -> "TrajectoryStep":
        return TrajectoryStep(action=RobotAction.from_dict(d["action"]), source=d["source"])


@dataclass
class Trajectory:
    session_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "steps": [s.to_dict() for s in self.steps]}

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        return Trajectory(
            session_id=d["session_id"],
            steps=[TrajectoryStep.from_dict(s) for s in d["steps"]],
        )


class DemoLogger:
    """Buffers actions per session_id and can export/persist them as a Trajectory."""

    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Trajectory] = {}
        self._active_session_id: Optional[str] = None

    def start_session(self, session_id: Optional[str] = None) -> str:
        session_id = session_id or f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self._sessions[session_id] = Trajectory(session_id=session_id)
        self._active_session_id = session_id
        return session_id

    def log_action(self, action: RobotAction, source: Source, session_id: Optional[str] = None) -> None:
        session_id = session_id or self._active_session_id
        if session_id is None:
            session_id = self.start_session()
        if session_id not in self._sessions:
            self._sessions[session_id] = Trajectory(session_id=session_id)
        self._sessions[session_id].steps.append(TrajectoryStep(action=action, source=source))

    def export_trajectory(self, session_id: str) -> Trajectory:
        if session_id not in self._sessions:
            raise KeyError(f"no logged actions for session_id={session_id!r}")
        return self._sessions[session_id]

    def save(self, session_id: str, out_path: Optional[Path] = None) -> Path:
        trajectory = self.export_trajectory(session_id)
        out_path = Path(out_path) if out_path else self.log_dir / f"{session_id}.json"
        out_path.write_text(json.dumps(trajectory.to_dict(), indent=2))
        return out_path


# ---------------------------------------------------------------------------
# Module-level convenience API matching the master-doc signatures directly:
#   log_action(action, source)
#   export_trajectory(session_id) -> Trajectory
# backed by a process-wide default logger instance.
# ---------------------------------------------------------------------------

_default_logger: Optional[DemoLogger] = None


def get_logger() -> DemoLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = DemoLogger()
    return _default_logger


def start_session(session_id: Optional[str] = None) -> str:
    return get_logger().start_session(session_id)


def log_action(action: RobotAction, source: Source, session_id: Optional[str] = None) -> None:
    get_logger().log_action(action, source, session_id)


def export_trajectory(session_id: str) -> Trajectory:
    return get_logger().export_trajectory(session_id)
