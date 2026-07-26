"""Shared `Trajectory` schema — the one interface all four roles touch (§5.7/§6.7/§6b.7).

`demo_logger.py` (Steps 3 & 4) produces these; `bc_pretrain.py` (Step 2)
consumes them via `ingest_demo()` / `finetune_policy()` (see
`training/retrain.py`). Settling on this shape is called out in the spec as
a group conversation, not something to design in isolation — this is
Teammate 2's proposed concrete version, used here so the training code has
something real to run against tonight.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Sequence

from simbiote.robot_iface.actions import RobotAction

DemoSource = Literal["teleop", "agentic"]


@dataclass(frozen=True)
class TrajectoryStep:
    """One (observation, action) pair plus bookkeeping for a single timestep."""

    timestamp: float
    observation: List[float]
    action: RobotAction
    reward: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = {
            "base_velocity": list(self.action.base_velocity),
            "arm_target_pose": {
                "position": list(self.action.arm_target_pose.position),
                "orientation": list(self.action.arm_target_pose.orientation),
            },
            "gripper_state": self.action.gripper_state.value,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        from simbiote.robot_iface.actions import GripperState, Pose

        a = d["action"]
        action = RobotAction(
            base_velocity=tuple(a["base_velocity"]),
            arm_target_pose=Pose(
                position=tuple(a["arm_target_pose"]["position"]),
                orientation=tuple(a["arm_target_pose"]["orientation"]),
            ),
            gripper_state=GripperState(a["gripper_state"]),
        )
        return cls(
            timestamp=d["timestamp"],
            observation=list(d["observation"]),
            action=action,
            reward=d.get("reward", 0.0),
        )


@dataclass
class Trajectory:
    """A logged demonstration session — one teleop or agentic run."""

    session_id: str
    source: DemoSource
    task: str  # "nav" | "grasp" | "wheelchair"
    steps: List[TrajectoryStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def observations(self) -> List[List[float]]:
        return [s.observation for s in self.steps]

    def action_vectors(self) -> List[tuple]:
        return [s.action.to_vector() for s in self.steps]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            session_id=d["session_id"],
            source=d["source"],
            task=d["task"],
            steps=[TrajectoryStep.from_dict(s) for s in d["steps"]],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        return cls.from_dict(json.loads(Path(path).read_text()))


def make_toy_trajectory(
    session_id: str,
    obs_dim: int,
    length: int = 20,
    task: str = "nav",
    source: DemoSource = "teleop",
    seed: int = 0,
) -> Trajectory:
    """A deterministic fake trajectory — used to smoke-test bc_pretrain.py
    before any real teleop/agentic demo exists (per §5.4's build order note)."""
    import random

    rng = random.Random(seed)
    steps = []
    for t in range(length):
        obs = [rng.uniform(-1, 1) for _ in range(obs_dim)]
        action = RobotAction(
            base_velocity=(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)),
        )
        steps.append(TrajectoryStep(timestamp=float(t), observation=obs, action=action, reward=0.0))
    return Trajectory(session_id=session_id, source=source, task=task, steps=steps)


def load_trajectories(paths: Sequence[str | Path]) -> List[Trajectory]:
    return [Trajectory.load(p) for p in paths]
