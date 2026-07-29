"""Shared `Trajectory` schema — the one interface all four roles touch (§5.7/§6.7/§6b.7).

`demo_logger.py` (Steps 3 & 4) produces these; `bc_pretrain.py` (Step 2)
consumes them via `ingest_demo()` / `finetune_policy()` (see
`training/retrain.py`). Settling on this shape was called out in the spec as
a group conversation, not something to design in isolation -- this is the
merge of the three teammates' independent proposals once Sky's (teleop) and
Andrew's (agentic) actual demo-producing code needed to plug into it:

- `observation` / `reward` per step (Suraj's) -- required for `bc_pretrain.py`
  to treat a step as a trainable (obs, action) pair; teleop/agentic don't
  have an observation to give yet, so it defaults to `[]` and
  `bc_pretrain.py` skips steps that don't have one rather than crashing.
- `source` per step, not just per `Trajectory` (Sky's and Andrew's) -- keeps
  each logged line self-describing, which matters for the JSONL streaming
  format in `demo_logger.py` (a session killed mid-run still leaves readable,
  self-contained records).
- `skill` / `ok` per step (Andrew's) -- the atomic skill that produced an
  agentic action (`navigate_to`, `pick_up`, ...; `None` for teleop) and
  whether the action was accepted without an executor-level error. Purely
  additive, so old records without these keys still load (default `None`/`True`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from simbiote.robot_iface.actions import RobotAction

DemoSource = Literal["teleop", "agentic"]


@dataclass(frozen=True)
class TrajectoryStep:
    """One (observation, action) pair plus bookkeeping for a single timestep."""

    timestamp: float
    action: RobotAction
    source: DemoSource
    observation: list[float] = field(default_factory=list)
    reward: float = 0.0
    skill: str | None = None
    ok: bool = True

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action.to_dict(),
            "source": self.source,
            "observation": list(self.observation),
            "reward": self.reward,
            "skill": self.skill,
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryStep:
        return cls(
            timestamp=d["timestamp"],
            action=RobotAction.from_dict(d["action"]),
            source=d["source"],
            observation=list(d.get("observation", [])),
            reward=d.get("reward", 0.0),
            skill=d.get("skill"),
            ok=bool(d.get("ok", True)),
        )


@dataclass
class Trajectory:
    """A logged demonstration session — one teleop or agentic run."""

    session_id: str
    source: DemoSource
    task: str  # "nav" | "grasp" | "wheelchair"
    steps: list[TrajectoryStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def observations(self) -> list[list[float]]:
        return [s.observation for s in self.steps]

    def action_vectors(self) -> list[tuple]:
        return [s.action.to_vector() for s in self.steps]

    def skills(self) -> list[str]:
        """Skills in first-appearance order, for a quick sanity read of an
        agentic run (teleop steps have `skill=None` and are omitted)."""
        seen: list[str] = []
        for s in self.steps:
            if s.skill and s.skill not in seen:
                seen.append(s.skill)
        return seen

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trajectory:
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
    def load(cls, path: str | Path) -> Trajectory:
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
        steps.append(
            TrajectoryStep(
                timestamp=float(t), observation=obs, action=action, source=source, reward=0.0
            )
        )
    return Trajectory(session_id=session_id, source=source, task=task, steps=steps)


def load_trajectories(paths: Sequence[str | Path]) -> list[Trajectory]:
    return [Trajectory.load(p) for p in paths]
