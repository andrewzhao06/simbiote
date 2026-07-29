"""`ingest_demo()` -> `finetune_policy()` — the OpenClaw orchestrator tool
pair from Part 1's table ("Log & retrain | ingest_demo() -> finetune_policy()
| Runs after a teleop/agentic session | Parts 5 & 6/6b"), owned by Step 2 on
the consuming end. §5.4's build-spec file tree didn't previously give this
pairing a file of its own -- this module is that home, wiring
`bc_pretrain.py` + `training/ppo.py` together into the loop described in
§5.4's "combined learning loop" paragraph:

    BC pretrain on whatever teleop demos exist -> PPO fine-tune in sim ->
    new teleop session arrives -> re-run bc_pretrain.py on the combined demo
    set, seeded from the current PPO checkpoint -> PPO fine-tune again.
"""

from __future__ import annotations

import re
from pathlib import Path

from simbiote.robot_iface.trajectory import Trajectory
from simbiote.sim_env.register_envs import make_env, register
from simbiote.training.bc_pretrain import train_bc
from simbiote.training.policy_net import ActorCriticMLP
from simbiote.training.ppo import PPOConfig, train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_STORE_DIR = (
    REPO_ROOT / "var" / "simbiote" / "stage" / "demos"
)  # matches §3's write-scoped stage dir
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# Demo file paths are built directly from `Trajectory.task`/`session_id`
# (attacker-controllable, since Step 3/4's teleop/agentic producers set
# them) -- restrict both to a safe charset so a value like `../../etc` can't
# escape `DEMO_STORE_DIR` (path traversal).
_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _sanitize_path_component(value: str, label: str) -> str:
    if not _SAFE_PATH_COMPONENT_RE.match(value):
        raise ValueError(
            f"Unsafe {label} '{value}': must match {_SAFE_PATH_COMPONENT_RE.pattern!r} "
            "(letters, digits, '_', '-' only)"
        )
    return value


def _task_demo_dir(task: str) -> Path:
    d = DEMO_STORE_DIR / _sanitize_path_component(task, "task")
    d.mkdir(parents=True, exist_ok=True)
    return d


def ingest_demo(trajectory: Trajectory) -> Path:
    """OpenClaw tool: `ingest_demo()`. Persists one logged demonstration
    (from Step 3's teleop or Step 4's agentic session) to the on-disk demo
    store, keyed by task and session id."""
    session_id = _sanitize_path_component(trajectory.session_id, "session_id")
    out_path = _task_demo_dir(trajectory.task) / f"{session_id}.json"
    trajectory.save(out_path)
    return out_path


def list_demos(task: str) -> list[Path]:
    return sorted(_task_demo_dir(task).glob("*.json"))


def load_all_demos(task: str) -> list[Trajectory]:
    return [Trajectory.load(p) for p in list_demos(task)]


def finetune_policy(
    task: str,
    current_checkpoint: str | Path | None = None,
    bc_epochs: int = 30,
    ppo_timesteps: int = 4_000,
    num_envs: int = 2,
    out_path: str | Path | None = None,
    seed: int = 0,
) -> Path:
    """OpenClaw tool: `finetune_policy()`. Runs the full "BC on ingested
    demos, seeded from the current checkpoint, then PPO fine-tune" loop for
    `task` ("nav" | "grasp") and returns the path of the new checkpoint.

    Raises `RuntimeError` if no demos have been ingested for `task` yet --
    call `ingest_demo()` at least once first (or run `bc_pretrain.py`
    directly against toy trajectories, per §5.4's build-order note).
    """
    demos = load_all_demos(task)
    if not demos:
        raise RuntimeError(
            f"finetune_policy('{task}'): no ingested demos found in {_task_demo_dir(task)} "
            "-- call ingest_demo() first, or use bc_pretrain.train_bc() directly "
            "against toy trajectories."
        )

    register()
    probe_env = make_env(task, seed=seed)
    act_low = tuple(probe_env.action_space.low.tolist())
    act_high = tuple(probe_env.action_space.high.tolist())
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    probe_env.close()

    seed_policy = ActorCriticMLP.load(current_checkpoint) if current_checkpoint else None

    bc_ckpt = train_bc(
        demos,
        policy_net=seed_policy,
        task=task,
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        epochs=bc_epochs,
        out_path=CHECKPOINT_DIR / f"{task}_bc.pt",
        seed=seed,
    )
    policy = ActorCriticMLP.load(bc_ckpt)

    def make_env_fn(env_index: int):
        # Each parallel env needs its own seed -- a shared seed across all
        # `num_envs` workers would replay identical episodes and just
        # duplicate transitions instead of adding independent experience.
        def env_fn():
            return make_env(task, seed=seed + env_index)

        return env_fn

    config = PPOConfig(total_timesteps=ppo_timesteps, seed=seed)
    trained = train_ppo([make_env_fn(i) for i in range(num_envs)], policy, config)

    out_path = Path(out_path) if out_path else CHECKPOINT_DIR / f"{task}_ppo.pt"
    trained.save(out_path)
    return out_path
