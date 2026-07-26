"""Behavioral Cloning pretraining — spec §5.4:

    def train_bc(trajectories: list[Trajectory], policy_net) -> checkpoint

Supervised (observation -> action) regression on teleop/agentic demos from
`demo_logger.export_trajectory()`, using the **same MLP actor-critic
architecture** as `train_nav.py`/`train_grasp.py` so the resulting weights
load straight into PPO fine-tuning as a warm start (spec's combined-learning-
loop section: "BC pretrain on whatever teleop demos exist -> PPO fine-tune in
sim -> new teleop session arrives -> re-run bc_pretrain.py ... -> PPO
fine-tune again").

Logged `Trajectory`/`RobotAction` steps are task-agnostic; `task_action()`
below is where they get projected onto each task's specific action
parameterization (nav: base velocity, directly; grasp: EE position delta +
gripper, derived from consecutive `arm_target_pose`s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from simbiote.robot_iface.actions import GripperState
from simbiote.robot_iface.trajectory import Trajectory
from simbiote.sim_env.grasp_task import MAX_EE_STEP
from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def task_action_vector(prev_ee_target: Optional[np.ndarray], action) -> Tuple[np.ndarray, np.ndarray]:
    """Project one logged `RobotAction` onto the grasp task's 4-dim action
    (dx, dy, dz, gripper_cmd). Returns (action_vector, new_ee_target)."""
    target = np.array(action.arm_target_pose.position, dtype=np.float32)
    if prev_ee_target is None:
        delta = np.zeros(3, dtype=np.float32)
        new_ee_target = target
    else:
        # Teleop/agentic poses can jump further per frame than `GraspEnv`
        # ever allows per step -- clip to the same per-step limit the env
        # enforces so BC regresses onto action labels the grasp environment
        # can actually execute (matching `GraspEnv.step()`'s `action_space`
        # bounds). `new_ee_target` tracks the *clipped* running target (what
        # `GraspEnv._ee_target` would actually reach), not the raw teleop
        # pose, so the next step's delta isn't computed against a target the
        # env never actually reached.
        delta = np.clip(target - prev_ee_target, -MAX_EE_STEP, MAX_EE_STEP)
        new_ee_target = prev_ee_target + delta
    gripper_cmd = 1.0 if action.gripper_state == GripperState.CLOSED else -1.0
    return np.concatenate([delta, [gripper_cmd]]).astype(np.float32), new_ee_target


def trajectories_to_dataset(trajectories: Iterable[Trajectory], task: str = "nav") -> Tuple[np.ndarray, np.ndarray]:
    """Flatten a list of `Trajectory` into (obs, action) arrays for `task`
    ("nav" | "grasp"). Every step across every trajectory becomes one row."""
    obs_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []

    skipped_no_obs = 0
    for traj in trajectories:
        prev_ee_target: Optional[np.ndarray] = None
        for step in traj.steps:
            if task == "grasp":
                # Compute the action (and advance prev_ee_target) regardless
                # of whether this step has an observation, so a step logged
                # without privileged state doesn't desync the running EE
                # target used to derive subsequent deltas.
                act_vec, prev_ee_target = task_action_vector(prev_ee_target, step.action)
            elif task != "nav":
                raise ValueError(f"Unknown task '{task}', expected 'nav' or 'grasp'")

            if not step.observation:
                # `demo_logger.log_action()` allows omitting `observation`
                # (e.g. agentic steps logged without privileged state) and
                # stores `[]` in that case. A zero-length row can't be
                # stacked into a fixed-width obs array, so skip it here
                # rather than corrupting/crashing BC training downstream.
                skipped_no_obs += 1
                continue

            obs_rows.append(np.asarray(step.observation, dtype=np.float32))
            if task == "nav":
                act_rows.append(np.asarray(step.action.base_velocity, dtype=np.float32))
            else:
                act_rows.append(act_vec)

    if not obs_rows:
        raise ValueError(
            "trajectories_to_dataset: no steps with observations found across the given "
            f"trajectories (skipped {skipped_no_obs} step(s) with no logged observation)"
        )

    return np.stack(obs_rows), np.stack(act_rows)


def train_bc(
    trajectories: List[Trajectory],
    policy_net: Optional[ActorCriticMLP] = None,
    task: str = "nav",
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
    hidden_sizes: Tuple[int, ...] = (128, 128),
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    out_path: Optional[str | Path] = None,
    act_low: Optional[Tuple[float, ...]] = None,
    act_high: Optional[Tuple[float, ...]] = None,
    seed: int = 0,
) -> Path:
    """Supervised warm-start. Returns the path of the saved checkpoint.

    If `policy_net` is given, its weights are fine-tuned in place (used by
    `training/retrain.py` to seed BC from the current PPO checkpoint, per
    the spec's combined-loop ordering). Otherwise a fresh network is built
    from `obs_dim`/`act_dim` (inferred from the data if not given).
    """
    torch.manual_seed(seed)
    obs_arr, act_arr = trajectories_to_dataset(trajectories, task=task)

    if policy_net is None:
        meta = PolicyMeta(
            obs_dim=obs_dim or obs_arr.shape[1],
            act_dim=act_dim or act_arr.shape[1],
            hidden_sizes=hidden_sizes,
            act_low=act_low,
            act_high=act_high,
        )
        policy_net = ActorCriticMLP(meta)
    else:
        assert policy_net.meta.obs_dim == obs_arr.shape[1], "policy_net obs_dim mismatch with trajectory data"
        assert policy_net.meta.act_dim == act_arr.shape[1], "policy_net act_dim mismatch with trajectory data"

    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
    act_t = torch.as_tensor(act_arr, dtype=torch.float32)

    optimizer = torch.optim.Adam(policy_net.actor_mean.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = obs_t.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred_mean = policy_net.actor_mean(obs_t[idx])
            loss = loss_fn(pred_mean, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    out_path = Path(out_path) if out_path else DEFAULT_CHECKPOINT_DIR / f"{task}_bc.pt"
    policy_net.save(out_path)
    return out_path
