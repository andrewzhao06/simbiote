"""Exercises training/ppo.py end-to-end against a standard Gymnasium env
(Pendulum-v1, ships with gymnasium itself -- no pybullet/extra deps needed)
instead of our PyBullet task envs, so the PPO *algorithm* is fully tested
even on machines without pybullet (e.g. this repo's dev Windows box -- see
README "Known issues"). `tests/sim_env/test_sim_env.py` covers the PyBullet-specific
task logic (nav_task/grasp_task/wheelchair_task/grasp_attach) separately,
gated on pybullet being importable.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

# Imported after the skip so the module still collects where gymnasium is absent.
from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta  # noqa: E402
from simbiote.training.ppo import PPOConfig, VecEnvList, compute_gae, train_ppo  # noqa: E402


def _make_pendulum():
    return gym.make("Pendulum-v1")


def test_vec_env_list_reset_and_step_shapes():
    vec = VecEnvList([_make_pendulum, _make_pendulum])
    obs = vec.reset()
    assert obs.shape == (2, 3)

    actions = np.zeros((2, 1), dtype=np.float32)
    obs2, rewards, _terminated, _truncated, infos = vec.step(actions)
    assert obs2.shape == (2, 3)
    assert rewards.shape == (2,)
    assert len(infos) == 2
    vec.close()


def test_compute_gae_shapes_and_zero_reward_case():
    T, N = 5, 3
    rewards = np.zeros((T, N), dtype=np.float32)
    values = np.zeros((T, N), dtype=np.float32)
    dones = np.zeros((T, N), dtype=np.float32)
    last_values = np.zeros(N, dtype=np.float32)

    advantages, returns = compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95)
    assert advantages.shape == (T, N)
    assert returns.shape == (T, N)
    np.testing.assert_allclose(advantages, 0.0, atol=1e-6)


def test_compute_gae_positive_reward_gives_positive_advantage():
    T, N = 4, 1
    rewards = np.ones((T, N), dtype=np.float32)
    values = np.zeros((T, N), dtype=np.float32)
    dones = np.zeros((T, N), dtype=np.float32)
    last_values = np.zeros(N, dtype=np.float32)

    advantages, _returns = compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95)
    assert np.all(advantages > 0)


def test_train_ppo_smoke_runs_without_crashing():
    """Not a convergence test -- just confirms the full rollout/update loop
    runs end to end on a real Gymnasium env and returns an updated policy."""
    meta = PolicyMeta(obs_dim=3, act_dim=1, hidden_sizes=(16, 16), act_low=(-2.0,), act_high=(2.0,))
    policy = ActorCriticMLP(meta)
    config = PPOConfig(
        total_timesteps=64, rollout_steps=32, train_iters=1, minibatch_size=16, seed=0
    )

    updates_seen = []
    trained = train_ppo(
        [_make_pendulum, _make_pendulum], policy, config, progress_callback=updates_seen.append
    )

    assert trained is policy
    assert len(updates_seen) >= 1
    assert updates_seen[0]["timesteps"] >= 64


@pytest.mark.slow
def test_train_ppo_improves_mean_return_on_pendulum():
    """A real (if small) convergence check: PPO should improve Pendulum-v1's
    mean episodic return over a modest number of updates, starting from a
    fresh random policy -- mirrors the spirit of §5.4's acceptance test
    ("reaches the goal noticeably more often than the un-fine-tuned
    baseline") without needing pybullet installed to verify the algorithm.
    """
    meta = PolicyMeta(obs_dim=3, act_dim=1, hidden_sizes=(32, 32), act_low=(-2.0,), act_high=(2.0,))
    policy = ActorCriticMLP(meta)
    config = PPOConfig(
        total_timesteps=1536, rollout_steps=256, train_iters=6, minibatch_size=64, lr=3e-3, seed=0
    )

    stats = []
    train_ppo([_make_pendulum for _ in range(4)], policy, config, progress_callback=stats.append)

    first_return = stats[0]["mean_episode_return"]
    last_return = stats[-1]["mean_episode_return"]
    assert last_return > first_return, (
        f"expected improvement, got first={first_return} last={last_return}"
    )
