"""A compact, from-scratch PPO loop — spec §5.3 explicitly allows "a plain
PyBullet + Stable-Baselines3 loop"; this is that same scope written directly
against `policy_net.ActorCriticMLP` instead, so BC-pretrained weights load
in and fine-tune with zero architecture-translation glue.

Not trying to be RSL-RL/RL-Games/SKRL — just "nothing exotic needed for a
one-day build" (spec's own words), clipped-objective PPO with GAE(lambda).
Swapping to a real RL library on the GB10 tomorrow means deleting this file,
not touching `nav_task.py`/`grasp_task.py`/`policy_net.py`.

`num_envs` is a plain Python list of independent env instances stepped
sequentially each rollout tick (`VecEnvList`) -- matches the spec's laptop
scale ("1 ... to a small handful via multiprocessing", implemented here
without the multiprocessing for simplicity; swap in true parallelism if a
handful of PyBullet DIRECT clients turns out to be the bottleneck).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from simbiote.training.policy_net import ActorCriticMLP


@dataclass
class PPOConfig:
    total_timesteps: int = 20_000
    rollout_steps: int = 512       # per env, per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 8
    minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    seed: int = 0


class VecEnvList:
    """Minimal synchronous vector-env: N independent envs, stepped in a loop."""

    def __init__(self, env_fns: list[Callable[[], object]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self):
        obs = [env.reset()[0] for env in self.envs]
        return np.stack(obs)

    def reset_one(self, i: int):
        return self.envs[i].reset()[0]

    def step(self, actions: np.ndarray):
        obs, rewards, terminated, truncated, infos = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            o, r, term, trunc, info = env.step(action)
            if term or trunc:
                o = self.reset_one(i)
            obs.append(o)
            rewards.append(r)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(terminated),
            np.array(truncated),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    lam: float,
):
    """rewards/values/dones: (T, N). last_values: (N,). Returns advantages, returns, both (T, N)."""
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train_ppo(
    env_fns: list[Callable[[], object]],
    policy: ActorCriticMLP,
    config: PPOConfig | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> ActorCriticMLP:
    config = config or PPOConfig()
    torch.manual_seed(config.seed)
    vec_env = VecEnvList(env_fns)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    obs = vec_env.reset()
    timesteps_done = 0
    update = 0

    while timesteps_done < config.total_timesteps:
        T, N = config.rollout_steps, vec_env.num_envs
        buf_obs = np.zeros((T, N, policy.meta.obs_dim), dtype=np.float32)
        buf_actions = np.zeros((T, N, policy.meta.act_dim), dtype=np.float32)
        buf_log_probs = np.zeros((T, N), dtype=np.float32)
        buf_rewards = np.zeros((T, N), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)
        buf_values = np.zeros((T, N), dtype=np.float32)
        episode_returns = []
        successes = []
        running_returns = np.zeros(N, dtype=np.float32)

        for t in range(T):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            action_t, log_prob_t, value_t = policy.act(obs_t)
            action_np = action_t.numpy()

            next_obs, rewards, terminated, truncated, infos = vec_env.step(action_np)
            dones = np.logical_or(terminated, truncated)

            buf_obs[t] = obs
            buf_actions[t] = action_np
            buf_log_probs[t] = log_prob_t.numpy()
            buf_rewards[t] = rewards
            buf_dones[t] = dones.astype(np.float32)
            buf_values[t] = value_t.numpy()

            running_returns += rewards
            for i, d in enumerate(dones):
                if d:
                    episode_returns.append(running_returns[i])
                    successes.append(bool(infos[i].get("success", False)))
                    running_returns[i] = 0.0

            obs = next_obs
            timesteps_done += N

        with torch.no_grad():
            last_values = policy.forward(torch.as_tensor(obs, dtype=torch.float32))[2].numpy()

        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_values, config.gamma, config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_obs = torch.as_tensor(buf_obs.reshape(T * N, -1), dtype=torch.float32)
        flat_actions = torch.as_tensor(buf_actions.reshape(T * N, -1), dtype=torch.float32)
        flat_log_probs = torch.as_tensor(buf_log_probs.reshape(T * N), dtype=torch.float32)
        flat_advantages = torch.as_tensor(advantages.reshape(T * N), dtype=torch.float32)
        flat_returns = torch.as_tensor(returns.reshape(T * N), dtype=torch.float32)

        n_samples = T * N
        for _ in range(config.train_iters):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, config.minibatch_size):
                idx = perm[start : start + config.minibatch_size]
                new_log_probs, entropy, values = policy.evaluate_actions(
                    flat_obs[idx], flat_actions[idx]
                )
                ratio = torch.exp(new_log_probs - flat_log_probs[idx])
                clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
                policy_loss = -torch.min(
                    ratio * flat_advantages[idx], clipped * flat_advantages[idx]
                ).mean()
                value_loss = ((values - flat_returns[idx]) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    + config.entropy_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        update += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "update": update,
                    "timesteps": timesteps_done,
                    "mean_episode_return": float(np.mean(episode_returns))
                    if episode_returns
                    else float("nan"),
                    "success_rate": float(np.mean(successes)) if successes else float("nan"),
                    "num_episodes": len(episode_returns),
                }
            )

    vec_env.close()
    return policy
