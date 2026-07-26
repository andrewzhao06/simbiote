"""The shared MLP actor-critic architecture — spec §5.4:

    "Same policy network (a standard MLP actor-critic -- the RSL-RL/RL-Games/
    SKRL default for this observation/action size, nothing exotic needed for
    a one-day build) receives updates from both sources [BC and PPO] rather
    than keeping them as two separate models."

One `ActorCriticMLP` class is used by `bc_pretrain.py` (supervised warm
start) and `train_nav.py`/`train_grasp.py` (PPO fine-tune) — a checkpoint
saved by one loads straight into the other, by construction.

Continuous, diagonal-Gaussian action distribution: policy outputs a mean per
action dim, plus a single learned (state-independent) log-std vector — the
standard, simplest-that-works choice for this kind of low-dimensional
control task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim: int, out_dim: int, hidden_sizes: Sequence[int], out_activation: nn.Module | None = None) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


@dataclass
class PolicyMeta:
    obs_dim: int
    act_dim: int
    hidden_sizes: Tuple[int, ...] = (128, 128)
    act_low: Tuple[float, ...] | None = None
    act_high: Tuple[float, ...] | None = None


class ActorCriticMLP(nn.Module):
    def __init__(self, meta: PolicyMeta):
        super().__init__()
        self.meta = meta
        self.actor_mean = _mlp(meta.obs_dim, meta.act_dim, meta.hidden_sizes)
        self.log_std = nn.Parameter(torch.zeros(meta.act_dim) - 0.5)
        self.critic = _mlp(meta.obs_dim, 1, meta.hidden_sizes)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def distribution(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean, std, value = self.forward(obs)
        return Normal(mean, std), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        if self.meta.act_low is not None:
            low = torch.as_tensor(self.meta.act_low, dtype=obs.dtype)
            high = torch.as_tensor(self.meta.act_high, dtype=obs.dtype)
            action = torch.clamp(action, low, high)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, value = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": self.meta, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "ActorCriticMLP":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(checkpoint["meta"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
