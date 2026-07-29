"""Loads a checkpoint for visual inspection — spec §5.4.

    python -m simbiote.training.play --checkpoint checkpoints/nav_ppo.pt --task nav --gui

Also usable headless (default) as the "does an exported/trained checkpoint
actually run outside the training loop" acceptance check from §5.4.
"""

from __future__ import annotations

import argparse

import numpy as np

from simbiote.sim_env.register_envs import make_env, register
from simbiote.training.policy_net import ActorCriticMLP


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play back a trained/BC checkpoint (spec Part 5, §5.4)."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["nav", "grasp", "wheelchair"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser


def run_episodes(
    checkpoint: str, task: str, episodes: int = 3, gui: bool = False, deterministic: bool = True
) -> list[dict]:
    import torch

    register()
    env = make_env(task, gui=gui)
    policy = ActorCriticMLP.load(checkpoint)

    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        steps = 0
        info = {}
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_t, _, _ = policy.act(obs_t, deterministic=deterministic)
            action = action_t.squeeze(0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        results.append({"episode": ep, "return": total_reward, "steps": steps, **info})
    env.close()
    return results


def main(argv=None) -> list[dict]:
    args = build_arg_parser().parse_args(argv)
    results = run_episodes(args.checkpoint, args.task, args.episodes, args.gui, args.deterministic)
    for r in results:
        print(
            f"[play] episode {r['episode']}: return={r['return']:.2f} "
            f"steps={r['steps']} success={r.get('success')}"
        )
    mean_return = float(np.mean([r["return"] for r in results]))
    print(f"[play] mean return over {len(results)} episodes: {mean_return:.2f}")
    return results


if __name__ == "__main__":
    main()
