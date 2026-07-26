"""CLI: --num_envs, --task, --checkpoint, --rl_lib -- same pattern as
train_nav.py (spec §5.4: "train_grasp.py — same pattern as train_nav.py").

    python -m simbiote.training.train_grasp --num_envs 4 --timesteps 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simbiote.sim_env.register_envs import make_env, register
from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta
from simbiote.training.ppo import PPOConfig, train_ppo

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the manipulation (grasp) policy (spec Part 5, §5.4).")
    parser.add_argument("--task", default="grasp", choices=["grasp"], help="kept for CLI-shape parity with train_nav.py")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--checkpoint", type=str, default=None, help="warm-start weights, e.g. checkpoints/grasp_bc.pt")
    parser.add_argument("--rl_lib", type=str, default="custom", choices=["custom"])
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT_DIR / "grasp_ppo.pt"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> Path:
    args = build_arg_parser().parse_args(argv)
    register()

    def make_env_fn(env_index: int):
        # Each parallel env needs its own seed -- otherwise every worker
        # replays the identical object spawn/pose and `--num_envs > 1`
        # just duplicates transitions instead of adding independent experience.
        def env_fn():
            return make_env("grasp", gui=args.gui, seed=args.seed + env_index)

        return env_fn

    probe_env = make_env_fn(0)()
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    act_low = tuple(probe_env.action_space.low.tolist())
    act_high = tuple(probe_env.action_space.high.tolist())
    probe_env.close()

    if args.checkpoint:
        policy = ActorCriticMLP.load(args.checkpoint)
        print(f"[train_grasp] warm-started from {args.checkpoint}")
    else:
        meta = PolicyMeta(obs_dim=obs_dim, act_dim=act_dim, act_low=act_low, act_high=act_high)
        policy = ActorCriticMLP(meta)
        print("[train_grasp] no --checkpoint given, starting from a fresh random policy")

    config = PPOConfig(total_timesteps=args.timesteps, seed=args.seed)

    def progress(stats: dict) -> None:
        print(
            f"[train_grasp] update={stats['update']} timesteps={stats['timesteps']} "
            f"mean_return={stats['mean_episode_return']:.2f} success_rate={stats['success_rate']:.2f} "
            f"episodes={stats['num_episodes']}"
        )

    trained = train_ppo([make_env_fn(i) for i in range(args.num_envs)], policy, config, progress_callback=progress)

    out_path = Path(args.out)
    trained.save(out_path)
    print(f"[train_grasp] saved checkpoint to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
