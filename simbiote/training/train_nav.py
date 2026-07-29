"""CLI: --num_envs, --task, --checkpoint, --rl_lib -- spec §5.4.

    python -m simbiote.training.train_nav --num_envs 4 --timesteps 20000

`--checkpoint` points at `bc_pretrain.py`'s output when teleop/agentic demos
exist -- PPO then fine-tunes that BC policy via autonomous exploration
against `nav_task.py`'s reward. Without demos yet, omit `--checkpoint` and
it falls back to a fresh random PPO warm-up, exactly as the spec describes.

`--rl_lib` is accepted for interface-compatibility with tomorrow's Isaac Lab
swap (RL-Games/SKRL) but only `custom` (this repo's `training/ppo.py`) is
implemented today.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simbiote.sim_env.register_envs import make_env, register
from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta
from simbiote.training.ppo import PPOConfig, train_ppo

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the navigation policy (spec Part 5, §5.4).")
    parser.add_argument(
        "--task",
        default="nav",
        choices=["nav"],
        help="kept for CLI-shape parity with train_grasp.py",
    )
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="warm-start weights, e.g. checkpoints/nav_bc.pt",
    )
    parser.add_argument(
        "--rl_lib",
        type=str,
        default="custom",
        choices=["custom"],
        help="RL-Games/SKRL land here tomorrow on the GB10",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT_DIR / "nav_ppo.pt"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> Path:
    args = build_arg_parser().parse_args(argv)
    register()

    def make_env_fn(env_index: int):
        # Each parallel env needs its own seed -- otherwise every worker
        # replays the identical obstacle layout/goal and `--num_envs > 1`
        # just duplicates transitions instead of adding independent experience.
        def env_fn():
            return make_env("nav", gui=args.gui, seed=args.seed + env_index)

        return env_fn

    probe_env = make_env_fn(0)()
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    act_low = tuple(probe_env.action_space.low.tolist())
    act_high = tuple(probe_env.action_space.high.tolist())
    probe_env.close()

    if args.checkpoint:
        policy = ActorCriticMLP.load(args.checkpoint)
        print(f"[train_nav] warm-started from {args.checkpoint}")
    else:
        meta = PolicyMeta(obs_dim=obs_dim, act_dim=act_dim, act_low=act_low, act_high=act_high)
        policy = ActorCriticMLP(meta)
        print("[train_nav] no --checkpoint given, starting from a fresh random policy")

    config = PPOConfig(total_timesteps=args.timesteps, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the best policy, not the last one -- see train_grasp.py for the
    # collapse this guards against.
    best = {"score": float("-inf"), "update": 0}

    def progress(stats: dict) -> None:
        print(
            f"[train_nav] update={stats['update']} timesteps={stats['timesteps']} "
            f"mean_return={stats['mean_episode_return']:.2f} "
            f"success_rate={stats['success_rate']:.2f} "
            f"episodes={stats['num_episodes']}"
        )
        score = stats["success_rate"]
        if score != score:  # NaN: no episode finished this update
            score = stats["mean_episode_return"]
        if score == score and score > best["score"]:
            best.update(score=float(score), update=int(stats["update"]))
            # train_ppo mutates `policy` in place, so this is the live network.
            policy.save(out_path)

    trained = train_ppo(
        [make_env_fn(i) for i in range(args.num_envs)], policy, config, progress_callback=progress
    )

    if best["score"] == float("-inf"):
        trained.save(out_path)
        print(f"[train_nav] saved checkpoint to {out_path}")
    else:
        print(
            f"[train_nav] saved best checkpoint to {out_path} "
            f"(update {best['update']}, score {best['score']:.2f})"
        )
    return out_path


if __name__ == "__main__":
    main()
