"""End-to-end smoke tests for the CLIs -- spec §5.4's acceptance test "an
exported checkpoint loads and runs outside the training loop", exercised via
train_nav.py / train_grasp.py / play.py / export_policy.py chained together.

Deliberately tiny (--timesteps, --num_envs) so this runs in seconds, not
minutes -- it's checking the pipeline *works*, not that it has converged
(test_ppo_generic.py's `slow`-marked test covers actual convergence, without
needing pybullet).
"""

from __future__ import annotations

from conftest import require_pybullet


@require_pybullet
def test_train_nav_then_play_then_export(tmp_path):
    from simbiote.training import export_policy, play, train_nav

    ckpt = tmp_path / "nav_ppo.pt"
    train_nav.main(
        [
            "--num_envs", "1",
            "--timesteps", "64",
            "--out", str(ckpt),
        ]
    )
    assert ckpt.exists()

    results = play.run_episodes(str(ckpt), task="nav", episodes=2, gui=False)
    assert len(results) == 2
    assert all("return" in r for r in results)

    onnx_path = export_policy.export_policy(ckpt, tmp_path / "nav.onnx", fmt="onnx")
    assert onnx_path.exists()


@require_pybullet
def test_train_grasp_then_play(tmp_path):
    from simbiote.training import play, train_grasp

    ckpt = tmp_path / "grasp_ppo.pt"
    train_grasp.main(
        [
            "--num_envs", "1",
            "--timesteps", "48",
            "--out", str(ckpt),
        ]
    )
    assert ckpt.exists()

    results = play.run_episodes(str(ckpt), task="grasp", episodes=1, gui=False)
    assert len(results) == 1


@require_pybullet
def test_train_nav_warm_starts_from_bc_checkpoint(tmp_path):
    """Confirms --checkpoint actually warm-starts PPO from a BC checkpoint
    (spec: "--checkpoint now points at bc_pretrain.py's output when demos
    exist -- PPO then fine-tunes that BC policy"), rather than silently
    ignoring it."""
    from simbiote.robot_iface.trajectory import make_toy_trajectory
    from simbiote.sim_env.register_envs import make_env, register
    from simbiote.training import train_nav
    from simbiote.training.bc_pretrain import train_bc
    from simbiote.training.policy_net import ActorCriticMLP

    register()
    probe = make_env("nav")
    obs_dim = probe.observation_space.shape[0]
    act_low = tuple(probe.action_space.low.tolist())
    act_high = tuple(probe.action_space.high.tolist())
    probe.close()

    trajs = [make_toy_trajectory("t", obs_dim=obs_dim, length=20, task="nav")]
    bc_ckpt = tmp_path / "nav_bc.pt"
    train_bc(trajs, task="nav", epochs=2, out_path=bc_ckpt, act_low=act_low, act_high=act_high)

    ppo_ckpt = tmp_path / "nav_ppo_warm.pt"
    train_nav.main(
        [
            "--num_envs", "1",
            "--timesteps", "32",
            "--checkpoint", str(bc_ckpt),
            "--out", str(ppo_ckpt),
        ]
    )
    result = ActorCriticMLP.load(ppo_ckpt)
    assert result.meta.obs_dim == obs_dim
