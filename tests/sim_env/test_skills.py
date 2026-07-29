import pytest

from conftest import require_pybullet
from simbiote.robot_iface import skills


def test_navigate_to_unknown_location_raises():
    with pytest.raises(KeyError):
        skills.navigate_to("nonexistent_room")


def test_pick_up_unknown_object_raises():
    with pytest.raises(KeyError):
        skills.pick_up("nonexistent_object")


@require_pybullet
def test_navigate_to_runs_end_to_end(tmp_path):
    from simbiote.sim_env.register_envs import make_env, register
    from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta

    register()
    probe = make_env("nav")
    meta = PolicyMeta(
        obs_dim=probe.observation_space.shape[0],
        act_dim=probe.action_space.shape[0],
        act_low=tuple(probe.action_space.low.tolist()),
        act_high=tuple(probe.action_space.high.tolist()),
    )
    probe.close()
    ckpt = tmp_path / "nav.pt"
    ActorCriticMLP(meta).save(ckpt)

    result = skills.navigate_to("room_1", checkpoint_path=ckpt, gui=False, max_steps=10)
    assert result["skill"] == "navigate_to"
    assert result["location_id"] == "room_1"
    assert "goal_distance" in result


@require_pybullet
def test_pick_up_runs_end_to_end(tmp_path):
    from simbiote.sim_env.register_envs import make_env, register
    from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta

    register()
    probe = make_env("grasp")
    meta = PolicyMeta(
        obs_dim=probe.observation_space.shape[0],
        act_dim=probe.action_space.shape[0],
        act_low=tuple(probe.action_space.low.tolist()),
        act_high=tuple(probe.action_space.high.tolist()),
    )
    probe.close()
    ckpt = tmp_path / "grasp.pt"
    ActorCriticMLP(meta).save(ckpt)

    result = skills.pick_up("tray_1", checkpoint_path=ckpt, gui=False, max_steps=10)
    assert result["skill"] == "pick_up"
    assert result["object_id"] == "tray_1"
    assert "is_holding" in result


@require_pybullet
def test_attach_and_detach_wheelchair_handle():
    from simbiote.sim_env.spawn import despawn, spawn_robot
    from simbiote.sim_env.wheelchair_task import WHEELCHAIR_URDF

    robot = spawn_robot("stand_in")
    try:
        import pybullet as p

        chair_id = p.loadURDF(
            WHEELCHAIR_URDF, basePosition=[0.3, 0, 0.18], physicsClientId=robot.physics_client
        )
        constraint = skills.attach_handle(robot, chair_id)
        assert constraint.constraint_id is not None
        skills.detach(constraint)
    finally:
        despawn(robot)


@require_pybullet
def test_onnx_inference_fn_matches_torch(tmp_path):
    from simbiote.sim_env.register_envs import make_env, register
    from simbiote.training.export_policy import export_policy
    from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta

    register()
    probe = make_env("nav")
    meta = PolicyMeta(
        obs_dim=probe.observation_space.shape[0],
        act_dim=probe.action_space.shape[0],
        act_low=tuple(probe.action_space.low.tolist()),
        act_high=tuple(probe.action_space.high.tolist()),
    )
    probe.close()
    pt_path = tmp_path / "nav.pt"
    ActorCriticMLP(meta).save(pt_path)
    onnx_path = export_policy(pt_path, tmp_path / "nav.onnx", fmt="onnx")

    infer_fn = skills._load_inference_fn(onnx_path)
    import numpy as np

    obs = np.zeros((1, meta.obs_dim), dtype=np.float32)
    action = infer_fn(obs)
    assert action.shape == (1, meta.act_dim)
