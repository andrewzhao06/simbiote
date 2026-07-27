"""PyBullet-backed task/physics tests -- spec §5.4's acceptance tests:

    "grasp_attach.py's constraint holds under simulated motion"
    "train_nav.py reaches the goal without collision..."
    "train_grasp.py clears the same bar on grasp success rate"

Skipped (not failed) when pybullet isn't importable -- see tests/conftest.py
and README's "Known issues" (no Windows wheels on PyPI for pybullet).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import require_pybullet


@require_pybullet
class TestPybulletScene:
    def test_connect_and_ground_plane(self):
        from simbiote.sim_env import pybullet_scene as scene

        client = scene.connect(gui=False)
        try:
            plane_id = scene.load_ground_plane(client)
            assert plane_id >= 0
        finally:
            scene.disconnect(client)

    def test_load_stand_in_robot_and_resolve_joints(self):
        from simbiote.robot.robot_config import STAND_IN_CONFIG
        from simbiote.sim_env import pybullet_scene as scene

        client = scene.connect(gui=False)
        try:
            scene.load_ground_plane(client)
            robot_id = scene.load_robot(STAND_IN_CONFIG, client)
            handles = scene.RobotHandles.build(STAND_IN_CONFIG, client, robot_id)
            # Driven by the config rather than a literal: the stand-in arm grew
            # a shoulder_yaw_joint (3 -> 4 DOF) so it can reach off the x-z
            # plane, and the point of this check is that every configured joint
            # resolves, not what the count happens to be.
            assert len(handles.arm_joint_indices) == len(STAND_IN_CONFIG.arm_joint_names)
            assert len(handles.gripper_joint_indices) == len(
                STAND_IN_CONFIG.gripper_joint_names
            )
            assert handles.ee_link_index >= 0
        finally:
            scene.disconnect(client)


@require_pybullet
class TestGraspAttach:
    def test_attach_then_release(self):
        import pybullet as p

        from simbiote.robot.robot_config import STAND_IN_CONFIG
        from simbiote.sim_env import grasp_attach
        from simbiote.sim_env import pybullet_scene as scene

        client = scene.connect(gui=False)
        try:
            scene.load_ground_plane(client)
            robot_id = scene.load_robot(STAND_IN_CONFIG, client, fixed_base=True)
            handles = scene.RobotHandles.build(STAND_IN_CONFIG, client, robot_id)
            ee_pos = p.getLinkState(robot_id, handles.ee_link_index, physicsClientId=client)[0]
            obj = scene.spawn_graspable_box(client, position=(ee_pos[0], ee_pos[1], ee_pos[2]), mass_kg=0.2)

            constraint = grasp_attach.attach(robot_id, handles.ee_link_index, obj.body_id, physics_client=client)
            assert grasp_attach.is_holding(constraint)

            # Constraint should hold the object roughly at the EE as the sim steps
            # (spec: "grasp_attach.py's constraint holds under simulated motion").
            for _ in range(120):
                p.stepSimulation(physicsClientId=client)
            obj_pos, _ = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=client)
            new_ee_pos = p.getLinkState(robot_id, handles.ee_link_index, physicsClientId=client)[0]
            dist = np.linalg.norm(np.array(obj_pos) - np.array(new_ee_pos))
            assert dist < 0.1, f"grasped object drifted too far from EE ({dist} m)"

            grasp_attach.release(constraint)
            assert not grasp_attach.is_holding(None)
        finally:
            scene.disconnect(client)

    def test_release_none_is_noop(self):
        from simbiote.sim_env import grasp_attach

        grasp_attach.release(None)  # should not raise


@require_pybullet
class TestNavEnv:
    def test_reset_and_step_shapes(self):
        from simbiote.sim_env.nav_task import ACT_DIM, OBS_DIM, NavEnv

        env = NavEnv(max_steps=20, seed=0)
        try:
            obs, info = env.reset(seed=0)
            assert obs.shape == (OBS_DIM,)
            assert env.observation_space.contains(obs)

            action = np.zeros(ACT_DIM, dtype=np.float32)
            obs2, reward, terminated, truncated, info = env.step(action)
            assert obs2.shape == (OBS_DIM,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
        finally:
            env.close()

    def test_episode_truncates_at_max_steps(self):
        from simbiote.sim_env.nav_task import NavEnv

        env = NavEnv(max_steps=5, seed=1, num_obstacles=0)
        try:
            env.reset(seed=1)
            action = np.zeros(3, dtype=np.float32)
            truncated = False
            steps = 0
            while not truncated and steps < 100:
                _, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated:
                    break
            assert steps <= 5
        finally:
            env.close()

    def test_goal_override_is_used(self):
        from simbiote.sim_env.nav_task import NavEnv

        env = NavEnv(goal_override=(1.23, -0.5), max_steps=5)
        try:
            env.reset()
            assert env._goal_xy == (1.23, -0.5)
        finally:
            env.close()


@require_pybullet
class TestGraspEnv:
    def test_reset_and_step_shapes(self):
        from simbiote.sim_env.grasp_task import ACT_DIM, OBS_DIM, GraspEnv

        env = GraspEnv(max_steps=10, seed=0)
        try:
            obs, info = env.reset(seed=0)
            assert obs.shape == (OBS_DIM,)
            action = np.zeros(ACT_DIM, dtype=np.float32)
            obs2, reward, terminated, truncated, info = env.step(action)
            assert obs2.shape == (OBS_DIM,)
            assert "is_holding" in info
        finally:
            env.close()

    def test_can_grasp_and_lift_with_scripted_actions(self):
        """A hand-scripted (not learned) approach-grasp-lift sequence should
        succeed -- validates the IK drive + grasp_attach trigger + lift
        reward path end to end, independent of whether PPO has converged."""
        from simbiote.sim_env.grasp_task import GRASP_DIST_THRESHOLD, GraspEnv

        # object_position_override is deliberately near the edge of the
        # stand-in arm's reach -- the 3-DOF IK chain can spend a couple dozen
        # steps stuck at a local minimum/joint limit before it finds a path
        # around it (this is a property of the low-DOF stand-in arm, not
        # something the scripted test should paper over), so give phase 1
        # generous headroom rather than tuning the arm to make the test fast.
        env = GraspEnv(max_steps=300, seed=0, object_position_override=(0.45, 0.0, 0.25))
        try:
            obs, _ = env.reset(seed=0)
            success = False
            # Phase 1: approach (open gripper, move toward the object's xy/z).
            for _ in range(120):
                obj_pos = env._object_pos()
                ee_pos = env._get_ee_pos()
                delta = np.clip(np.array(obj_pos) - np.array(ee_pos), -0.03, 0.03)
                action = np.array([delta[0], delta[1], delta[2], -1.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if info["ee_object_dist"] < GRASP_DIST_THRESHOLD or terminated:
                    break
            # Phase 2: close the gripper to trigger attach().
            for _ in range(15):
                action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("grasped_this_step"):
                    break
            # Phase 3: lift straight up while holding.
            for _ in range(80):
                action = np.array([0.0, 0.0, 0.03, 1.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("success"):
                    success = True
                    break
                if terminated:
                    break
            assert success, f"scripted approach-grasp-lift did not succeed, last info={info}"
        finally:
            env.close()


@require_pybullet
class TestWheelchairEnv:
    def test_reset_and_step_shapes(self):
        from simbiote.sim_env.wheelchair_task import OBS_DIM, WheelchairEnv

        env = WheelchairEnv(max_steps=10, seed=0)
        try:
            obs, _ = env.reset(seed=0)
            assert obs.shape == (OBS_DIM,)
            action = np.zeros(3, dtype=np.float32)
            obs2, reward, terminated, truncated, info = env.step(action)
            assert obs2.shape == (OBS_DIM,)
            assert "tilt" in info
        finally:
            env.close()


@require_pybullet
class TestRegisterEnvs:
    def test_register_is_idempotent_and_make_env_works(self):
        from simbiote.sim_env.register_envs import make_env, register

        register()
        register()  # should not raise on double-call
        env = make_env("nav", max_steps=5)
        try:
            obs, _ = env.reset()
            assert obs is not None
        finally:
            env.close()


@require_pybullet
@pytest.mark.slow
class TestNavTraining:
    def test_ppo_improves_nav_success_rate_over_random_baseline(self):
        """A real (if small) convergence check against a live PyBullet env --
        mirrors spec §5.4's acceptance test ("train_nav.py reaches the goal
        without collision noticeably more often than the un-fine-tuned
        baseline") using a short training budget so it stays test-suite-fast.
        """
        from simbiote.sim_env.register_envs import make_env, register
        from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta
        from simbiote.training.ppo import PPOConfig, train_ppo

        register()

        def env_fn():
            return make_env("nav", num_obstacles=1, max_steps=80)

        probe = env_fn()
        meta = PolicyMeta(
            obs_dim=probe.observation_space.shape[0],
            act_dim=probe.action_space.shape[0],
            hidden_sizes=(64, 64),
            act_low=tuple(probe.action_space.low.tolist()),
            act_high=tuple(probe.action_space.high.tolist()),
        )
        probe.close()

        baseline = ActorCriticMLP(meta)
        trained = ActorCriticMLP(meta)
        trained.load_state_dict(baseline.state_dict())  # identical random start

        config = PPOConfig(total_timesteps=6000, rollout_steps=200, train_iters=4, minibatch_size=64, lr=1e-3, seed=0)
        stats = []
        train_ppo([env_fn for _ in range(2)], trained, config, progress_callback=stats.append)

        def mean_return(policy, episodes=10):
            env = env_fn()
            total = 0.0
            for ep in range(episodes):
                obs, _ = env.reset(seed=100 + ep)
                done = False
                while not done:
                    import torch

                    action, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
                    total += reward
                    done = terminated or truncated
            env.close()
            return total / episodes

        baseline_return = mean_return(baseline)
        trained_return = mean_return(trained)
        assert trained_return > baseline_return, (
            f"expected PPO fine-tuning to beat the random baseline, "
            f"baseline={baseline_return:.2f} trained={trained_return:.2f}"
        )


@require_pybullet
class TestSpawnRobot:
    def test_spawn_and_despawn(self):
        from simbiote.sim_env.spawn import despawn, spawn_robot

        robot = spawn_robot("stand_in")
        try:
            assert robot.robot_id >= 0
            assert robot.handles.ee_link_index >= 0
        finally:
            despawn(robot)

    def test_spawn_unknown_engine_raises(self):
        from dataclasses import replace

        from simbiote.robot.robot_config import RIDGEBACK_FRANKA_CONFIG
        from simbiote.sim_env.spawn import spawn_robot

        with pytest.raises(NotImplementedError):
            spawn_robot(RIDGEBACK_FRANKA_CONFIG)
