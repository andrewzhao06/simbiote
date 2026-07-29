"""Guards that the RL tasks are actually winnable.

Both cases here were live bugs where PPO reported a perfectly plausible-looking
training curve against an objective it could never reach. A success rate that
converges to zero is indistinguishable from "needs more timesteps" unless
something asserts the task is solvable in the first place.
"""

from __future__ import annotations

import numpy as np

from conftest import require_pybullet


@require_pybullet
class TestNavEpisodesAreDecidedByThePolicy:
    """NavEnv used to place obstacles and goals by unconstrained uniform draw,
    so ~23% of episodes ended in a collision on step 1 and ~3% started already
    inside the goal. A quarter of every success rate was decided at reset."""

    def test_reset_does_not_hand_out_free_wins_or_instant_collisions(self):
        from simbiote.sim_env.nav_task import NavEnv

        env = NavEnv()
        try:
            collisions = successes = 0
            episodes = 40
            for seed in range(episodes):
                env.reset(seed=seed)
                # A do-nothing action: anything that happens is the reset's doing.
                _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
                collisions += bool(info["collided"])
                successes += bool(info["success"])
            assert collisions <= episodes * 0.05, (
                f"{collisions}/{episodes} episodes collide before the policy acts"
            )
            assert successes == 0, (
                f"{successes}/{episodes} episodes start inside the goal threshold"
            )
        finally:
            env.close()

    def test_goal_is_far_enough_to_require_traversal(self):
        from simbiote.sim_env.nav_task import NavEnv

        env = NavEnv()
        try:
            for seed in range(20):
                env.reset(seed=seed)
                spawn = env.robot_config.default_spawn_pose.position
                distance = float(
                    np.hypot(env._goal_xy[0] - spawn[0], env._goal_xy[1] - spawn[1])
                )
                assert distance >= env.min_goal_distance - 1e-6
                assert distance > env.goal_threshold
        finally:
            env.close()


@require_pybullet
class TestGraspIsReachable:
    """The stand-in arm's two positioning joints both rotated about Y, making it
    planar: end-effector world y was pinned at 0.000 while objects spawn at
    y in [-0.2, 0.2]. Separately the attach threshold was tighter than the
    collision geometry allows. Either alone makes success impossible."""

    def test_end_effector_reaches_objects_spawned_off_the_xz_plane(self):
        """The planar-arm bug showed up as: objects with y != 0 were unreachable.

        Tested by closing the loop on the object itself rather than on a fixed
        waypoint -- position-only IK on this stand-in wanders when asked to hold
        a static pose, which is a separate (and tolerated, per spec 5.4
        "don't chase accuracy") limitation and would make this a flaky proxy.
        """
        from simbiote.sim_env.grasp_task import (
            GRASP_DIST_THRESHOLD,
            MAX_EE_STEP,
            GraspEnv,
        )

        for lateral in (0.18, -0.18):
            env = GraspEnv(object_position_override=(0.45, lateral, 0.25))
            try:
                env.reset(seed=0)
                closest = 9.9
                for _ in range(env.max_steps):
                    ee = np.array(env._get_ee_pos())
                    obj = np.array(env._object_pos())
                    closest = min(closest, float(np.linalg.norm(ee - obj)))
                    delta = np.clip(obj - ee, -MAX_EE_STEP, MAX_EE_STEP)
                    env.step(np.array([*delta, -1.0], dtype=np.float32))
                assert closest < GRASP_DIST_THRESHOLD, (
                    f"EE got no closer than {closest:.3f} m to an object at "
                    f"y={lateral}; the arm cannot reach off the x-z plane"
                )
            finally:
                env.close()

    def test_attach_threshold_is_geometrically_achievable(self):
        """Press the EE into the object and check it can actually trigger a grasp."""
        from simbiote.sim_env.grasp_task import (
            GRASP_DIST_THRESHOLD,
            MAX_EE_STEP,
            GraspEnv,
        )

        env = GraspEnv()
        try:
            closest = 9.9
            env.reset(seed=7)
            for _ in range(env.max_steps):
                ee = np.array(env._get_ee_pos())
                obj = np.array(env._object_pos())
                closest = min(closest, float(np.linalg.norm(ee - obj)))
                delta = np.clip(obj - ee, -MAX_EE_STEP, MAX_EE_STEP)
                env.step(np.array([*delta, -1.0], dtype=np.float32))
            assert closest < GRASP_DIST_THRESHOLD, (
                f"EE bottoms out at {closest:.3f} m but attach needs "
                f"{GRASP_DIST_THRESHOLD:.3f} m -- no grasp can ever trigger"
            )
        finally:
            env.close()

    def test_a_scripted_oracle_can_complete_the_task(self):
        """If a hand-written reach-grasp-lift controller cannot win, a zero
        success rate from PPO says nothing about the policy."""
        from simbiote.sim_env.grasp_task import (
            GRASP_DIST_THRESHOLD,
            MAX_EE_STEP,
            GraspEnv,
        )

        env = GraspEnv()
        try:
            wins = 0
            episodes = 8
            for seed in range(episodes):
                env.reset(seed=seed)
                holding = False
                info: dict = {}
                for _ in range(env.max_steps):
                    ee = np.array(env._get_ee_pos())
                    obj = np.array(env._object_pos())
                    if holding:
                        delta, grip = np.array([0.0, 0.0, MAX_EE_STEP]), 1.0
                    else:
                        distance = float(np.linalg.norm(obj - ee))
                        delta = np.clip(obj - ee, -MAX_EE_STEP, MAX_EE_STEP)
                        grip = 1.0 if distance < GRASP_DIST_THRESHOLD else -1.0
                    _, _, terminated, truncated, info = env.step(
                        np.array([*delta, grip], dtype=np.float32)
                    )
                    holding = holding or info["is_holding"]
                    if terminated or truncated:
                        break
                wins += bool(info.get("success"))
            assert wins >= episodes // 2, (
                f"scripted oracle only solved {wins}/{episodes}; the task is not "
                "reliably winnable, so PPO's success rate is not measuring learning"
            )
        finally:
            env.close()
