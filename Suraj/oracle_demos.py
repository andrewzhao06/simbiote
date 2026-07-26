"""Generate scripted demonstrations for BC warm-starting (nav and grasp).

Spec §5.4's combined learning loop is "BC pretrain on whatever demos exist ->
PPO fine-tune". Teleop demos don't exist yet, and PPO from a cold start does not
find the grasp on its own: the reward for the lift is sparse enough that in 80k
timesteps it peaked at ~4% success. A scripted reach-grasp-lift controller
solves it ~75% of the time, so it makes a perfectly good demonstrator to seed
BC with until real teleop sessions arrive from Step 3.

    $FF_PY Suraj/oracle_demos.py --episodes 60 --out checkpoints/grasp_bc.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simbiote.robot_iface.actions import GripperState, Pose, RobotAction
from simbiote.robot_iface.trajectory import Trajectory, TrajectoryStep
from simbiote.sim_env.grasp_task import GRASP_DIST_THRESHOLD, MAX_EE_STEP, GraspEnv
from simbiote.training.bc_pretrain import train_bc


def rollout(env: GraspEnv, seed: int) -> tuple[Trajectory, bool]:
    """One scripted reach -> close -> lift episode, logged in the shared schema."""
    observation, _ = env.reset(seed=seed)
    steps: list[TrajectoryStep] = []
    holding = False
    info: dict = {}

    for index in range(env.max_steps):
        ee = np.array(env._get_ee_pos())
        obj = np.array(env._object_pos())
        if holding:
            delta = np.array([0.0, 0.0, MAX_EE_STEP])
            closed = True
        else:
            delta = np.clip(obj - ee, -MAX_EE_STEP, MAX_EE_STEP)
            closed = float(np.linalg.norm(obj - ee)) < GRASP_DIST_THRESHOLD

        # Log the *resulting* EE target, not the delta: task_action_vector()
        # rebuilds per-step deltas from consecutive poses.
        target = env._ee_target + delta
        steps.append(
            TrajectoryStep(
                timestamp=float(index) / 60.0,
                action=RobotAction(
                    arm_target_pose=Pose(position=tuple(float(v) for v in target)),
                    gripper_state=GripperState.CLOSED if closed else GripperState.OPEN,
                ),
                source="teleop",
                observation=[float(v) for v in observation],
            )
        )

        observation, _, terminated, truncated, info = env.step(
            np.array([*delta, 1.0 if closed else -1.0], dtype=np.float32)
        )
        holding = holding or info["is_holding"]
        if terminated or truncated:
            break

    return (
        Trajectory(session_id=f"oracle-{seed:04d}", source="teleop", task="grasp", steps=steps),
        bool(info.get("success")),
    )


def nav_rollout(env, seed: int) -> tuple[Trajectory, bool]:
    """Drive at the goal with a repulsion term for nearby obstacles.

    Deliberately simple: a potential-field controller, not a planner. It only
    has to be good enough to demonstrate the behaviour PPO struggles to
    discover on its own.
    """
    observation, _ = env.reset(seed=seed)
    limits = env.robot_config.action_limits
    steps: list[TrajectoryStep] = []
    info: dict = {}

    for index in range(env.max_steps):
        x, y, _ = env._robot_pose()
        goal = np.array([env._goal_xy[0] - x, env._goal_xy[1] - y])
        distance = float(np.linalg.norm(goal))
        heading = goal / max(distance, 1e-6)

        # Push away from any obstacle inside the avoidance radius, weighted by
        # how close it is.
        import pybullet as p

        push = np.zeros(2)
        for obstacle_id in env._obstacle_ids:
            position, _ = p.getBasePositionAndOrientation(
                obstacle_id, physicsClientId=env._client
            )
            away = np.array([x - position[0], y - position[1]])
            gap = float(np.linalg.norm(away))
            if gap < 0.75:
                push += (away / max(gap, 1e-6)) * (0.75 - gap) / 0.75

        direction = heading + 1.6 * push
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
        speed = limits.max_linear_vel * min(1.0, max(distance, 0.15) / 0.5)
        velocity = direction * speed

        steps.append(
            TrajectoryStep(
                timestamp=float(index) / 60.0,
                action=RobotAction(
                    base_velocity=(float(velocity[0]), float(velocity[1]), 0.0)
                ),
                source="teleop",
                observation=[float(v) for v in observation],
            )
        )
        observation, _, terminated, truncated, info = env.step(
            np.array([velocity[0], velocity[1], 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break

    return (
        Trajectory(session_id=f"oracle-nav-{seed:04d}", source="teleop", task="nav", steps=steps),
        bool(info.get("success")),
    )


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["grasp", "nav"], default="grasp")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--out", type=str, default="checkpoints/grasp_bc.pt")
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Train on every rollout instead of only the successful ones",
    )
    args = parser.parse_args(argv)

    if args.task == "nav":
        from simbiote.sim_env.nav_task import NavEnv

        env = NavEnv()
        episode_fn = nav_rollout
        limits = env.robot_config.action_limits
        bounds = (
            (-limits.max_linear_vel, -limits.max_linear_vel, -limits.max_angular_vel),
            (limits.max_linear_vel, limits.max_linear_vel, limits.max_angular_vel),
        )
    else:
        env = GraspEnv()
        episode_fn = rollout
        bounds = (
            (-MAX_EE_STEP, -MAX_EE_STEP, -MAX_EE_STEP, -1.0),
            (MAX_EE_STEP, MAX_EE_STEP, MAX_EE_STEP, 1.0),
        )

    demos: list[Trajectory] = []
    wins = 0
    try:
        for seed in range(args.episodes):
            trajectory, success = episode_fn(env, seed)
            wins += success
            # Regressing onto failed attempts teaches the policy to fail too.
            if success or args.keep_failures:
                demos.append(trajectory)
    finally:
        env.close()

    print(f"[oracle] {wins}/{args.episodes} scripted {args.task} episodes succeeded")
    print(f"[oracle] kept {len(demos)} demonstrations, "
          f"{sum(len(d.steps) for d in demos)} transitions")
    if not demos:
        raise RuntimeError("the scripted controller solved nothing; the task is unwinnable")

    out = train_bc(
        demos,
        task=args.task,
        epochs=args.epochs,
        out_path=args.out,
        act_low=bounds[0],
        act_high=bounds[1],
    )
    print(f"[oracle] BC checkpoint saved to {out}")
    return out


if __name__ == "__main__":
    main()
