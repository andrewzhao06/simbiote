"""Simbiote — Teammate 2 (Step 2): Physics, Simulation & Training.

See simbiote_teammate2_physics_training.md (Part 5) for the full spec this
package implements. Package layout:

    simbiote/
        robot_iface/   shared cross-team schemas (RobotAction, Trajectory) +
                        the navigate_to()/pick_up() skill API this step
                        produces for Step 3 (teleop) and Step 4 (agentic)
        robot/         robot_config.py — joint names, action limits, spawn pose
        sim_env/       Gymnasium task envs (nav, grasp, wheelchair-stretch)
                        + the PyBullet grasp-attach constraint logic
        training/      BC pretrain, PPO fine-tune, play, export, distillation

Today (laptop): PyBullet physics + a hand-built stand-in URDF robot.
Tomorrow (GB10): swap in Isaac Sim + PhysX 5 and the real
`ridgeback_franka.usd` / `hospital.usd` assets — see §5.2/§5.4 of the spec.
"""

__version__ = "0.1.0"
