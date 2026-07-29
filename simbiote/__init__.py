"""Simbiote — offline scan → simulate → train → operate for mobile manipulators.

See `docs/SIMBIOTE_MASTER_PLAN.md` for the full platform plan. Package layout:

    simbiote/
        mapper/        1. Stray Scanner capture -> OpenUSD scene + scene graph
        sim_env/       2. Gymnasium task envs (nav, grasp, wheelchair-stretch),
                          the PyBullet grasp-attach constraint logic, and the
                          Isaac hospital tier
        training/      2. BC pretrain, PPO fine-tune, play, export, distillation
        robot/         2. robot_config.py — joint names, action limits, spawn pose
        robot_iface/      shared cross-team schemas (RobotAction, Pose,
                          Trajectory) + the navigate_to()/pick_up() skill API
        teleop/        3. hand tracking -> RobotAction, plus the UDP bridge into
                          a simulator running under another interpreter
        agentic/       4. natural language -> validated skill plan -> execution
        sim_stub/         toy PyBullet robot for teleop preview
        assets/           URDFs and stand-in scene graphs shipped with the package
        demo_logger.py    shared session logger; every session feeds retraining

Today (laptop): PyBullet physics, MediaPipe hands, a rule-based or local LLM.
Tomorrow (GB10): Isaac Sim + PhysX 5, WiLoR, Nemotron — each behind the same
interface. See the "What runs today vs on the GB10" table in the README.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth is pyproject.toml; keeping a literal here as well
    # is how the two drifted apart (0.1.0 vs 0.3.0) in the first place.
    __version__ = version("simbiote")
except PackageNotFoundError:  # a source tree that was never pip-installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
