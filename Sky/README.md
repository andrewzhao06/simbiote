# Sky — teleoperation

Owns `simbiote/teleop/` and `simbiote/sim_stub/`. Tests live in
`Sky/tests/` and run as part of the root `pytest` suite.

## What's here

- `simbiote/teleop/` — the hand-tracking teleop chain:
  - `camera_source.py` — camera acquisition (`$SIMBIOTE_CAMERA_INDEX` to pick
    a device).
  - `hand_tracking.py` — MediaPipe hand-landmark estimation.
  - `ik_bridge.py` — retargets hand landmarks into `RobotAction` commands
    (base velocity + arm target pose + gripper state). Holds the previous
    pose so a frame with no arm command doesn't snap back to a default.
  - `teleop_session.py` — ties camera -> hand tracking -> IK -> action
    logging together into one runnable session ("Simbiote Teleop" window).
- `simbiote/sim_stub/` — `toy_robot_env.py` + `toy_robot.urdf`, a minimal
  PyBullet robot used to visualize teleop output without needing the full
  training sim environments.
- `run_teleop_demo.py` (repo root) — convenience entry point, equivalent to
  `python -m simbiote.teleop.teleop_session` / the `simbiote-teleop` console
  script (`pyproject.toml`). See `teleop_session.py`'s argparse setup for
  options (`--sink console|pybullet`, `--camera-index`, `--preview`, `--save`).

Every action produced by a teleop session is logged via
`simbiote.demo_logger.log_action(action, "teleop", ...)`, so teleop demos and
Andrew's agentic demos are trainable through the exact same
`training/bc_pretrain.py` / `retrain.py` path.

## Tests

`Sky/tests/test_teleop_logic.py` — smoke tests for the retargeting logic and
`demo_logger` integration (neutral action shape, session start/log/export).

## GB10 next steps

- Swap `toy_robot_env.py` for the real Isaac Sim robot once Suraj's Isaac
  Sim environments land, keeping `ik_bridge.py`'s output contract
  (`RobotAction`) unchanged.
- Confirm camera latency and MediaPipe throughput on the GB10's hardware
  before the live demo.
