# Sky — teleoperation

Owns `simbiote/teleop/` and `simbiote/sim_stub/`. Tests live in
`Sky/tests/` and run as part of the root `pytest` suite.

## What's here

- `simbiote/teleop/` — the hand-tracking teleop chain:
  - `camera_source.py` — camera acquisition. Takes either an OS device index
    (`$SIMBIOTE_CAMERA_INDEX`) or a stream URL (`$SIMBIOTE_CAMERA_URL`).
  - `hand_tracking.py` — hand-landmark estimation and backend selection.
  - `wilor_backend.py` — the WiLoR backend (GB10/aarch64).
  - `ik_bridge.py` — retargets hand landmarks into `RobotAction` commands
    (base velocity + arm target pose + gripper state). Holds the previous
    pose so a frame with no arm command doesn't snap back to a default.
  - `preview.py` — the "Simbiote Teleop" window: camera feed + hand skeleton
    on the left, the `RobotAction` it retargets to on the right.
  - `action_bridge.py` — UDP transport for `RobotAction`, so teleop can drive
    a simulator running under a different Python interpreter.
  - `teleop_session.py` — ties camera -> hand tracking -> IK -> action
    logging together into one runnable session.

## Driving the hospital robot in Isaac Sim

```bash
scripts/gb10/run_teleop_hospital.sh http://<iphone-ip>:4747/video/640x480
```

Opens Isaac Sim with `hospital.usd` + the Ridgeback/Franka and the webcam
preview window, and lets your hand drive the base and gripper. Full detail in
`docs/TELEOP_ISAAC_HOSPITAL.md`.

Isaac Sim and teleop **must** run as two processes — Isaac's bundled python has
no `cv2`/`ultralytics`/`smplx`, and the `.venv` has no `isaacsim`/`omni`/`pxr`.
`action_bridge.py` is deliberately stdlib-only so it imports under both.

## Running it on the GB10

```bash
./.venv/bin/python run_teleop_demo.py \
    --camera-url http://<iphone-ip>:4747/video/1280x720 \
    --backend wilor --sink pybullet
```

The window opens by default (`--no-preview` for headless). **Pinch switches
modes**: open hand steers the base, pinched parks the base and flies the claw
up and down with the pincer closed. See `ik_bridge.ControlMode` — one hand
can't do both jobs at once without them fighting.

Two things on this box are not preferences but hard constraints, both because
the upstream artifact does not exist for `linux-aarch64`:

- **Hand tracking must use WiLoR, not MediaPipe.** MediaPipe publishes no
  aarch64 wheel. `create_tracker("auto")` resolves to `wilor` here
  automatically. Measured ~11.5 FPS at 1280x720 (the ViT is ~84 ms/frame and
  dominates; the YOLO hand detector is ~10 ms and only runs every 5th frame).
- **The camera must come in over the network, not via Iriun.** Iriun's Linux
  client is an x86-64 binary and cannot execute here. See
  `docs/TELEOP_IPHONE_CAMERA.md`.

WiLoR needs a chumpy-free MANO pickle, because chumpy doesn't import on
Python 3.12. It's already generated at `assets/mano/MANO_RIGHT.pkl`; to
rebuild it:

```bash
./.venv/bin/python scripts/gb10/dechumpify_mano.py --out assets/mano/MANO_RIGHT.pkl
```

Note the teleop chain runs under the repo `.venv`, not Isaac Sim's bundled
Python — the latter lacks `cv2`, `ultralytics`, and `smplx`.
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
- Calibrate `PALM_SCALE_NEAR`/`PALM_SCALE_FAR` in `ik_bridge.py` against the
  actual venue camera and lighting. They're still the laptop-era guesses, and
  they set the arm's forward reach.
- WiLoR predicts metric root-relative 3D joints and a camera translation, so
  forward reach could come from real depth instead of the apparent-palm-size
  proxy. `HandLandmarks.points[:, 2]` currently carries rescaled relative
  depth to stay MediaPipe-compatible; using true depth is a contract change
  worth making once the reach mapping is being tuned anyway.
- Measure end-to-end glass-to-action latency over Wi-Fi before the live demo.
  Model throughput is known (~11.5 FPS); the phone-to-GB10 hop is not.
