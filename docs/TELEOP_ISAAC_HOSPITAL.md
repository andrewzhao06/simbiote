# Hand-teleoperating the hospital robot in Isaac Sim

Drives the Ridgeback+Franka through `hospital.usd` with your hand, via the
iPhone camera. Two windows open: Isaac Sim showing the hospital and the robot,
and "Simbiote Teleop" showing your webcam with the tracked hand skeleton and
the `RobotAction` it produces.

## Run it

```bash
cd ~/simbiote-Gagan1_Suraj2_Sky3_Andrew4
scripts/gb10/run_teleop_hospital.sh http://<iphone-ip>:4747/video/640x480
```

That script starts both halves and shuts both down on Ctrl+C. It checks the
camera first, so an unreachable phone fails immediately instead of after a
multi-minute Isaac boot.

To run them separately (useful when iterating — you avoid re-booting Isaac):

```bash
# terminal 1 -- the simulator. First boot takes a few minutes.
/home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
    scripts/gb10/teleop_hospital.py

# terminal 2 -- camera, hand tracking, preview window
./.venv/bin/python run_teleop_demo.py --sink udp \
    --camera-url http://<iphone-ip>:4747/video/640x480
```

Isaac prints `listening for teleop on udp://...` when it's ready. Start teleop
after that; anything sent earlier goes nowhere.

## Controls

Pinching switches modes. One hand can't steer a base and pose an arm at the
same time without the two fighting — tilting to steer would drag the gripper
with it. The preview window shows a green **DRIVE** or red **MANIPULATE**
banner so the current mode is never ambiguous.

**Open hand — DRIVE the base:**

| Hand | Robot |
| --- | --- |
| Above frame centre | base drives forward |
| Below frame centre | base drives backward |
| Left / right of centre | base turns |
| Inside the centre box | base stationary (deadzone) |
| Out of frame | base stops, arm held |

**Pinched — MANIPULATE the claw:**

| Hand | Robot |
| --- | --- |
| Pinch thumb + index | base parks, pincer closes |
| Hand raised / lowered | claw moves up / down |
| Open the hand again | back to DRIVE |

Pinch uses separate enter (0.35) and release (0.50) thresholds. A single
threshold makes the mode flicker whenever a pinch hovers near it, which reads
as the robot twitching between driving and manipulating.

A tracking dropout mid-grasp deliberately does **not** reset the mode — a
momentary glitch shouldn't open the gripper and drop whatever is being carried.

Forward means **wherever the robot is facing** (`--frame body`, the default).
`IsaacHospital._apply_velocity` is world-frame because that's what NavEnv's
policy emits; for hand driving that's wrong — after turning 90° "forward"
would still push along world +X. `--frame world` restores the raw behaviour.

## Why it's two processes

The two halves cannot share an interpreter on this box:

- Teleop needs `cv2`, `ultralytics`, `smplx` — in the repo `.venv`, absent
  from Isaac Sim's bundled python.
- `IsaacHospital` needs `isaacsim`/`omni`/`pxr` — only inside Isaac Sim's
  bundled python, not pip-installable into a venv.

So teleop stays in the `.venv` and speaks to Isaac over UDP
(`simbiote/teleop/action_bridge.py`). UDP because teleop is a stream of
*states*, not events: only the newest hand pose matters, a dropped frame beats
a late one, and a slow simulator can never stall the camera loop. The receiver
drains its socket each tick and keeps only the last datagram.

The alternative — pip-installing OpenCV and ultralytics into the Kit runtime —
would risk shifting dependencies underneath Isaac Sim for no gain.

## Safety behaviour

If the teleop process stops sending (phone backgrounded, window closed, process
killed), the receiver's actions go stale after 0.5 s and the Isaac loop commands
a **full stop** rather than letting the base coast on its last velocity. You'll
see `teleop stream went quiet -- base stopped`.

## What is and isn't teleoperated

**Is:** base velocity (drive + turn), the Franka's gripper, and the claw's
*height*.

**Isn't:** full 6-DOF arm posing. The claw moves up and down, not sideways and
not in orientation.

### How the claw height works

Isaac Sim's Lula/RMPflow solvers need robot-descriptor YAMLs this asset pack
doesn't ship, and cuRobo (staged at `/home/dell/AI/repos/curobo`) isn't wired
up. Rather than fake it with a hand-tuned joint mapping,
`simbiote/sim_env/arm_lift.py` measures the one row of the Jacobian the task
actually needs: at startup it nudges each arm joint and watches how far the
end-effector moves in world z, then solves the 1-DOF task with the
pseudo-inverse of that row. Same "measure, don't assume" approach
`IsaacHospital._calibrate_base_axes` uses for the base axes.

The measurement validates itself. On the Ridgeback's Franka:

```
dz/dq = [-0.0, -0.132, -0.0, 0.45, 0.0, 0.122, 0.0]
```

The three roll joints come out at exactly zero — they physically cannot change
the end-effector's height — and the elbow dominates. None of that is hardcoded.

**Limitation:** a Jacobian is only valid locally, and these sensitivities are
measured once, at the folded pose. Joint travel is therefore clamped to
`MAX_JOINT_TRAVEL` (0.9 rad) around it, giving about `LIFT_RANGE` = 0.45 m of
vertical claw travel centred on the fold height (~0.89 m). Enough to raise and
lower the claw; not enough to reach across a room. Full arm posing wants
cuRobo, and would re-solve the Jacobian every tick rather than once.

## Measured behaviour

Validated headless by feeding `IKBridge` output over the bridge:

- Base drove 7.81 m → 13.84 m across the hospital under hand-derived velocity,
  533 actions received, 0 dropped.
- Claw servoed 0.89 m (home) → 1.11 m (commanded high) → 0.66 m (commanded
  low), with the base parked throughout — confirming manipulate mode stops the
  base as well as moving the arm.
- Constant "forward + turn" traces a circle under `--frame body` and a straight
  world +X line under `--frame world`, as expected.
- Simulator ran ~23 control ticks/s headless; teleop produces ~14 actions/s, so
  the sim is not the bottleneck.
- Stopping the sender stopped the base, which then held position.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Isaac prints `idle` forever | Teleop isn't running, or is on a different `--udp-port`. |
| `Isaac Sim exited during startup` | Check the log — usually GPU memory. Another Isaac instance may already be running (`pgrep -af isaac`). |
| Robot drifts with no hand in frame | Expected only briefly: the EMA smoother decays the last velocity. It zeroes within a few frames. |
| Robot moves the wrong way when you turn | You're on `--frame world`. Drop the flag. |
| Robot barely moves | Lower `--speed-scale` bounds, or your hand is inside the deadzone. Move further off-centre. |
| Isaac boot killed after ~2 min | It needs longer than that. Launch it detached (`setsid nohup ... &`), not inside a command with a short timeout. |
