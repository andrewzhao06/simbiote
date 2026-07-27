# Simbiote

**An offline platform that takes a mobile-manipulator robot from _scan → simulate → train → operate_ — entirely air-gapped, on a single machine, in one sitting.**

> Built for the Dell × NVIDIA Hackathon — *"Local AI on Dell Pro Max with GB10,"* Seattle Tech Week.
> No cloud. No external APIs. Every model, every training run, and every demo runs on the box.

Point Simbiote at a robot spec and an environment — either a space you scanned
with a phone, or a ready-made scene from a library (a hospital, for this demo) —
and get back a robot that has **learned to navigate that space without
collisions and pick up objects in it**, plus two ways to drive and keep training
it: tracked **hand movements**, or plain-language **instructions**. The entire
loop is offline by design.

---

## The loop

```
                         ┌──────────────────────────────────────────────┐
   iPhone (LiDAR) ─────▶│ 1. SCAN      simbiote/mapper/                 │
   Stray Scanner         │    phone scan → OpenUSD scene + scene graph  │
                         └───────────────────────┬──────────────────────┘
                                                 ▼
                         ┌──────────────────────────────────────────────┐
                         │ 2. SIMULATE & TRAIN   simbiote/sim_env/,     │
                         │    training/, robot/                         │
                         │    spawn robot in scene → RL: navigation +   │
                         │    grasping → exported policy checkpoints    │
                         └───────────────┬───────────────┬──────────────┘
                                         ▼               ▼
              ┌────────────────────────────────┐  ┌────────────────────────────────┐
              │ 3.TELEOPERATE  simbiote/teleop/│  │ 4. COMMAND   simbiote/agentic/ │
              │   hand tracking → robot control│  │   "pick up the tray" → LLM →   │
              │                                │  │  behavior-tree skill execution │
              └───────────────┬────────────────┘  └────────────────┬───────────────┘
                              │      logged demonstrations          │
                              └──────────────┬──────────────────────┘
                                             ▼
                         ┌──────────────────────────────────────────────┐
                         │  RETRAIN   training/retrain.py               │
                         │  every teleop/agentic session becomes a demo │
                         │  that fine-tunes the policy (BC → PPO)       │
                         └──────────────────────────────────────────────┘
```

Every stage hands off through a small, explicit contract, so the four modules
were built in parallel and still snap together.

---

## The four modules

Simbiote is one Python package (`simbiote/`) made of four modules. Each was
owned by one teammate; here is what each one actually does.

### 1. Scan → Map — `simbiote/mapper/`

Turns an **iPhone LiDAR scan** (a [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662)
export: RGB video, LiDAR depth, confidence maps, IMU, and ARKit poses) into a
**simulation-ready OpenUSD scene** tagged with a navigable floor and graspable
objects, plus a machine-readable **scene graph** JSON.

- `ingest.py` — strict Stray Scanner parser (`load_capture_bundle`); matches
  depth/confidence frames by frame-id, not list order, and validates the bundle.
- `stages.py` — the five pipeline stages: pose refine (COLMAP) → depth
  completion (Depth Anything) → reconstruction (3DGUT Gaussian splatting) →
  semantic labeling (SAM 3) → scene-graph assembly with physics metadata.
- `usd.py` — writes the OpenUSD stage and runs `validate_map`, the coded
  handoff contract to Module 2 (meter scale, Y-up, collision API, a navigable
  floor, a graspable object with mass + grasp type).
- `cli.py` — the `simbiote-map` command (`doctor`, `ingest`, `run`, `validate`).

Runs in three modes: **`proxy`** (no models, contract testing), **`preview`**
(real but sparse LiDAR point cloud — works on a CPU laptop), and
**`production`** (the full GPU reconstruction on the GB10). See
[`docs/GB10_OPERATIONS.md`](docs/GB10_OPERATIONS.md).

### 2. Simulate & Train — `simbiote/sim_env/`, `simbiote/training/`, `simbiote/robot/`

Loads the scene, spawns the reference robot (a 4-wheel omnidirectional base + arm
+ gripper), and runs reinforcement learning.

- `sim_env/` — PyBullet [Gymnasium](https://gymnasium.farama.org/) environments:
  `NavEnv` (collision-free point-to-point navigation), `GraspEnv` (approach,
  grasp, lift a tagged object), and `WheelchairEnv` (stretch: constrained
  co-navigation). `grasp_attach.py` dynamically welds a grasped object to the
  gripper with a runtime fixed constraint — the mechanism the whole
  manipulation task depends on.
- `training/` — a shared actor-critic network (`policy_net.py`) trained two
  ways that feed the *same* weights: behavioral cloning from demonstrations
  (`bc_pretrain.py`) and from-scratch PPO (`ppo.py`, `train_nav.py`,
  `train_grasp.py`). `retrain.py` chains them (BC → PPO) into the combined
  learning loop; `export_policy.py` emits ONNX/TorchScript for Modules 3 & 4.
- `robot_iface/` — the cross-team schemas everything else builds on:
  `RobotAction`, `Pose`, `Trajectory`, and the deployable skill API
  (`navigate_to`, `pick_up`).

The physics backend is **PyBullet today**; the code is deliberately structured so
it swaps to **Isaac Sim / PhysX 5** on the GB10 behind unchanged interfaces.

### 3. Teleoperate — `simbiote/teleop/`

Drives the robot in real time from **hand tracking**, and logs every session as a
demonstration.

- `camera_source.py` → `hand_tracking.py` → `ik_bridge.py` → `RobotAction`.
- The iPhone becomes a webcam (via [Iriun](https://iriun.com/)); MediaPipe
  extracts 21 hand landmarks; an analytic IK bridge retargets them to base
  velocity + arm pose + gripper, with smoothing and sensible behavior when the
  hand leaves frame.
- `teleop_session.py` runs the loop and drives either a console sink or a
  PyBullet toy robot (`simbiote-teleop`).

GB10 swaps (same interfaces): WiLoR for MediaPipe, cuMotion/cuRobo for the IK.

### 4. Command (Agentic) — `simbiote/agentic/`

Lets a human **just ask** the robot to do something. A natural-language
instruction is parsed by an LLM into a validated sequence of skills, then
executed autonomously.

- `scene_query.py` — the robot's world model; resolves noun phrases
  ("the tray in the supply room") to real scene-graph ids so the model can't
  invent them.
- `command_parser.py` + `llm_backend.py` — **one** LLM call per instruction,
  with a single corrective retry. Backends: `FakeBackend` (deterministic
  rule-based planner, **no model needed**), an OpenAI-compatible client (Ollama
  on a laptop, Nemotron on the GB10), and a `FallbackBackend` that degrades
  gracefully if the model server is down.
- `task_executor.py` — a finite-state machine that runs the plan one skill at a
  time, checks preconditions, times out a hung skill, and compensates (auto-
  release) on failure — so a bad instruction can't leave the robot in a broken
  state.
- `agentic_session.py` — parse → execute → log (`simbiote-agentic`).

---

## What runs today (laptop) vs. on the GB10

Simbiote is **not finished** — it is a working laptop-tier build with every GB10
hardware/model swap staged behind a stable interface. Being honest about the
line:

| Capability | Runs now (laptop / CPU) | Needs the GB10 |
| --- | --- | --- |
| Scan ingest + validation | ✅ any machine | — |
| Scan → map | ✅ `proxy` + `preview` (real LiDAR point cloud) | Full reconstruction (COLMAP + Depth Anything + 3DGUT + SAM 3) |
| Physics + RL (nav & grasp) | ✅ PyBullet + PPO/BC¹ | Isaac Sim / PhysX 5, thousands of parallel envs |
| Teleoperation | ✅ MediaPipe + console/PyBullet sink | WiLoR, cuMotion, real robot |
| Agentic commands | ✅ `FakeBackend` (no model) or local Ollama | Nemotron on the box |
| Retrain from sessions | ✅ ingest + BC→PPO loop | scale |

¹ PyBullet has no Windows PyPI wheel, so physics-dependent tests **skip cleanly**
on native Windows (see [Known limitations](#known-limitations)). They run on
Linux/macOS and the GB10.

**Known incomplete pieces** (tracked, not hidden): COLMAP poses are computed but
not yet fused back into the pipeline; scene-graph *edges* (on-top-of / blocks-
path relations) are not yet produced; SAM 3 labeling currently uses the first
frame only; the wheelchair-transport task and vision-distillation are explicit
stretch goals; and the agentic `CheckpointBackend`'s wheelchair skills raise
`NotImplementedError` pending a persistent env handle.

---

## Quickstart (laptop)

```bash
# Python 3.11+
pip install -e ".[dev]"      # or: uv sync --extra dev
```

**Run the tests:**

```bash
python -m pytest
```

**Ask the robot to do something — no model required** (uses the rule-based
backend and a stubbed robot):

```bash
simbiote-agentic "pick up the tray in the supply room"
```

**Validate a phone scan** (drop a Stray Scanner export in
`UPLOAD_PHONE_SCANS_HERE/` first, or import from anywhere):

```bash
python scripts/import_stray_capture.py "C:\path\to\StrayScannerExport" --name my-scan
simbiote-map ingest "./UPLOAD_PHONE_SCANS_HERE/my-scan"
```

**Teleoperate with your webcam** (prints actions to the console):

```bash
simbiote-teleop --sink console
```

Add `--help` to any command for the full flag set. For the air-gapped
production run on the GB10, see [`docs/GB10_OPERATIONS.md`](docs/GB10_OPERATIONS.md).

---

## Repository layout

```text
simbiote/            # the platform (one installable package)
├── mapper/          #  1. scan → OpenUSD map
├── sim_env/         #  2. PyBullet physics + Gymnasium envs
├── training/        #  2. PPO + behavioral cloning
├── robot/           #  2. engine-agnostic robot config
├── robot_iface/     #     shared schemas: RobotAction, Pose, Trajectory, skills
├── teleop/          #  3. hand-tracking teleoperation
├── agentic/         #  4. natural-language command → skills
├── sim_stub/        #     toy PyBullet robot for teleop preview
└── demo_logger.py   #     shared session logger (feeds retraining)

scripts/             # capture import + GB10 production adapters
config/              # mapper config (example + generated GB10)
docs/                # master plan, GB10 ops, downloads, SSD layout
assets/              # stand-in URDFs for laptop testing
{Andrew,Gagan,Sky,Suraj}/   # per-owner tests + module notes
```

Console entry points (`pyproject.toml`): `simbiote-map`, `simbiote-teleop`,
`simbiote-agentic`.

---

## Known limitations

- **PyBullet on Windows** has no PyPI wheel; install requires MSVC Build Tools,
  or use WSL2 / Linux / macOS. The package degrades gracefully — non-physics
  code runs fine, and physics tests skip rather than crash.
- **Isaac Sim / PhysX 5, Nemotron, WiLoR, cuMotion** are GB10-day swaps; the
  laptop build uses PyBullet, MediaPipe, and local/rule-based LLM backends.
- Config defaults are Linux/GB10 paths; on a Windows dev box, pass a config file
  or environment overrides for anything beyond `proxy`/`preview`.

---

## Documentation

- [`docs/SIMBIOTE_MASTER_PLAN.md`](docs/SIMBIOTE_MASTER_PLAN.md) — the full
  platform plan, tech stack, and per-step design.
- [`docs/GB10_OPERATIONS.md`](docs/GB10_OPERATIONS.md) — production run on the
  GB10: adapters, setup, and the handoff contract.
- [`docs/GB10_DOWNLOADS.md`](docs/GB10_DOWNLOADS.md) — model/download checklist.
- [`docs/SSD_LAYOUT.md`](docs/SSD_LAYOUT.md) — external SSD layout.

---

## Team

| Module | Owner |
| --- | --- |
| Scan → Map (`mapper/`) | Gagan |
| Simulate & Train (`sim_env/`, `training/`, `robot/`) | Suraj |
| Teleoperation (`teleop/`) | Sky |
| Agentic Control (`agentic/`) | Andrew |
