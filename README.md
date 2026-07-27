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

| Owner | Responsibility | Primary code |
| --- | --- | --- |
| Gagan | Stray Scanner capture → OpenUSD scene map | `src/factoryflow_mapper/` |
| Suraj | Simulation, robot interfaces, and policy training | `simbiote/sim_env/`, `simbiote/training/` |
| Sky | Hand-tracking teleoperation | `simbiote/teleop/` |
| Andrew | Scene querying and agentic robot control | `simbiote/agentic/` |

Each owner folder (`Gagan/`, `Suraj/`, `Sky/`, `Andrew/`) contains that module's
README and tests. Run the complete suite with `python -m pytest`.

---

## The loop

```
                         ┌──────────────────────────────────────────────┐
   iPhone (LiDAR) ─────▶│ 1. SCAN      src/factoryflow_mapper/          │
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
                         │  RETRAIN   simbiote/training/retrain.py      │
                         │  every teleop/agentic session becomes a demo │
                         │  that fine-tunes the policy (BC → PPO)       │
                         └──────────────────────────────────────────────┘
```

Every stage hands off through a small, explicit contract, so the four modules
were built in parallel and still snap together.

---

## The four modules

### 1. Scan → Map — `src/factoryflow_mapper/`

Turns an **iPhone LiDAR scan** (a [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662)
export: RGB video, LiDAR depth, confidence maps, IMU, and ARKit poses) into a
**simulation-ready OpenUSD scene** tagged with a navigable floor and graspable
objects, plus a machine-readable **scene graph** JSON.

- `ingest.py` — strict Stray Scanner parser (`load_capture_bundle`); matches
  depth/confidence frames by the `frame` column, not list order, including
  non-contiguous frames, and validates the bundle.
- `stages.py` — the five pipeline stages: pose refine (COLMAP) → depth
  completion (Depth Anything) → reconstruction (3DGRUT Gaussian splatting) →
  semantic labeling (SAM 3) → scene-graph assembly with physics metadata.
- `usd.py` — writes the OpenUSD stage and runs `validate_map`, the coded
  handoff contract to Module 2 (meter scale, Y-up, collision API, a navigable
  floor, a graspable object with mass + grasp type).
- `cli.py` — the `factoryflow-map` command (`doctor`, `ingest`, `run`, `validate`).

Runs in three modes: **`proxy`** (no models, contract testing), **`preview`**
(real but sparse LiDAR point cloud — works on a CPU laptop), and
**`production`** (the full GPU reconstruction on the GB10).

Proxy output is visibly marked and fails production validation unless
`--allow-proxy` is explicitly supplied. It is for wiring tests, not the demo map.

### 2. Simulate & Train — `simbiote/sim_env/`, `simbiote/training/`, `simbiote/robot/`

Loads the scene, spawns the reference robot (a 4-wheel omnidirectional base + arm
+ gripper), and runs reinforcement learning.

- `sim_env/` — PyBullet [Gymnasium](https://gymnasium.farama.org/) environments:
  `NavEnv` (collision-free point-to-point navigation), `GraspEnv` (approach,
  grasp, lift a tagged object), and `WheelchairEnv` (stretch: constrained
  co-navigation). `grasp_attach.py` dynamically welds a grasped object to the
  gripper with a runtime fixed constraint — the mechanism the whole
  manipulation task depends on.
- `sim_env/isaac_nav.py`, `sim_env/hospital_map.py` — the Isaac tier: the trained
  nav policy driving `hospital.usd` at true scale, over an occupancy grid + A*.
  Measured **16/20 ordered location pairs at 0.96x path efficiency**, including a
  75 m traversal. See [`Suraj/README.md`](Suraj/README.md).
- `training/` — a shared actor-critic network (`policy_net.py`) trained two
  ways that feed the *same* weights: behavioral cloning from demonstrations
  (`bc_pretrain.py`) and from-scratch PPO (`ppo.py`, `train_nav.py`,
  `train_grasp.py`). `retrain.py` chains them (BC → PPO); `export_policy.py`
  emits ONNX/TorchScript for Modules 3 & 4.
- `robot_iface/` — the cross-team schemas everything else builds on:
  `RobotAction`, `Pose`, `Trajectory`, and the deployable skill API
  (`navigate_to`, `pick_up`).

The laptop physics backend is **PyBullet**; on the GB10 the same interfaces run
against **Isaac Sim / PhysX 5**.

### 3. Teleoperate — `simbiote/teleop/`

Drives the robot in real time from **hand tracking**, and logs every session as a
demonstration.

- `camera_source.py` → `hand_tracking.py` → `ik_bridge.py` → `RobotAction`.
- The phone becomes a webcam; a hand tracker extracts 21 landmarks; an analytic
  IK bridge retargets them to base velocity + arm pose + gripper, with smoothing
  and sensible behavior when the hand leaves frame.
- `teleop_session.py` runs the loop and drives either a console sink or a
  PyBullet toy robot (`simbiote-teleop`).
- On aarch64 the backend is **WiLoR** (`wilor_backend.py`), behind the same
  `get_hand_landmarks()` contract — MediaPipe ships no linux-aarch64 wheel. See
  [`docs/TELEOP_IPHONE_CAMERA.md`](docs/TELEOP_IPHONE_CAMERA.md) and
  [`docs/TELEOP_ISAAC_HOSPITAL.md`](docs/TELEOP_ISAAC_HOSPITAL.md).

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
- `robot_tools.py` — the skill-execution backends. `StubBackend` (no simulator),
  `CheckpointBackend` (Step 2's exports in a throwaway PyBullet arena), and
  `IsaacBackend` (`--robot isaac`), which owns a **persistent** `hospital.usd`
  simulator so the robot keeps its pose between skills and a multi-step
  instruction actually crosses the building.
- `agentic_session.py` — parse → execute → log (`simbiote-agentic`).

---

## What runs today (laptop) vs. on the GB10

Simbiote is **not finished** — it is a working laptop-tier build with every GB10
hardware/model swap staged behind a stable interface. Being honest about the
line:

| Capability | Runs now (laptop / CPU) | Needs the GB10 |
| --- | --- | --- |
| Scan ingest + validation | ✅ any machine | — |
| Scan → map | ✅ `proxy` + `preview` (real LiDAR point cloud) | Full reconstruction (COLMAP + Depth Anything + 3DGRUT + SAM 3) |
| Physics + RL (nav & grasp) | ✅ PyBullet + PPO/BC¹ | Isaac Sim / PhysX 5, thousands of parallel envs |
| Hospital navigation | — | ✅ Isaac Sim + `hospital.usd` |
| Teleoperation | ✅ MediaPipe + console/PyBullet sink | WiLoR, cuMotion, real robot |
| Agentic commands | ✅ `FakeBackend` (no model) or local Ollama | Nemotron on the box; `--robot isaac` |
| Retrain from sessions | ✅ ingest + BC→PPO loop | scale |

¹ PyBullet has no Windows PyPI wheel, so physics-dependent tests **skip cleanly**
on native Windows (see [Known limitations](#known-limitations)). They run on
Linux/macOS and the GB10.

**Known incomplete pieces** (tracked, not hidden): COLMAP poses are computed but
not yet fused back into the pipeline; scene-graph *edges* (on-top-of / blocks-
path relations) are not yet produced; the wheelchair-transport task and
vision-distillation are explicit stretch goals; and the wheelchair *manipulation*
skills (`align_gripper` / `attach_handle` / `detach`) still raise
`NotImplementedError` — they need the arm driven to a handle pose by IK and a
PhysX joint against a wheelchair prim, neither of which exists in the hospital
scene yet. The wheelchair *navigation* halves are wired in `IsaacBackend`.

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

**Teleoperate with your webcam** (prints actions to the console):

```bash
simbiote-teleop --sink console
```

Add `--help` to any command for the full flag set.

---

## Phone scan upload (start here)

Put Stray Scanner exports in **`UPLOAD_PHONE_SCANS_HERE/`** — the drop zone in
this repo. Each scan should be its own subfolder containing `camera_matrix.csv`,
`odometry.csv`, `imu.csv`, `rgb.mp4`, `depth/`, and `confidence/`.

```powershell
# Option A: drag the phone export folder into UPLOAD_PHONE_SCANS_HERE in File Explorer

# Option B: import from anywhere
python scripts\import_stray_capture.py "C:\path\to\StrayScannerExport" --name hospital-walkthrough

# Optional: also copy onto the SSD for GB10
python scripts\import_stray_capture.py "C:\path\to\StrayScannerExport" --name hospital-walkthrough --ssd
```

Then validate:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m factoryflow_mapper.cli ingest ".\UPLOAD_PHONE_SCANS_HERE\hospital-walkthrough"
```

### Metadata-only Stray Scanner export

Some exports — including pose/IMU-only downloads — contain only
`camera_matrix.csv`, `odometry.csv`, and `imu.csv`. They verify that pose data
reaches the mapper, but they cannot reconstruct or label a room: there are no RGB
frames or LiDAR depth. Validate one explicitly:

```bash
factoryflow-map --config config/mapper.gb10.toml \
  ingest /path/to/metadata-only-scan --allow-metadata-only
```

In `proxy` mode, `run` accepts this capture and produces a clearly marked test
USD. Production mode always requires `rgb.mp4`, `depth/`, and `confidence/`.

---

## Local CPU preview

A laptop can run `preview` mode even without NVIDIA CUDA. It fuses the Stray
Scanner's confidence-filtered LiDAR depth with its ARKit poses into a sparse
OpenUSD `Points` layer, then composes that into the mapper output. This is a real
capture-geometry preview; it is not a 3DGRUT reconstruction.

```powershell
$env:PYTHONPATH="$PWD\src"
$env:FACTORYFLOW_MODE="preview"
$env:FACTORYFLOW_WORK_ROOT="$PWD\.local\work"

python -m factoryflow_mapper.cli --config config/mapper.example.toml run `
  --capture ".\UPLOAD_PHONE_SCANS_HERE\<scan-name>" `
  --out ".\.local\preview.usda"
```

The output references `lidar_preview.usda` under its timestamped work folder.
Note that preview mode **still runs SAM 3** whenever `FACTORYFLOW_SAM3_COMMAND`
is set; without it you get a placeholder `proxy_graspable` cube instead of real
labels.

---

## GB10 operations

### SSD layout

The prepared SSD has one canonical `AI/` layout. Do not duplicate models or
captures inside the Git repository:

```text
<SSD>/AI/
├── captures/
├── models/
│   ├── depth-anything/
│   ├── nemotron/
│   ├── sam3/
│   ├── theia/
│   └── wilor/
├── repos/
│   ├── 3dgrut/
│   ├── colmap/
│   ├── Depth-Anything-3/
│   ├── sam3/
│   ├── IsaacSim/
│   └── IsaacLab/
└── assets/
    └── hospital/
        └── hospital.usd
```

`scripts/setup_gb10_mapper.sh` detects `<SSD>/AI` automatically and generates the
mapper config for this layout. Isaac Sim and Isaac Lab source are staged on the
SSD, but must be built on the ARM64 GB10.

### Setup

```bash
chmod +x scripts/setup_gb10_mapper.sh
./scripts/setup_gb10_mapper.sh /absolute/path/to/ssd
```

This creates the mapper work root, generates `config/mapper.gb10.toml`, and
installs the Python package with `uv`. Edit the generated config to match the
actual downloaded directories, then:

```bash
factoryflow-map --config config/mapper.gb10.toml doctor
```

`doctor` returns nonzero if a requirement for the selected mode is missing. The
architecture check is informational so development on x86 laptops still works.

### Production adapters

The downloaded model repositories are independently versioned and expose no
stable shared Python API, so the mapper uses **executable adapter contracts**
rather than importing unpinned repository internals. Set these after sourcing a
local `.env` or shell script:

```bash
export FACTORYFLOW_COLMAP_COMMAND="/opt/factoryflow/bin/run_colmap.sh {capture} {output}"
export FACTORYFLOW_DEPTH_COMMAND="/opt/factoryflow/bin/run_depth.sh {capture} {colmap} {checkpoint} {output}"
export FACTORYFLOW_DGRUT_COMMAND="/opt/factoryflow/bin/run_3dgrut.sh {capture} {colmap} {depth} {output}"
export FACTORYFLOW_SAM3_COMMAND="/opt/factoryflow/bin/run_sam3.sh {capture} {geometry} {checkpoint} {output}"
```

| Adapter | Placeholders | Must produce |
| --- | --- | --- |
| COLMAP | `{capture}`, `{output}` | Seed/refine from `odometry.csv`; write COLMAP artifacts under `{output}` |
| Depth Anything | `{capture}`, `{colmap}`, `{checkpoint}`, `{output}` | Confidence-filter LiDAR depth, complete/register it at RGB resolution; `{output}` must be nonempty |
| 3DGRUT | `{capture}`, `{colmap}`, `{depth}`, `{output}`, `{dgrut}` | Optimize the Gaussian scene, generate collision-capable mesh, emit at least one `.usd`/`.usda`/`.usdc` **directly** in `{output}` — a raw `.ply` or `.obj` is not a valid Step 2 handoff |
| SAM 3 | `{capture}`, `{geometry}`, `{checkpoint}`, `{output}` | `{output}/detections.json` (schema below) |

```json
{
  "nodes": [
    {
      "node_id": "floor_0",
      "label": "navigable floor",
      "kind": "floor",
      "confidence": 0.98,
      "bounds": { "center": [0.0, 0.0, 0.0], "size": [6.0, 0.05, 8.0] },
      "source_frame_ids": [0, 1, 2]
    },
    {
      "node_id": "tray_0",
      "label": "tray",
      "kind": "object",
      "confidence": 0.91,
      "bounds": { "center": [1.2, 0.8, -0.4], "size": [0.4, 0.08, 0.3] }
    }
  ]
}
```

Set `mode = "production"` only after all adapters pass `doctor`.

### Adapter runtime

The production adapters live in `scripts/gb10/` and use the native tool
interfaces rather than mock output:

1. `run_colmap.sh` extracts `rgb.mp4` and produces `images/` plus `sparse/0/`.
2. `run_depth.sh` runs DA3 and writes `exports/mini_npz/results.npz`.
3. `run_3dgrut.sh` trains the existing `colmap_3dgut` configuration and requires
   its exported USD/USDZ layer.
4. `run_sam3.sh` uses local `sam3.pt` from the official `facebook/sam3`
   repository to turn fixed text prompts into the `detections.json` contract.
   For **multi-frame** labeling use `Gagan/adapters/sam3_detect.py`;
   `scripts/gb10/sam3_labels.py` labels only the first video frame.

Run the setup script, then source its generated environment:

```bash
./scripts/setup_gb10_mapper.sh /mnt/factoryflow-ssd
source config/mapper.gb10.env
```

The repositories have separate CUDA Python environments. Before a production run,
point these at the executables created when installing them:

```bash
export DGRUT_PYTHON="$DGRUT_ROOT/.venv/bin/python"
export DA3_BIN="/path/to/depth-anything-3/.venv/bin/da3"
export SAM3_PYTHON="/path/to/sam3/.venv/bin/python"
```

The adapters require `ffmpeg`, a built `colmap` executable in `PATH`, and
CUDA-enabled PyTorch in the DA3, SAM 3, and 3DGRUT environments. Run each
repository's documented install process on the GB10 ARM64 system; copying a
Windows environment will not work.

Run the strict hardware and asset check before attempting a production map:

```bash
scripts/gb10/preflight_mapper.sh /mnt/factoryflow-ssd/AI "$PWD"
```

It blocks on missing ARM64/NVIDIA runtime, checkpoints, repositories, COLMAP,
3DGRUT's Python environment, or non-executable adapters. The hospital asset is
reported as a warning so a live scan can still proceed, but it remains required
for the reliable fallback demo.

### Production run and handoff

```bash
factoryflow-map --config config/mapper.gb10.toml \
  run --capture "$FACTORYFLOW_SSD_ROOT/captures/demo" \
  --out "$FACTORYFLOW_WORK_ROOT/demo.usda"

factoryflow-map validate "$FACTORYFLOW_WORK_ROOT/demo.usda"
```

Successful output includes:

- `demo.usda` — Step 2 scene.
- `demo.scene_graph.json` — planner/query contract.
- `<work-root>/mapper/<timestamp>/` — stage manifests and evidence.

The validator requires meter scale, Y-up coordinates, collision APIs, a navigable
floor, and at least one graspable object with mass and grasp type. Production
validation rejects proxy output.

---

## Repository layout

```text
src/factoryflow_mapper/   # 1. scan → OpenUSD map
simbiote/                 # the platform package
├── sim_env/              #  2. PyBullet envs + the Isaac hospital tier
├── training/             #  2. PPO + behavioral cloning
├── robot/                #  2. engine-agnostic robot config
├── robot_iface/          #     shared schemas: RobotAction, Pose, Trajectory, skills
├── teleop/               #  3. hand-tracking teleoperation
├── agentic/              #  4. natural-language command → skills
├── sim_stub/             #     toy PyBullet robot for teleop preview
└── demo_logger.py        #     shared session logger (feeds retraining)

scripts/                  # capture import
scripts/gb10/             # GB10 production adapters + Isaac teleop
checkpoints/              # trained policies (nav_bc.pt is the one to use)
config/                   # mapper config (example + generated GB10)
docs/                     # master plan, downloads, SSD layout, teleop notes
assets/                   # stand-in URDFs for laptop testing
UPLOAD_PHONE_SCANS_HERE/  # phone scan drop zone
{Andrew,Gagan,Sky,Suraj}/ # per-owner tests + module notes
```

Console entry points (`pyproject.toml`): `factoryflow-map`, `simbiote-teleop`,
`simbiote-agentic`.

---

## Known limitations

- **PyBullet on Windows** has no PyPI wheel; install requires MSVC Build Tools,
  or use WSL2 / Linux / macOS. The package degrades gracefully — non-physics
  code runs fine, and physics tests skip rather than crash.
- **On linux-aarch64** (the GB10), `pybullet`, `mediapipe`, and `usd-core` have
  no wheels. Teleop uses WiLoR, USD comes from Isaac Sim's bundled `pxr`, and a
  locally-built pybullet wheel is used. See [`Gagan/README.md`](Gagan/README.md).
- **Isaac Sim / PhysX 5, Nemotron, WiLoR, cuMotion** are GB10-day swaps; the
  laptop build uses PyBullet, MediaPipe, and local/rule-based LLM backends.
- Config defaults are Linux/GB10 paths; on a Windows dev box, pass a config file
  or environment overrides for anything beyond `proxy`/`preview`.

---

## Documentation

- [`docs/SIMBIOTE_MASTER_PLAN.md`](docs/SIMBIOTE_MASTER_PLAN.md) — the full
  platform plan, tech stack, and per-step design.
- [`docs/GB10_DOWNLOADS.md`](docs/GB10_DOWNLOADS.md) — model/download checklist.
- [`docs/GB10_MEMORY_BUDGET.md`](docs/GB10_MEMORY_BUDGET.md) — what fits on the box.
- [`docs/SSD_LAYOUT.md`](docs/SSD_LAYOUT.md) — external SSD layout.
- [`docs/TELEOP_IPHONE_CAMERA.md`](docs/TELEOP_IPHONE_CAMERA.md) — phone-as-webcam on aarch64.
- [`docs/TELEOP_ISAAC_HOSPITAL.md`](docs/TELEOP_ISAAC_HOSPITAL.md) — hand teleop into Isaac Sim.
- `Gagan/SCAN_MAP.md` — scan-to-map design notes.

---

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

---

## Team

| Module | Owner |
| --- | --- |
| Scan → Map (`src/factoryflow_mapper/`) | Gagan |
| Simulate & Train (`sim_env/`, `training/`, `robot/`) | Suraj |
| Teleoperation (`teleop/`) | Sky |
| Agentic Control (`agentic/`) | Andrew |
