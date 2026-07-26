# Simbiote — Teammate 2: Physics, Simulation & Training

**Shared context identical across all four files — jump to Part 5 for your work.**

---

## Part 0 — Project Overview

**Project:** Simbiote — Offline Platform for Scanning, Simulating & RL-Training Mobile Manipulator Robots
**Target:** Dell × NVIDIA Hackathon "Local AI on Dell Pro Max with GB10," Seattle Tech Week — **July 26, 2026, one-day sprint (9 AM – 9 PM)**
**Hardware:** Dell Pro Max with GB10 (provided day-of) + team laptops (dev machines) + an iPhone with LiDAR (capture + teleop camera)
**Team:** 4 people, one per role (see Parts 4–7): **Teammate 1** Scan & Map · **Teammate 2** Isaac Sim, Physics & Training · **Teammate 3** Hand-Tracking Teleoperation · **Teammate 4** Robot Prompting (Agentic Control)

### What Simbiote is

A platform other robotics companies use to train mobile-manipulator robots with reinforcement learning, entirely offline. A robotics company points Simbiote at their robot spec and either their own facility (scanned with a phone) or a ready-made environment from a library (a hospital, for this demo), and gets back a robot that's learned to navigate that space without collisions and pick up objects in it — plus a way to remotely operate and further train that same robot with tracked hand movements or plain-language instructions. The whole loop runs air-gapped, on one GB10, in a single sitting.

### The four roles

1. **Scan the environment to create a 3D map** (Teammate 1). Phone (Stray Scanner) → Omniverse-class reconstruction pipeline → a `.usd` scene, tagged with navigable floor and graspable objects.
2. **Load the map into Isaac Sim; the robot explores, learns to navigate, and gets its physics right** (Teammate 2). Two coupled RL fine-tunes — navigation and manipulation — plus the physics engine work (today: laptop-friendly PyBullet; tomorrow: PhysX 5 inside Isaac Sim).
3. **Remotely control the robot with tracked hand movements** (Teammate 3). Hand-tracking teleoperation, logging demonstrations for the fine-tune loop.
4. **Let a human just ask the robot to do things** (Teammate 4). Natural-language commands, decomposed into a sequence of high-level skills and executed autonomously via Teammate 2's trained policies.

### Event constraints (these gate everything)

| Constraint | Implication |
| :---- | :---- |
| One-day build, doors 9 AM, wrap 9 PM | ~10 usable build hours; everything heavy is prepared before the event, off-box |
| GB10 available only on the day | All ARM/aarch64 builds, containers, and wheels must be prepared blind and validated in the first hour |
| No cloud APIs; demo runs locally on the box | Matches the air-gap thesis exactly — scan, sim, train, and teleop all stay on-device |
| Weak venue Wi-Fi; bring models on USB/external drive | All weights, containers, and environment assets on a fast NVMe USB drive (500 GB+) |
| Required stack: OpenClaw + NemoClaw + OpenShell | Orchestration/security spine for the whole pipeline (Part 1) |
| Judging frame: "always-on business agent" | Pitch as a standing B2B platform: load a robot spec, pick or scan an environment, get back a trained policy |
| Teams of 2–4; top 8 pitch live that evening | Reliability > feature count — the flagship demo environment is pre-built (a hospital scene), not live-scanned |
| Laptop/phone are "for development"; demo must run on GB10 | GB10 runs everything judged; phone and laptop are framed strictly as input devices |

### Goals

- **G1** Scan a real facility via phone and reconstruct it as a sim-ready map (offline).
- **G2** Ship a pre-built environment library scene (hospital) as the reliable demo path.
- **G3** Support a 4-wheeled mobile base + single pick-and-place arm as the reference robot.
- **G4** Train navigation (collision-free movement) in sim.
- **G5** Train manipulation (approach + pick up tagged objects) in sim.
- **G6** Teleoperate and train via tracked hand movements, plus an "ask it to do things" agentic command mode.
- **G7** Enforce a verifiable air gap across every stage.
- **G8** Complete environment-load → trained policy → live teleop/agentic demo inside a stage-friendly time budget.
- **G9 (stretch)** GPU-accelerated PhysX 5 articulation dynamics for non-rigid environment interactions — swiveling wheelchair casters, a robot pushing a wheelchair via a dynamically-attached grasp constraint, accurate surface friction. Layered on top of the core pick-and-place task (G5), not a replacement for it — see Part 5.

### Confirmed tooling (verified this week, not hypothetical)

- **Capture:** Stray Scanner (open-source iOS app; requires a LiDAR-equipped iPhone 12 Pro/Pro Max or later Pro-line, or iPad Pro). Confirmed export format: `camera_matrix.csv`, `odometry.csv`, `imu.csv`, `depth/*.png` (192×256), `confidence/*.png` (0/1/2), `rgb.mp4`.
- **Teleop camera:** Iriun Webcam — the same iPhone, connected via USB (preferred, ~1–5 ms latency) or Wi-Fi, appears as a standard virtual webcam on the laptop/GB10. Capture and teleop are a role-swap on one device, not two separate devices.

---

## Part 1 — System Architecture

### Topology

```
┌───────────────────────────────┐      Cat6/Wi-Fi, static IPs   ┌──────────────────────────────────────┐
│ INPUT DEVICES                  │  192.168.1.50 ⇄ .10:8555      │ DEVICE — Dell Pro Max GB10             │
│ • iPhone — Stray Scanner       │ ─────────────────────────────▶│ "The Brain" (hackathon submission)     │
│   (Step 1) then Iriun Webcam   │   TCP, length-prefixed        │                                        │
│   (Step 3) — same device,      │   scan data / hand-pose /     │ • OpenClaw (orchestrator)              │
│   role swap between stages     │   camera stream               │ • NemoClaw (guardrails/firewall)      │
│ • Laptop — dev machine, runs   │ ◀───────────────────────────  │ • OpenShell (Landlock sandbox)         │
│   the Iriun desktop client     │   robot pose / policy         │ • Reconstruction pipeline (Step 1)     │
│                                 │   feedback for display        │ • Environment library (hospital scene) │
└───────────────────────────────┘                                │ • Isaac Sim 5.x + Isaac Lab (Step 2)   │
                                                                  │ • cuMotion / cuRobo (Step 2 & 3)       │
                                                                  │ • Isaac Sim (rendered, WebRTC out)     │
                                                                  └──────────────────────────────────────┘
```

All judged compute lives on the GB10 — the phone and laptop are sensor/input devices only.

### OpenClaw as pipeline orchestrator

| Stage | OpenClaw tool | Trigger | Owner (Part) |
| :---- | :---- | :---- | :---- |
| Load environment | `load_environment(source)` | phone scan → reconstruction, or a library id (`hospital_01`) | Part 4 (Step 1) |
| Validate map | `validate_map(usd)` | NemoClaw-gated: watertight floor, correct scale, pickable objects tagged | Part 4 (Step 1) |
| Spawn robot | `spawn_robot(spec)` | Loads the 4-wheel + arm reference robot into the scene | Part 5 (Step 2) |
| Train | `train_nav()` / `train_grasp()` | Warm-started fine-tunes | Part 5 (Step 2) |
| Teleop session | `start_teleop(robot_id)` | Operator-initiated | Part 6 (Step 3) |
| Agentic command | `parse_instruction(text)` → task hierarchy → tool calls | Natural-language instruction | Part 6b (Step 4) |
| Log & retrain | `ingest_demo()` → `finetune_policy()` | Runs after a teleop/agentic session | Parts 5 & 6 |

**Multi-agent structure:** **Mapper** (Step 1), **Trainer** (Step 2), **Teleop Bridge** (Step 3), **Task Planner** (Step 4 — owns the LLM instruction parsing and Behavior Tree/FSM task hierarchy), **Deployer** (the only agent permitted to push a policy update or hand control to a human — spans Steps 2, 3 & 4).

### Live pipeline (per demo run)

| # | Stage | Owner | Budget |
| :---- | :---- | :---- | :---- |
| 1 | Load environment (library or phone scan) | Step 1 | seconds (library) / minutes (scan) |
| 2 | Validate & spawn robot | Step 1 → Step 2 | seconds |
| 3 | Train navigation | Step 2 | minutes, parallel envs |
| 4 | Train manipulation | Step 2 | minutes, parallel envs |
| 5 | Teleoperate (Step 3) or issue an agentic command (Step 4) | Steps 3 & 4 | real-time |
| 6 | Retrain from the session | Step 2 (consumes logged trajectories from Steps 3 & 4) | seconds–minutes |
| 7 | Deploy & display | All | seconds |

### Memory budget (128 GB unified) — hard ceiling ~105 GB, ≥20 GB headroom enforced by a watchdog

| Resident component | Est. footprint | Owner |
| :---- | :---- | :---- |
| Reconstruction pipeline (scan path only, transient) | ~15–25 GB | Step 1 |
| Environment library scene (hospital, USD + assets) | ~5–8 GB | Step 1 |
| Isaac Sim (headless training) + Isaac Lab + PhysX | ~20–30 GB | Step 2 |
| Isaac Sim (rendered, WebRTC out) | ~10–15 GB | Step 2 / demo |
| cuMotion / cuRobo | ~2–3 GB | Step 2 & 3 |
| Hand-pose model (teleop only) | <1 GB | Step 3 |
| **Nemotron 3 Super (120B-A12B, NVFP4, ~60 GB)** — shared reasoning LLM | **~60 GB** | Steps 1, 2, 4 |
| OpenClaw / NemoClaw / OpenShell / buffers | ~8 GB | Shared |

**Rule, extended:** reconstruction and training were already sequential by design; **Nemotron Super and a fully-loaded Isaac Sim (headless + rendered) now join that same sequencing rule** — at ~60 GB, Super alone is roughly half the machine. Don't plan on all three (reconstruction, full Isaac Sim, Super) resident simultaneously; the realistic pattern is: reconstruct → unload reconstruction, load Super for scene-graph generation → unload or reduce Super, load Isaac Sim for training → keep Isaac Sim (rendered instance only) + swap Super back in for the live agentic/demo phase. This is a real architecture decision, not a minor footnote — confirm it works with a full dry run tomorrow morning before trusting it on stage.

**Simpler fallback:** if that sequencing feels like too much risk to add the day of, **Nemotron 3 Nano (30B-A3B, ~25 GB)** is the "just works, stays resident the whole time" choice — noticeably lower reasoning quality than Super, but it coexists with a fully-loaded Isaac Sim without any of the above juggling. Decide which trade-off you want before 9 AM, not during the rehearsal window.

**Nemotron 3 Ultra (550B-A55B) is not on the table at all** — even at its native NVFP4 checkpoint it's ~275 GB, nearly 2.5x GB10's entire 128 GB of unified memory. This isn't a quality/risk trade-off like Super vs. Nano; it's a hard hardware ceiling. Don't attempt it.

---

## Part 2 — The Whole Tech Stack

| Layer | Technology | Owner | Notes |
| :---- | :---- | :---- | :---- |
| OS | DGX OS (Ubuntu-based, aarch64) | Shared | Ships on GB10; CUDA 13.x, ARM builds required for every dependency |
| Containerization | Docker + NVIDIA Container Toolkit | Shared | One container per service; `--network=host` inside OpenShell scope |
| Capture app | **Stray Scanner** (iOS, open source) | Step 1 | Requires a LiDAR-equipped iPhone/iPad; exports camera_matrix.csv, odometry.csv, imu.csv, depth/, confidence/, rgb.mp4 |
| Feature matching / pose refinement | COLMAP (CPU laptop build → CUDA build on GB10) | Step 1 | Seeded with the phone's ARKit poses, not solved cold |
| Depth completion | Depth Anything V2 (Small → Large) | Step 1 | Guided upsampling of Stray Scanner's real 192×256 LiDAR depth to full RGB resolution |
| Gaussian reconstruction | 3DGUT (`nv-tlabs/3dgrut`) | Step 1 | Handles rolling shutter / lens distortion natively; regularized against real, confidence-weighted depth |
| Semantic labeling | Grounding DINO + SAM 2 (tiny/small → large) | Step 1 | Open-vocabulary segmentation, projected 2D→3D |
| Scene-graph reasoning | Phi-4-mini / Gemma 3 4B (laptop) → **Nemotron 3 Super** (NVFP4, ~60 GB) or **Nano** (~25 GB) (GB10) | Step 1 | Same Nemotron instance doubles as OpenClaw's orchestration brain; Super needs sequencing around Isaac Sim |
| Environment library | Pre-authored SimReady-style USD scene (hospital) | Step 1 | Ships on the USB drive; the reliable default demo path |
| Simulation & training | Isaac Sim 5.x + Isaac Lab | Step 2 | Headless (training) + rendered (WebRTC) instances |
| Physics engine (today) | **PyBullet** (laptop, no NVIDIA GPU required) | Step 2 | Runtime `createConstraint`/`removeConstraint` for grasp-attach logic; URDF articulation for wheels/casters |
| Physics engine (tomorrow) | **NVIDIA PhysX 5** (native inside Isaac Sim 5.x) | Step 2 | GPU Featherstone articulation solver, SDF collision meshes (`UsdPhysics.MeshCollisionAPI`), 120 Hz sub-stepping |
| Asset physics rigging | `UsdPhysics` + V-HACD/SDF collision, `UsdPhysics.PhysicsMaterial` friction | Step 1 & 2 | Joint friction/damping, tire/gripper friction materials, handle grasp constraints |
| RL library | RSL-RL / SKRL (laptop, 16–64 envs) → RL-Games / SKRL (GB10, 2,000–4,096+ envs) | Step 2 | All are Gymnasium-compatible; swapping scale doesn't change the task definition |
| Vision backbone (optional student policy) | ResNet-18 (laptop) → Theia / ResNet-50 (GB10) | Step 2 | Teacher (privileged state) → student (vision) distillation |
| Robot platform | Clearpath Ridgeback (4 mecanum wheels) + Franka Panda arm — built-in Isaac Sim asset | Step 2 | `RidgebackFranka/ridgeback_franka.usd`; no custom robot authoring needed |
| High-level task planner | Behavior Tree / Finite State Machine | Step 4 | Decomposes an LLM-parsed instruction into atomic skills (approach → align → attach constraint → navigate → detach) |
| Motion planning / IK | cuMotion / cuRobo (CUDA) | Step 2 & 3 | Shared by nav re-planning, grasp approach, and hand-tracking retargeting |
| Teleop camera | **Iriun Webcam** (iPhone → virtual webcam, USB preferred) | Step 3 | Same iPhone as Step 1's Stray Scanner, role-swapped |
| Hand tracking | MediaPipe Hands (laptop) → WiLoR (GB10) | Step 3 | Same interface both ways — WiLoR does its own hand detection end-to-end |
| Agentic command parsing | Qwen3 8B / Phi-4-mini (laptop) → **Nemotron 3 Super/Nano** (GB10) | Step 4 | Same shared reasoning model as Step 1's scene-graph generation |
| Orchestration | OpenClaw gateway + NemoClaw wrapper + OpenShell sandbox | Shared | Mandatory event stack; conducts the whole pipeline |
| Transport | Python asyncio TCP, length-prefixed frames, port 8555 | Shared | Phone/laptop ⇄ GB10 |
| Glue language | Python 3.11 everywhere | Shared | Single `uv`-managed monorepo |
| Dashboard | OpenClaw log web-tail + Isaac Sim WebRTC client | Shared | Wired display output only |

**ARM caveat (do this Day 1, first thing):** GB10 is aarch64. Verify ARM wheels/containers exist for the reconstruction pipeline, Isaac Sim's ARM image, and every inference runtime (hand-pose model, LLMs). Anything that won't build on ARM gets discovered on Day 1, not demo day.

---

## Part 3 — Security Architecture

### OpenShell / Landlock (`openshell_policy.yaml`)

- `allow_external: false`, `allow_loopback: true`; bind the LAN interface only; whitelist the phone and laptop's IPs, port `8555`.
- Socket/port-level Landlock scoping requires Landlock ABI v4 (kernel ≥ 6.7). Check DGX OS kernel on day 0; nftables fallback if needed.
- Filesystem: read-only reconstruction/training weights + environment-library directory; write access only to `/var/simbiote/stage/` (scanned maps, trained policies, teleop logs).

### NemoClaw guardrails

- **Ingress sanitization:** strips identifiers from phone-scan video and hand-tracking streams before they reach any model or log.
- **Role-based policies (per-agent):** Mapper and Trainer are read-only outside their own stage; only Planner/Deployer may call `start_teleop()` or push a policy update. No agent executes a generated script that hasn't passed static scan.
- **Egress-to-execution:** static analysis of every auto-generated USD injection script and PhysX/training config before Isaac Sim executes it.
- **Audit trail as pitch closer:** every environment load, training run, teleop/agentic session, and policy update logged in structured form.

---

## Part 5 — Step 2: Physics, Simulation & Training
**(Teammate 2's step)**

### 5.1 What this step does

Takes Step 1's `.usd` scene (or the built-in `hospital.usd`), spawns the reference robot (Clearpath Ridgeback + Franka arm) in it, and runs coupled, warm-started RL fine-tunes: navigation, manipulation, and — as a stretch goal — wheelchair transport. Produces exported policy checkpoints that Step 3 (teleop) and Step 4 (agentic) both call directly.

### 5.2 Physics engine — today (laptop, no Isaac Sim) vs tomorrow (GB10, PhysX 5)

Isaac Sim isn't usable right now, so tonight's work happens in a different engine that's designed to port forward, not a from-scratch prototype you'll throw away.

**Today: PyBullet.** Chosen specifically over MuJoCo for one reason: the hardest physics problem in this project — a robot arm grasping something and forming a rigid attachment (a wheelchair handle, or just an object) — has a purpose-built, well-documented PyBullet mechanism: `p.createConstraint(..., jointType=p.JOINT_FIXED, ...)` dynamically welds two bodies together at runtime, and `p.removeConstraint()` releases it. That's a direct match for "dynamically instantiate a fixed joint on grasp." PyBullet also loads URDFs with proper revolute joints (for wheels and swiveling casters) and per-link friction (`changeDynamics(lateralFriction=...)`), and none of it needs an NVIDIA GPU.

**Tomorrow: NVIDIA PhysX 5**, native inside Isaac Sim 5.x. The concepts map over directly, not just conceptually:

| Concept | PyBullet (today) | PhysX 5 / Isaac Sim (tomorrow) |
| :---- | :---- | :---- |
| Grasp-attach (closed kinematic chain) | `p.createConstraint(..., p.JOINT_FIXED, ...)` at grasp, `p.removeConstraint()` at release | Runtime attach via `omni.physx`'s scene interface, or a `UsdPhysics.FixedJoint` authored at grasp time |
| Articulated joints (wheels, casters) | URDF revolute joints | `UsdPhysics.RevoluteJoint`, with real joint damping/drive stiffness |
| Surface friction | `changeDynamics(lateralFriction=...)` — single Coulomb coefficient | `UsdPhysics.PhysicsMaterial` — separate static/dynamic friction, restitution |
| Collision geometry | Convex hulls / primitive shapes | SDF (`UsdPhysics.MeshCollisionAPI`) for thin/complex geometry |
| Solver | Bullet's default solver | Featherstone articulation solver, GPU-accelerated, configurable position/velocity iteration counts |

**What this means practically:** validate reward shaping, task logic, and the grasp-attach/release trigger conditions in PyBullet tonight. Tomorrow you're not redesigning any of that — you're re-pointing the same logic at Isaac Lab's task API and getting a real solver, real friction materials, and GPU parallelism underneath it for free.

**PhysX 5 config for tomorrow** (drop into `sim_env/` once you're on the GB10):

```python
physics_config = {
    "physics_engine": "PhysX5",
    "substeps": 2,              # 120 Hz at 60 FPS sim render
    "solver_type": 1,           # TGS (Temporal Gauss-Seidel) for articulation stability
    "enable_sdf_collisions": True,
    "default_friction_material": {
        "static_friction": 0.8,
        "dynamic_friction": 0.65,
        "restitution": 0.0,
    },
}
```

### 5.3 Models — laptop (today) vs GB10 (tomorrow)

| | Laptop (today) | GB10 (tomorrow) |
| :---- | :---- | :---- |
| Physics engine | PyBullet | PhysX 5 (native, Isaac Sim 5.x) |
| RL library | RSL-RL or SKRL (or a plain PyBullet + Stable-Baselines3 loop) | RL-Games or SKRL |
| Parallel environments | 1 (PyBullet is single-instance by default) to a small handful via multiprocessing | 2,000–4,096+ (Isaac Lab's own H1-humanoid example runs 4096 in parallel on one GPU) |
| Navigation approach | Fine-tune a small pretrained nav checkpoint on a simplified stand-in scene | Same checkpoint, fine-tuned against the real `hospital.usd`, full env count |
| Perception backbone | Privileged state only (skip vision), or a small ResNet-18 to test the path | Teacher–student distillation: privileged-state teacher → vision student on **Theia** or **ResNet-50** |
| Manipulation approach | Fine-tune a pretrained grasp prior on 1–2 simple shapes | Fine-tune against the actual tagged pickable objects in the hospital scene |
| Orchestration reasoning model | Qwen3 8B or Phi-4-mini | **Nemotron 3 Super** (NVFP4, ~60 GB; needs sequencing around Isaac Sim, see Part 1) — or **Nemotron 3 Nano** (~25 GB) if you want it always resident — shared with Steps 1, 3, and 4 |

### 5.4 Build spec

**Environment & robot — confirmed, nothing to author from scratch, once you're on Isaac Sim:**
- **Environment:** `hospital.usd`, a built-in sample scene (multiple rooms, collision geometry included) — `Create > Environments > Hospital`. This is your Step 1 "environment library" scene.
- **Robot:** `RidgebackFranka/ridgeback_franka.usd` — a pre-assembled Clearpath Ridgeback (4 mecanum-wheel omnidirectional base) + Franka Emika Panda arm (7-DOF, 2-finger gripper) — `Create > Robots > Wheeled Robots > Clearpath > Ridgeback Franka`.
- **For tonight (PyBullet):** you'll need a stand-in — `pybullet_data` ships common robot URDFs but not this exact Ridgeback+Franka combo, so build or find a simple 4-wheel-base + arm URDF stand-in for logic testing. Don't chase visual or dimensional accuracy here; you're validating reward logic and constraint behavior, not producing a demo asset.
- **Optional stretch:** NVIDIA's Isaac for Healthcare (i4h) asset catalog (`github.com/isaac-for-healthcare/i4h-asset-catalog`) for extra hospital props (trays, carts, equipment).

```
simbiote/robot/
└── robot_config.py   — joint names, action limits, default spawn pose,
                         sensor attachment points; same config shape
                         whether it's pointing at a PyBullet URDF today
                         or ridgeback_franka.usd tomorrow

simbiote/sim_env/
├── nav_task.py       — obs = privileged state (pose, nearby obstacles, goal);
│                        action = base velocity; reward = progress-to-goal,
│                        collision penalty, smoothness
├── grasp_task.py     — obs = privileged state (object pose, EE pose);
│                        action = EE target pose + gripper; reward =
│                        approach progress, grasp success, drop penalty
├── grasp_attach.py   — the shared grasp-constraint logic from §5.2:
│                        def attach(robot, ee_link, target_body) -> constraint_id
│                        def release(constraint_id) -> None
│                        (PyBullet impl today; swap internals for the
│                        Isaac Sim/PhysX equivalent tomorrow, same signature)
└── register_envs.py  — registers tasks as Gymnasium envs

simbiote/training/
├── bc_pretrain.py       — def train_bc(trajectories: list[Trajectory], policy_net) -> checkpoint
│                          NEW: Behavioral Cloning on teleop demos from Step 3's
│                          demo_logger.export_trajectory() — supervised
│                          (observation → action) regression, same MLP
│                          actor-critic architecture as train_nav.py so the
│                          weights load straight into it as a warm start
├── train_nav.py         — CLI: --num_envs, --task, --checkpoint, --rl_lib
│                          --checkpoint now points at bc_pretrain.py's output
│                          when demos exist — PPO then fine-tunes that BC
│                          policy via autonomous exploration in nav_task.py
│                          (the "just moving around" learning source);
│                          without demos yet, falls back to a random/small
│                          PPO warm-up exactly as before
├── train_grasp.py       — same pattern as train_nav.py
├── play.py              — loads a checkpoint for visual inspection
├── export_policy.py     — exports to ONNX/TorchScript for Steps 3 & 4
└── distill_student.py   — (GB10-only stretch) teacher → vision-student distillation
```

**The combined learning loop, concretely:** the nav policy learns from two sources, not one — Step 3's teleop sessions (imitation, via `bc_pretrain.py`) and Isaac Lab's own autonomous rollouts (reinforcement, via `train_nav.py`'s PPO loop against `nav_task.py`'s reward). Order: **BC pretrain on whatever teleop demos exist → PPO fine-tune in sim → new teleop session arrives → re-run `bc_pretrain.py` on the combined demo set, seeded from the current PPO checkpoint → PPO fine-tune again.** Same policy network (a standard MLP actor-critic — the RSL-RL/RL-Games/SKRL default for this observation/action size, nothing exotic needed for a one-day build) receives updates from both sources rather than keeping them as two separate models you'd have to merge later. Model architecture and this whole loop are identical on PyBullet tonight and Isaac Lab tomorrow — only the environment underneath changes.

**Build order:** stand-in robot URDF → `grasp_attach.py` (get the constraint create/release logic right first — this is the piece everything else, including the stretch wheelchair task, depends on) → `nav_task.py` → `grasp_task.py` → `register_envs.py` → `bc_pretrain.py` (test it against a handful of fake/toy demo trajectories before real teleop data exists) → `train_nav.py` (laptop-scale, confirm PPO improves on top of the BC checkpoint, not just from scratch) → `train_grasp.py` → `play.py` → `export_policy.py`. Tomorrow: swap the PyBullet stand-in for `ridgeback_franka.usd` + `hospital.usd`, re-implement `grasp_attach.py`'s internals against `omni.physx`, everything else stays.

**Acceptance tests:** `train_nav.py` reaches the goal without collision noticeably more often than (a) the un-fine-tuned baseline and (b) BC-only with no PPO fine-tune — confirming both learning sources are actually contributing, not just one; `train_grasp.py` clears the same bar on grasp success rate; `grasp_attach.py`'s constraint holds under simulated motion; an exported checkpoint loads and runs outside the training loop.

### 5.5 Stretch task — wheelchair transport (Tier 2/3, not core scope)

Layered on top of the core pick-and-place task, not a replacement for it. This is a genuinely harder problem — a closed kinematic chain (base → arm → wheelchair handle → wheelchair frame → floor) with a co-navigation reward that has to penalize tipping, not just collisions. Treat it as something to attempt only after Tier 1 (plain pick-and-place) is solid and rehearsed.

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  1. Approach    │ ──▶ │  2. Align & Grasp   │ ──▶ │ 3. Constrained Nav   │ ──▶ │ 4. Release & Park    │
│  Nav Policy     │     │  Grasp Policy       │     │  Co-Nav Policy       │     │  Detaches Joint      │
│  (Base Only)    │     │  (Arm Alignment)    │     │  (Base + Wheelchair) │     │  (Task Complete)     │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

- **Wheelchair rigging:** an articulated body — main rear wheels as driven revolute joints, front casters as revolute joints with a physical offset so they trail naturally rather than sliding "like a block of ice" (the offset is what makes a caster self-align to the direction of travel; without it, expect exactly that ice-sliding failure mode). Build a minimal placeholder URDF for tonight's PyBullet testing — 2 driven wheels + 2 offset-caster front wheels is enough to validate the reward logic; visual fidelity doesn't matter yet.
- **Grasp/attach:** uses `grasp_attach.py` from §5.4 — when the end-effector reaches the handle, call `attach()`; on task completion, call `release()`.
- **Co-navigation reward:** fine-tune the navigation policy specifically with the wheelchair attached — reward progress toward the destination, penalize excessive tilt angle and sharp turns that would tip it.
- **Object-pick task tie-in (Task B, core scope):** semantic tagging from Step 1 already carries `is_graspable`, `grasp_type`, `mass_kg` attributes per object — `grasp_task.py`'s reward and approach logic should read these rather than hardcoding assumptions, since it's the same interface the wheelchair task's handle-grasp uses.

### 5.6 Tonight's realistic scope

Get `grasp_attach.py` right first in PyBullet — everything else, including the stretch wheelchair task, depends on that constraint logic behaving correctly. Then `nav_task.py`/`grasp_task.py` against a stand-in robot and scene, trained and smoke-tested at laptop scale. The wheelchair task is explicitly optional for tonight — only start it if the core loop is solid with time to spare. Tomorrow: confirm `hospital.usd`/`ridgeback_franka.usd` load, re-implement `grasp_attach.py` against PhysX, switch task configs to the real scene, raise `--num_envs` into the thousands.

### 5.7 Handoff

**You receive from Step 1:** a validated `.usd` scene with navigable-floor and grasp-affordance tags (including `is_graspable`/`grasp_type`/`mass_kg` per object). **You produce for Step 3 (teleop) and Step 4 (agentic):** exported ONNX/TorchScript checkpoints, callable via `navigate_to(location_id)` / `pick_up(object_id)` — and, if the wheelchair stretch task is attempted, `approach_wheelchair()` / `attach_handle()` / `nav_with_payload()` / `detach()`. **You receive back from Steps 3 & 4:** logged trajectories via `ingest_demo()` — coordinate the trajectory schema as a group, since it's the one interface all four of you touch.

### 5.8 Implementation status (post-build) — what's actually in `simbiote/` tonight

Everything in §5.4's build order is coded and passing 67 automated tests (`pytest -q`, including two slow PyBullet-in-the-loop PPO convergence checks). This is the concrete state Teammates 3 & 4 should build against, and what carries forward to the GB10 tomorrow.

```
simbiote/
├── robot_iface/
│   ├── actions.py     — RobotAction / Pose / GripperState, exactly the §6.2/§6b.2 shape,
│   │                    + to_vector()/from_vector() (11-dim: vx,vy,omega,ee_xyz,ee_quat,gripper)
│   ├── trajectory.py  — TrajectoryStep/Trajectory (the resolved schema below) + JSON save/load
│   └── skills.py       — navigate_to()/pick_up()/approach_wheelchair()/attach_handle()/
│                          nav_with_payload()/detach() — the §5.7 skill API, concrete now
├── robot/robot_config.py — RobotConfig (STAND_IN_CONFIG for PyBullet today,
│                            RIDGEBACK_FRANKA_CONFIG stubbed for tomorrow)
├── sim_env/
│   ├── pybullet_scene.py — connect/load_robot/RobotHandles/spawn_graspable_box/spawn_table/spawn_wall
│   ├── grasp_attach.py   — attach()/release(), exact §5.4 signature
│   ├── nav_task.py, grasp_task.py, wheelchair_task.py — Gymnasium envs (NavEnv/GraspEnv/WheelchairEnv)
│   ├── register_envs.py  — gym.register() for all three, idempotent
│   └── spawn.py           — spawn_robot(spec) OpenClaw tool (§1's orchestration table)
├── training/
│   ├── policy_net.py   — ActorCriticMLP (shared BC + PPO net), save()/load()
│   ├── ppo.py            — train_ppo() + PPOConfig + GAE, validated independently against Pendulum-v1
│   ├── bc_pretrain.py    — train_bc(trajectories, policy_net, ...) -> checkpoint
│   ├── train_nav.py / train_grasp.py — CLIs, --checkpoint warm-starts PPO from a BC checkpoint
│   ├── play.py / export_policy.py — checkpoint inspection + ONNX/TorchScript export
│   ├── retrain.py         — ingest_demo()/finetune_policy() OpenClaw tool pair (below)
│   └── distill_student.py — vision-student stub, GB10-only stretch, not wired up tonight
├── demo_logger.py       — shared log_action()/export_trajectory() session logger for Steps 3 & 4
└── assets/{robots,wheelchair}/*.urdf — stand-in URDFs for tonight's PyBullet testing
```

**Trajectory schema — resolved (§5.7/§6.7/§6b.7's "coordinate as a group" item):**

```python
TrajectoryStep(timestamp: float, observation: list[float], action: RobotAction, reward: float = 0.0)
Trajectory(session_id: str, source: "teleop"|"agentic", task: "nav"|"grasp"|"wheelchair",
           steps: list[TrajectoryStep])
# .save(path) / Trajectory.load(path) round-trip to JSON; RobotAction nests inside each step.
```

This is Teammate 2's proposed concrete version (in `robot_iface/trajectory.py`) so `bc_pretrain.py` has something real to train against tonight — Teammates 3 & 4's `demo_logger.py` should target this shape; flag it in the group sync if either of you needs a different field.

**`ingest_demo()` / `finetune_policy()` signatures (Part 1's orchestrator table, homed in `training/retrain.py`):**

```python
def ingest_demo(trajectory: Trajectory) -> Path
# Persists one logged demo (from Step 3/4) to var/simbiote/stage/demos/<task>/<session_id>.json

def finetune_policy(
    task: str,                                  # "nav" | "grasp"
    current_checkpoint: str | Path | None = None,
    bc_epochs: int = 30, ppo_timesteps: int = 4000, num_envs: int = 2,
    out_path: str | Path | None = None, seed: int = 0,
) -> Path
# Runs BC-on-ingested-demos (seeded from current_checkpoint) -> PPO fine-tune,
# per §5.4's combined-learning-loop paragraph. Raises RuntimeError if no
# demos have been ingested for `task` yet.
```

**`spawn_robot(spec)` signature (Part 1's orchestrator table, homed in `sim_env/spawn.py`):**

```python
def spawn_robot(spec: str | RobotConfig = "stand_in", gui: bool = False, fixed_base: bool = False) -> SpawnedRobot
# spec is a registry name ("stand_in" today / "ridgeback_franka" — raises
# NotImplementedError until tomorrow's Isaac Sim swap) or a RobotConfig directly.
```

**Skill API signatures (`robot_iface/skills.py`, called directly by Step 4's `robot_tools.py`):**

```python
navigate_to(location_id: str, checkpoint_path=..., gui=False, max_steps=300) -> dict
pick_up(object_id: str, checkpoint_path=..., gui=False, max_steps=200) -> dict
approach_wheelchair(location_id: str, ...) -> dict
attach_handle(robot: SpawnedRobot, wheelchair_body_id: int) -> GraspConstraint
nav_with_payload(location_id: str, ...) -> dict
detach(constraint: GraspConstraint) -> None
```

Location/object name → world position is a placeholder registry (`DEFAULT_LOCATIONS`/`DEFAULT_OBJECTS`) in `skills.py` today; swap it for real calls into Step 1's `scene_query.py` once that scene graph exists — the function signatures don't change.

**Deviations/clarifications discovered while building, worth knowing before the GB10 swap:**
- The 3-DOF stand-in arm can't track a full 6-DOF end-effector pose (over-constrained) — `grasp_task.py`'s IK is **position-only** tonight (`p.calculateInverseKinematics` without `targetOrientation`). The real 7-DOF Franka arm tomorrow has the DOF to track orientation too; re-enable it then if the grasp task wants it.
- `p.calculateInverseKinematics`'s result array is indexed by **movable joints only**, not raw URDF joint indices — `RobotHandles` now exposes `dof_joint_indices` + an `ik_angle()` lookup to avoid mis-indexing; worth double-checking this maps cleanly onto Isaac Lab's own IK helpers tomorrow, which may not have the same indexing quirk.
- `GraspEnv` spawns a small static table under the graspable object (`pybullet_scene.spawn_table`) — without it the object free-falls before the arm can reach it. `hospital.usd` presumably already has real support surfaces, so this is likely PyBullet-stand-in-only and can probably be dropped tomorrow.
- **PyBullet has no official Windows wheel for Python 3.13** (this repo's system Python). Tests that need it use a `require_pybullet` fixture (`tests/conftest.py`) that skips gracefully if the import fails, so the suite is honest about coverage on a bare Windows/3.13 setup. For full local validation, a `micromamba`/conda env with Python 3.11 (which does have `pybullet` win-64 wheels via conda-forge) was used — see `tools/pbenv/`. On the GB10's Linux/aarch64 DGX OS this isn't an issue at all; standard `pip install pybullet` (or just skip straight to Isaac Sim) works.
- PyTorch's newer `torch.onnx.export` Dynamo path needs `onnxscript`, which isn't pulled in automatically — added to `requirements.txt` explicitly.

**Test suite:** `tests/` has one file per module (67 tests total) plus `test_train_smoke.py` (CLI smoke tests for `train_nav`/`train_grasp`/`play`/`export_policy`) and `test_skills.py` (the navigate_to/pick_up skill API end-to-end). Two are marked `@pytest.mark.slow` (real PPO convergence checks against a live PyBullet env and against `Pendulum-v1`) and are excluded by default — run `pytest -q -m slow` to include them.

---

## Reference — Day, Risks, Open Questions, Appendix

## Part 7 — Implementation Plan (Event Day, 9 AM – 9 PM)

| Time | Task |
| :---- | :---- |
| 9:00–10:00 | Unbox; `docker load` + weight copy from USB; kernel check; GPU smoke test |
| 10:00–12:00 | Bring up OpenClaw/NemoClaw/OpenShell; validate tool-call round trip |
| 12:00–14:00 | Step 1: hospital library scene loads cleanly into Isaac Sim; robot spawns; phone-scan path validated end-to-end |
| 14:00–16:00 | Wire the full loop: load environment → train nav → train grasp → teleop/agentic → retrain. First complete run. |
| 16:00–18:00 | Rehearse ≥10 full loops; record timing; fix the worst failure |
| 18:00–submission | Freeze code; record a backup video of a clean run; submit; prep 3-beat pitch |
| Evening | Live pitch (if top 8): air gap → hospital scene training → teleop/agentic correction → (bonus) live phone scan |

**Fallback tiers:** Tier 1 (MVP) = hospital library scene, pretrained policies as-is, live teleop only, no live training. Tier 2 (target) = + live fine-tune training shown on stage. Tier 3 (stretch) = + a live phone scan of a new space. Never demo a tier that hasn't survived three consecutive clean rehearsals on the GB10 that day.

---

## Part 8 — Risks & Mitigations

| Risk | Owner | Mitigation |
| :---- | :---- | :---- |
| Isaac Lab fine-tune (nav or grasp) doesn't converge in time | Step 2 | Warm-start from pretrained policies; bounded iteration cap; Tier 1 skips live training |
| Grasp policy fails on stage props | Step 2 | Rehearse with the actual props; keep a small known object-class set |
| Reconstruction quality poor under venue lighting | Step 1 | Hospital library scene is the default judged path; live scan is a bonus beat |
| ARM behavior differs on real GB10 at 9 AM | Shared | Everything vendored on USB; dry run on an ARM box pre-event |
| Hand-tracking latency/jitter, or arm-control precision | Step 3 | Calibrate on venue-similar lighting; USB over Wi-Fi for Iriun; joystick fallback |
| Wheelchair co-nav task not solid in time | Step 2 | It's an explicit stretch goal (§5.5) — Tier 1 pick-and-place doesn't depend on it |
| Agentic parser mishandles a compound instruction | Step 4 | Task hierarchy (§6b.4) gives each atomic skill its own success/failure check rather than trusting one LLM call end to end |
| Judges rule the phone/laptop violate "runs on the box" | Shared | Tier 1 is fully self-contained; phone/laptop framed as input feeds only |
| Memory OOM under concurrent load | Shared | Hard 105 GB ceiling + ≥20 GB headroom + watchdog + worst-case rehearsal |
| Landlock ABI < v4 on DGX OS | Shared | nftables fallback, documented honestly |
| Judge asks "is this live RL from scratch?" | Step 2 | Answer directly: warm-started fine-tune, not from-scratch training |

---

## Part 9 — Open Questions (send to organizers pre-event)

1. Exact DGX OS kernel version on the GB10 unit (gates Landlock v4 vs nftables) — discoverable only at 9 AM.
2. Is a live phone scan of the venue permitted, or stick to the pre-built hospital scene + a pre-captured "bring your own space" beat?
3. Confirm the phone (scan/teleop input) and laptop (dev machine) acting as input feeds only is acceptable under "demo must run on the GB10."
4. Judging rubric ("always-on business agent") — is "always-on" scored literally?
5. Root access on the GB10 (needed for Landlock/nftables and Docker)?
6. Any constraint on props brought for the robot to pick up, since the grasp policy trains against specific object classes?

---

## Appendix — Tool/Component Card Summary

| Owner | Tool | Role | Mem (approx.) |
| :---- | :---- | :---- | :---- |
| Step 1 | Stray Scanner (iOS) | Capture: RGB, LiDAR depth, confidence, IMU, poses, intrinsics | on-device |
| Step 1 | COLMAP | Feature matching, pose refinement | CPU/modest GPU |
| Step 1 | Depth Anything V2 (S/L) | Guided depth completion/upsampling | <2 GB / 2–4 GB |
| Step 1 | 3DGUT (`nv-tlabs/3dgrut`) | Gaussian reconstruction, distorted cameras | 15–25 GB transient |
| Step 1 | Grounding DINO + SAM 2 | Open-vocabulary semantic labeling | <2 GB / 3–5 GB |
| Step 1/4 | Nemotron | Scene-graph reasoning, OpenClaw orchestration, agentic parsing | shared |
| Step 2 | Isaac Sim 5.x + Isaac Lab | Simulation, RL training | 20–30 GB |
| Step 2 | PyBullet → PhysX 5 | Physics engine, today → tomorrow | — |
| Step 2 | RSL-RL / RL-Games / SKRL | RL libraries | — |
| Step 2/3 | cuMotion / cuRobo | Motion planning, IK, retargeting | 2–3 GB |
| Step 3 | Iriun Webcam | iPhone → virtual webcam for teleop | on-device |
| Step 3 | MediaPipe Hands / WiLoR | Hand-pose estimation | <1 GB / few GB |
| Step 4 | Nemotron / Qwen3 / Phi-4-mini | Instruction parsing, task hierarchy | shared |
| Shared | OpenClaw / NemoClaw / OpenShell | Orchestration, guardrails, sandboxing | ~8 GB |
