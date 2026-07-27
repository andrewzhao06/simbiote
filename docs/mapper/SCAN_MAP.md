# FactoryFlow — Teammate 1: Scan the Environment → 3D Map

**Shared context below is historical Teammate 1 planning material.** For the
current project layout, see `README.md`, `../docs/GB10_DOWNLOADS.md`, and
`../docs/SSD_LAYOUT.md`.

---

## Part 0 — Project Overview

**Project:** FactoryFlow — Offline Platform for Scanning, Simulating & RL-Training Mobile Manipulator Robots
**Target:** Dell × NVIDIA Hackathon "Local AI on Dell Pro Max with GB10," Seattle Tech Week — **July 26, 2026, one-day sprint (9 AM – 9 PM)**
**Hardware:** Dell Pro Max with GB10 (provided day-of) + team laptops (dev machines) + an iPhone with LiDAR (capture + teleop camera)
**Team:** 4 people, one per role (see Parts 4–7): **Teammate 1** Scan & Map · **Teammate 2** Isaac Sim, Physics & Training · **Teammate 3** Hand-Tracking Teleoperation · **Teammate 4** Robot Prompting (Agentic Control)

### What FactoryFlow is

A platform other robotics companies use to train mobile-manipulator robots with reinforcement learning, entirely offline. A robotics company points FactoryFlow at their robot spec and either their own facility (scanned with a phone) or a ready-made environment from a library (a hospital, for this demo), and gets back a robot that's learned to navigate that space without collisions and pick up objects in it — plus a way to remotely operate and further train that same robot with tracked hand movements or plain-language instructions. The whole loop runs air-gapped, on one GB10, in a single sitting.

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
- Filesystem: read-only reconstruction/training weights + environment-library directory; write access only to `/var/factoryflow/stage/` (scanned maps, trained policies, teleop logs).

### NemoClaw guardrails

- **Ingress sanitization:** strips identifiers from phone-scan video and hand-tracking streams before they reach any model or log.
- **Role-based policies (per-agent):** Mapper and Trainer are read-only outside their own stage; only Planner/Deployer may call `start_teleop()` or push a policy update. No agent executes a generated script that hasn't passed static scan.
- **Egress-to-execution:** static analysis of every auto-generated USD injection script and PhysX/training config before Isaac Sim executes it.
- **Audit trail as pitch closer:** every environment load, training run, teleop/agentic session, and policy update logged in structured form.

---

## Part 4 — Step 1: Scan the Environment → 3D Map
**(Teammate 1's step)**

### 4.1 What this step does

A phone sweep becomes a simulation-ready `.usd` scene, tagged with navigable floor and graspable objects, that Step 2 loads directly into Isaac Sim.

```
iPhone (Stray Scanner)
├── RGB Images (rgb.mp4)              ├── Camera Intrinsics (camera_matrix.csv + per-frame in odometry.csv)
├── LiDAR (depth/*.png + confidence/*.png)  ├── Camera Extrinsics — ARKit VIO (odometry.csv)
├── IMU (imu.csv)                     └── Timestamps (imu.csv, odometry.csv)
     ▼
Capture Engine (Stray Scanner itself — nothing custom to build)
     ▼
Sensor Fusion  →  COLMAP + ARKit + IMU Optimization  →  3DGUT / Gaussian Reconstruction
     ▼
Semantic AI  →  Scene Graph  →  Physics + Robotics Metadata
     ▼
OpenUSD  →  handed to Step 2 (Isaac Sim / Omniverse / AI Agents)
```

**Hardware requirement:** Stray Scanner needs a LiDAR-equipped device — iPhone 12 Pro/Pro Max or later Pro-line, or an iPad Pro. Confirm this before relying on it tomorrow morning.

### 4.2 Confirmed capture format (verified against a real test upload)

```
<scan>/
├── camera_matrix.csv     — static 3×3 intrinsic matrix
├── odometry.csv          — per frame: timestamp, frame, x, y, z, qx, qy, qz, qw,
│                            fx, fy, cx, cy, distortion_center_x, distortion_center_y
├── imu.csv               — timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z (~100 Hz)
├── depth/000000.png ...  — one 192×256 depth map per RGB frame (verify the mm→m scale
│                            factor empirically rather than trusting it blindly)
├── confidence/000000.png ... — one confidence map per RGB frame (0 low / 1 med / 2 high)
└── rgb.mp4                — HEVC video, one frame per odometry.csv row
```

Match depth/confidence files to RGB frames by the `frame` column, not list position — Stray Scanner has had frame-drop bugs historically.

### 4.3 Pipeline mechanics, stage by stage

- **Sensor Fusion:** temporal alignment of IMU onto each frame's timestamp; depth-to-RGB registration (192×256 → ~1920×1440, roughly a 7.5× gap — a real engineering task, not an afterthought); confidence-based filtering (keep confidence ≥ 1, treat 2 as hard constraint); coordinate unification into `odometry.csv`'s own frame.
- **COLMAP + ARKit + IMU Optimization:** seed COLMAP bundle adjustment with `odometry.csv` poses rather than solving cold; add the real, confidence-filtered LiDAR depth as a depth-consistency term; use IMU preintegration to catch and down-weight motion-blurred frames.
- **3DGUT / Gaussian Reconstruction (`nv-tlabs/3dgrut`):** trains directly on distorted, rolling-shutter phone frames via the Unscented Transform rather than requiring pre-undistortion; regularized against real LiDAR depth, which reduces floating-artifact noise in low-texture regions (blank walls, ceilings — exactly what a hospital corridor has plenty of).
- **Semantic AI:** **SAM 3** (Meta, open-vocabulary concept segmentation) over posed RGB frames — detects, segments, *and* projects into 3D from a short text prompt directly (e.g. "navigable floor," "graspable tray"), replacing the earlier Grounding DINO + SAM 2 two-model combo with one model that does both jobs natively.
- **Scene Graph:** each labeled region/object becomes a node (label, bounding volume, pose); relationships become edges (on-top-of, blocks-path, etc.) — this is the layer OpenClaw/Nemotron agents query directly.
- **Physics + Robotics Metadata:** collision mesh generation, mass/friction defaults by semantic class, navigable-floor tagging (feeds Step 2's nav costmap), grasp-affordance tagging (feeds Step 2's grasp targets).
- **OpenUSD export:** `UsdGeom` for mesh, `UsdPhysics` for collision, custom schema attributes for semantic/scene-graph data — the portable file Step 2 loads directly, and the reason any USD-aware tool (not just Isaac Sim) can open it too.

### 4.4 Models — laptop (today) vs GB10 (tomorrow)

| Stage | Laptop (today) | GB10 (tomorrow) |
| :---- | :---- | :---- |
| Feature matching | COLMAP, CPU/modest GPU, sequential matcher | COLMAP, CUDA build, exhaustive/vocab-tree matcher |
| Depth completion | Depth Anything V2 – **Small**, guided upsampling of real LiDAR | **Depth Anything 3** (newer generation — verify exact checkpoint/repo tonight) or Depth Anything V2 – **Large** as the known-good fallback |
| Gaussian reconstruction | 3DGUT at reduced density/iterations/resolution | 3DGUT at full density/iterations/resolution, `--with_ut` on |
| Semantic labeling | Grounding DINO (tiny/base) + SAM 2 – tiny/small (laptop-friendly stand-in) | **SAM 3** (or SAM 3.1) — one model, native text-prompt concept segmentation, replaces the two-model laptop combo entirely |
| Scene-graph reasoning | Phi-4-mini (3.8B) or Gemma 3 4B | **Nemotron 3 Super** (120B-A12B, NVFP4, ~60 GB) — confirmed to run on a single GB10; shared with OpenClaw orchestration and Step 4. Can't stay resident alongside a fully-loaded Isaac Sim (see Part 1's sequencing note) — **Nemotron 3 Nano** (~25 GB) is the simpler always-resident fallback |

Confidence handling either way: level-2 depth pixels are hard anchors, level-1 soft, level-0 dropped before it reaches the upsampling model.

### 4.5 Build spec

```
factoryflow/
├── capture_ingest/
│   └── ingest.py         — def load_capture_bundle(path: str) -> FusedFrames
│                            Parses the confirmed schema directly (§4.2)
├── reconstruction/
│   ├── colmap_refine.py  — def refine_poses(frames: FusedFrames) -> RefinedPoses
│   └── gaussian_recon.py — def reconstruct(refined: RefinedPoses) -> GaussianScene
├── semantic/
│   └── label_scene.py    — def label(scene: GaussianScene) -> LabeledScene
├── scenegraph/
│   └── build_graph.py    — def build_scene_graph(labeled: LabeledScene) -> SceneGraph
│                            LLM call; define and validate the JSON schema explicitly
├── physics_meta/
│   └── attach_physics.py — def attach_physics(graph: SceneGraph) -> PhysicsScene
├── usd_export/
│   └── export.py         — def export_usd(scene: PhysicsScene, out_path: str) -> None
└── run_pipeline.py        — CLI: ingest → refine → reconstruct → label →
                              build_graph → attach_physics → export_usd
```

**Build order:** `ingest.py` → `colmap_refine.py` → `gaussian_recon.py` → `label_scene.py` → `build_graph.py` → `attach_physics.py` → `export.py` → wire together in `run_pipeline.py`. Each step has its own sanity check before moving to the next — don't chain them blind.

**Acceptance tests:**
- *Capture itself:* real translation through the space (multiple meters, moving between areas), not a short in-place pan — the first test upload was ~1 m of translation over 8.5 s, enough to validate `ingest.py`'s parsing, not enough baseline for COLMAP to meaningfully refine.
- *Full pipeline:* `python run_pipeline.py --capture path/to/scan --out hospital_test.usd`, then confirm the `.usd` opens cleanly in Isaac Sim with geometry recognizable as the scanned room, at least one navigable-floor region, and at least one graspable object.

### 4.6 Tonight's realistic scope

Get one or two real, full-length Stray Scanner walkthroughs done (depth/ and confidence/ folders included). Build and validate `ingest.py` through `attach_physics.py` on your laptop using the laptop-tier models — none of that needs the GB10. The Isaac Sim open-check is tomorrow morning's first task.

### 4.7 Handoff to Step 2 (Teammate 2)

**You produce:** a `.usd` file with geometry, semantic labels, navigable-floor tags, and grasp-affordance tags attached via `UsdPhysics`/custom schema. **They consume it** via `load_environment()` and `validate_map()` — make sure `validate_map()`'s checks (watertight floor, correct scale, at least one tagged pickable object) are things your export actually guarantees, since that's the contract Step 2 is coding against.

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
