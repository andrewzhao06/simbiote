# Simbiote — GB10 Model Download Checklist (Top-of-Line Edition)

**Get all of this onto the USB drive tonight.** This revises the earlier "safe/small" picks up to the best available option in each category — flagged clearly wherever "best" and "fits in 128 GB alongside everything else" are in tension, so you're choosing with full information, not discovering a conflict tomorrow morning.

---

## ⚠️ Read this before downloading anything

**Top-of-line everywhere, all loaded at once, does not fit in 128 GB.** That's not a soft caution — running the largest pick in every category simultaneously (Nemotron 3 Super ~60 GB + SAM 3 + Isaac Sim headless & rendered ~30–45 GB + cuMotion/hand-pose/orchestration overhead) lands at or past the ceiling on its own, before reconstruction's transient 15–25 GB is even in the picture. The fix isn't to downgrade everything — it's **sequencing**: the biggest item (Nemotron 3 Super) gets loaded and unloaded around Isaac Sim's heaviest phases rather than staying resident the whole day, the same way reconstruction and training already don't run concurrently. Full detail is in the master doc's Part 1. If that sequencing feels like too much risk to add today, **Nemotron 3 Nano is the one-line fallback** that stays resident the whole time with no juggling, at a real but smaller reasoning-quality cost.

**Nemotron 3 Ultra is excluded entirely, not just discouraged.** Even at its native NVFP4 checkpoint it's ~275 GB — more than double GB10's entire 128 GB of unified memory. This isn't a "if you have room" item; it physically cannot run on this hardware. Don't download it.

---

## Step 1 — Scan & Map

| Model | Pick | Size (approx.) | Source | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Depth completion | **Depth Anything 3** (newer generation than V2) | unconfirmed — verify tonight | Search for the official repo/checkpoint release | Cited in recent papers (Nov 2025); if the checkpoint or repo isn't cleanly available when you check, fall back to Depth Anything V2 – Large (~1.3 GB, `depth-anything` org on Hugging Face) without losing much — it's still a very strong model |
| Semantic labeling | **SAM 3** (or the SAM 3.1 update) | checkpoints vary, low single-digit GB | GitHub `facebookresearch/sam3`; also on Hugging Face | Genuine upgrade *and* a simplification: SAM 3 does open-vocabulary detect + segment + track from a short text prompt natively, replacing the earlier two-model Grounding DINO + SAM 2 combo entirely — one download instead of two |
| Gaussian reconstruction | 3DGUT (`nv-tlabs/3dgrut`) | code only | GitHub | Still the right tool — this is a per-scene optimization method, not a swappable pretrained checkpoint, so "top of line" doesn't apply the same way here |
| Feature matching | COLMAP (CUDA build) | binary | Build from source | Not a neural model — verify the ARM+CUDA build compiles |

---

## Step 2 — Physics, Simulation & Training

| Asset | Pick | Size (approx.) | Source | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Isaac Sim 5.x | latest release, ARM build | several GB | NVIDIA Omniverse / NGC | No "quality tier" here — it's the engine, get the current release |
| Environment + robot assets | `hospital.usd` + `RidgebackFranka/ridgeback_franka.usd` | several GB | Isaac Sim's Local Assets Pack | Built-in, already top-of-line for this use case |
| Vision backbone (distillation stretch) | **Theia** (robot-specific) — DINOv2 worth a look as a more general alternative if Theia sourcing is unclear | varies | Isaac Lab's documented pretrained-backbone examples; DINOv2 on Hugging Face (`facebook/dinov2-*`) | Only needed if attempting the stretch distillation goal |
| Pretrained nav warm-start checkpoint | verify tonight | varies | Isaac Lab's example checkpoints | Still flagging this honestly: a wheeled-Ridgeback-specific checkpoint isn't confirmed off-the-shelf — check before counting on it |

---

## Step 3 — Hand-Tracking Teleoperation

| Model | Pick | Size (approx.) | Source | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Hand-pose estimation | **WiLoR** | ~few hundred MB | Hugging Face Space `rolpotamias/WiLoR` — `pretrained_models/wilor_final.ckpt` + `model_config.yaml` | Already the top pick in this category — nothing clearly better turned up |
| Motion planning / IK | cuMotion / cuRobo | code + small assets | NVIDIA Isaac GitHub | — |

---

## Step 4 — Robot Prompting (Agentic Control)

| Model | Pick | Size (approx.) | Source | Notes |
| :---- | :---- | :---- | :---- | :---- |
| **Nemotron 3 Super** (120B-A12B, NVFP4) | Top-of-line pick | **~60 GB** | Hugging Face, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | Confirmed by NVIDIA's own deployment guide to run on a single DGX Spark (GB10) at `--tensor-parallel-size 1`. Needs the reasoning-parser file too: `super_v3_reasoning_parser.py`. Requires a vLLM **nightly** build (`cu130`) for the MTP+NVFP4 combination on single-GPU Spark — the pinned 0.17.1 release doesn't support it. **Must be sequenced around Isaac Sim** — see the warning above. |
| *(Simpler fallback)* Nemotron 3 Nano (30B-A3B) | Fits alongside everything, no sequencing | ~25 GB | Hugging Face, NVIDIA org | Real quality drop vs. Super, but stays resident all day with zero juggling |
| ~~Nemotron 3 Ultra~~ | **Excluded — cannot run on this hardware** | ~275 GB (NVFP4) | — | 2.5x GB10's total memory. Don't download. |

**Serving path for whichever Nemotron you pick:** `build.nvidia.com/spark/nemotron` — NVIDIA's guide covers both models on Spark/GB10 hardware specifically.

---

## Decision to make tonight, not tomorrow

**Nemotron Super vs. Nano** is the one real judgment call left on this list, and it changes your Step 2 memory planning meaningfully — decide as a group before 9 AM:
- **Super** → best reasoning quality, but Teammate 2 needs to build the load/unload sequencing around Isaac Sim's training phases, and it needs testing tomorrow morning before you trust it live.
- **Nano** → noticeably lower quality, but zero added complexity — it just sits in memory the whole day like everything else.

## Download-order priority

1. **Isaac Sim + the ARM build verification** — still the highest-risk item, do this first.
2. **Isaac Sim asset pack** (hospital + Ridgeback Franka).
3. **Whichever Nemotron you pick** — Super is a large download (~60 GB) and needs the vLLM nightly build sorted too, so start this early regardless of which way the decision above goes.
4. **SAM 3** — small relative to the above, but do it before the smaller items since it now replaces two downloads with one.
5. Everything else on this list is comparatively small and low-risk to grab later in the evening.
