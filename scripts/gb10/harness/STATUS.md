# GB10 build-out status

Long-horizon tracker for the 3DGRUT build and the event stack. **Ground truth is
`./run_checks.sh`, not this file** — every claim below must correspond to a check
that flips `BLOCKED → PASS` and stays there.

Run it: `scripts/gb10/harness/run_checks.sh` · latest machine-readable report:
`reports/latest.json`

## Parts

<!-- BEGIN GENERATED: parts (render_status.py) -->
<!-- generated 2026-07-26T22:07:59Z from reports/20260726T220733Z.json -- do not edit by hand, run ./render_status.py -->

| # | Part | Checks | Owner | State |
| :-- | :---- | :---- | :---- | :---- |
| 1 | 3DGRUT core env — venv, PyTorch cu130/aarch64, editable install | `10_`, `20_` | agent A | BLOCKED <br><sub>1/2 pass</sub> |
| 2 | Native builds — tiny-cuda-nn, Kaolin from source on CUDA 13 | `30_` | agent B | BLOCKED <br><sub>0/1 pass</sub> |
| 3 | 3DGUT tracer + slangc + extra requirements | `40_` | agent C | BLOCKED <br><sub>0/1 pass</sub> |
| 4 | Mapper integration — preflight green, production mode reachable | `50_` | agent C | **PASS** <br><sub>1/1 pass</sub> |
| 5 | Isaac Sim handoff — stage meets the Step 2 contract | `60_` | done | **PASS** <br><sub>1/1 pass</sub> |
| 6 | Event stack — OpenClaw / NemoClaw / OpenShell | `70_`, `71_`, `72_` | agent D | BLOCKED <br><sub>0/1 pass, 2 check(s) not written</sub> |
| 7 | Autonomous test harness — regression detection, watch mode | — | agent E | **PASS** <br><sub>harness itself: trend.py / watch.sh / render_status.py</sub> |

Source: `reports/20260726T220733Z.json` — PASS 3 · FAIL 0 · BLOCKED 4 · SKIP 0. Regenerate with `./render_status.py`; trend with `./trend.py`.
<!-- END GENERATED: parts -->

## Standing constraints

- **No discrete VRAM.** GPU allocations come out of the same 128 GB as the OS.
  A resident vLLM server plus a GUI Isaac Sim hard-locked this box on 2026-07-26.
  See `docs/GB10_MEMORY_BUDGET.md`. Cap build parallelism; check `MemAvailable` first.
- **aarch64 + CUDA 13 + sm_121.** Generic x86 wheels do not apply. Native builds must
  target `TORCH_CUDA_ARCH_LIST=12.1` or they produce kernels the GPU cannot launch.
- **`sudo` requires a password and is unavailable.** No `apt install`; build against
  what is already on the box.
- 3DGUT (rasterisation) is what the mapper needs. 3DGRT (ray tracing, OptiX) is
  optional — do not let it block the mapper path.
