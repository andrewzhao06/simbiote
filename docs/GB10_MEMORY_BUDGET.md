# GB10 Memory Budget — there is no VRAM to add

## The one thing to internalise

The GB10 has **no discrete VRAM**. It is a Grace-Blackwell superchip with **128 GB
of unified memory** (121.6 GB visible to the OS, 122 GiB visible to CUDA). The GPU
allocates out of the same pool as the OS, GNOME, Docker, and every Python process.

Consequences:

- `nvidia-smi` reports `FB Memory Usage: Not Supported`. That is normal, not a fault.
- Warp reporting `"cuda:0" : "NVIDIA GB10" (122 GiB, sm_121, mempool enabled)` is
  reporting the **whole machine**, not a dedicated framebuffer.
- vLLM's `--gpu-memory-utilization 0.35` does not mean "35% of a spare GPU". It means
  **~42 GB carved out of the machine's total RAM**, unavailable to anything else.
- You cannot "allocate more VRAM". You can only decide who gets the one pool.

GPU acceleration itself is already enabled and needs no work: driver 580.159.03,
CUDA 13.0, Vulkan renderer, `--/physics/cudaDevice=0`, sm_121 with native FP8.

## Postmortem — 2026-07-26 18:42, hard lock and reboot

| Time | Event |
| :---- | :---- |
| 18:29:57 | Isaac Sim **Full** (GUI + RTX renderer) launched |
| 18:38:01 | `vllm-nemotron` finished loading weights, `--gpu-memory-utilization 0.35` (~42 GB) |
| 18:42:18 | `NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY] @ mem_desc.c:1359`, repeating |
| 18:43:24 | `systemd-journald: Under memory pressure, flushing caches`; X input lagging |
| 18:44:17 | Last NVRM OOM; box unresponsive |
| 18:46:03 | Cold reboot |

The crash record from the run captured `RAM: 40.47/121.627GB` available. This is
exactly the sequencing violation the master plan warns about in Part 1: reconstruction,
a fully-loaded Isaac Sim, and a resident LLM **cannot all be resident at once**.

## Measured footprints on this box

| Workload | Peak unified memory |
| :---- | :---- |
| Isaac Sim headless, `isaacsim.exp.base.python.kit` (USD validation) | **~5 GB** |
| Isaac Sim **Full**, GUI + RTX renderer | tens of GB — the crashing configuration |
| `vllm-nemotron` at `--gpu-memory-utilization 0.35` | **~42 GB** |
| Nemotron 3 Super (120B-A12B, NVFP4) | ~60 GB |
| Nemotron 3 Nano (30B-A3B) | ~25 GB |

Budget rule from the master plan: **hard ceiling ~105 GB, keep ≥20 GB headroom.**

## Rules

1. **Never boot Isaac Sim Full while a vLLM server is resident.** Stop it first:
   `docker stop $(docker ps -q --filter name=vllm)`.
2. **Use the headless experience for anything that isn't a visual demo.** USD
   validation, contract checks, and training don't need the RTX renderer.
   `isaacsim.exp.base.python.kit` costs ~5 GB; `isaacsim.exp.full.kit` costs many times that.
3. **Check before you launch.** `awk '/MemAvailable/ {print $2/1024/1024" GB"}' /proc/meminfo`
4. **Isaac Sim Full logs at Info level by default** — the 18:29 run wrote 26 MB of
   per-frame render spam in 12 minutes. Not the crash cause, but don't leave it running.

`scripts/gb10/mapper/validate_usd_isaac.sh` enforces rules 1–3 automatically: it refuses to
start if less than 25 GB is available or if a container matching `vllm` is running
(override with `ALLOW_RESIDENT_LLM=1`).

## Swap is not a workaround

The box has a 16 GB swapfile. Swap backs **CPU** allocations only — the NVIDIA driver
cannot fault GPU allocations out to swap, which is why `NV_ERR_NO_MEMORY` fired while
14.6 GB of swap was still free. Adding swap will not prevent a repeat.

## Validating a mapper scene

```bash
scripts/gb10/mapper/validate_usd_isaac.sh /home/dell/factoryflow/stage/proxy-demo.usda \
    --json /home/dell/factoryflow/stage/proxy-demo.isaac_validation.json
```

Exit codes: `0` pass, `1` contract failure, `3` memory preflight refused,
`4` validator crashed before reporting. Kit exits 0 even when its embedded Python
raises, so the wrapper decides from the validator's `RESULT:` line, not the process code.
