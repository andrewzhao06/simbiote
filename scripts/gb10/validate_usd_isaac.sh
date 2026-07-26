#!/usr/bin/env bash
# Boot Isaac Sim headlessly to validate a mapper .usd against the Step 2 contract,
# with a memory preflight in front of it.
#
# Why the preflight: GB10 has no discrete VRAM. The GPU allocates out of the same
# 128 GB unified pool as the OS, so a resident vLLM server and an Isaac Sim stage
# compete for one budget. On 2026-07-26 Isaac Sim "Full" (GUI + RTX) was already up
# when vllm-nemotron claimed --gpu-memory-utilization 0.35 (~42 GB); the driver hit
# NV_ERR_NO_MEMORY and the box hard-locked. This script refuses to start in that state.
#
# Usage: scripts/gb10/validate_usd_isaac.sh <scene.usda> [--json report.json]

set -euo pipefail

ISAAC_RELEASE="${ISAAC_RELEASE:-/home/dell/IsaacSim/_build/linux-aarch64/release}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="${SCRIPT_DIR}/validate_usd_isaac.py"

# Isaac Sim base-headless needs roughly 8-12 GB. Per the master plan's memory budget
# (hard ceiling ~105 GB, >=20 GB headroom), require 25 GB free before booting.
REQUIRED_AVAILABLE_GB="${REQUIRED_AVAILABLE_GB:-25}"

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <scene.usda> [--json report.json]" >&2
    exit 2
fi

available_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "preflight: ${available_gb} GB available (need >= ${REQUIRED_AVAILABLE_GB} GB)"

if (( available_gb < REQUIRED_AVAILABLE_GB )); then
    echo "ABORT: only ${available_gb} GB available. Free the unified pool first --" >&2
    echo "       'docker stop \$(docker ps -q --filter name=vllm)' is usually the culprit." >&2
    exit 3
fi

if command -v docker >/dev/null 2>&1; then
    running_llm=$(docker ps --format '{{.Names}}' --filter name=vllm 2>/dev/null || true)
    if [[ -n "${running_llm}" ]]; then
        echo "ABORT: a vLLM server is resident and will contend for unified memory:" >&2
        echo "${running_llm}" | sed 's/^/       /' >&2
        echo "       Stop it, or set ALLOW_RESIDENT_LLM=1 to override at your own risk." >&2
        [[ "${ALLOW_RESIDENT_LLM:-0}" == "1" ]] || exit 3
    fi
fi

# Kit exits 0 even when the embedded Python raises, so the process exit code is not
# trustworthy. Tee the run and decide from the RESULT line the validator prints.
run_log="$(mktemp -t isaac_usd_validate.XXXXXX.log)"
trap 'rm -f "${run_log}"' EXIT

set +e
"${ISAAC_RELEASE}/python.sh" "${VALIDATOR}" "$@" 2>&1 | tee "${run_log}"
set -e

if grep -q "^RESULT: PASS" "${run_log}"; then
    exit 0
fi

if grep -q "^RESULT: FAIL" "${run_log}"; then
    exit 1
fi

echo "ABORT: validator did not report a RESULT -- it crashed before finishing." >&2
grep -E "py stderr|Traceback|^FAIL" "${run_log}" | tail -20 >&2
exit 4
