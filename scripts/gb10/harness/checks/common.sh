#!/usr/bin/env bash
# Shared definitions for harness checks. Source, don't execute.
#
# Exit-code contract: 0 PASS, 1 FAIL, 2 BLOCKED (prereq missing), 3 SKIP.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dell/simbiote-Gagan1_Suraj2_Sky3_Andrew4}"
AI_ROOT="${AI_ROOT:-/home/dell/AI}"
DGRUT_ROOT="${DGRUT_ROOT:-${AI_ROOT}/repos/3dgrut}"
DGRUT_PYTHON="${DGRUT_PYTHON:-${DGRUT_ROOT}/.venv/bin/python}"
ISAAC_RELEASE="${ISAAC_RELEASE:-/home/dell/IsaacSim/_build/linux-aarch64/release}"
STAGE_ROOT="${STAGE_ROOT:-/home/dell/factoryflow/stage}"

# GB10 is sm_121 (Blackwell). Native builds must target this or they silently
# produce kernels the GPU cannot launch.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"

pass()    { echo "PASS: $*";    exit 0; }
fail()    { echo "FAIL: $*";    exit 1; }
blocked() { echo "BLOCKED: $*"; exit 2; }
skip()    { echo "SKIP: $*";    exit 3; }

require_dgrut_python() {
    [[ -x "${DGRUT_PYTHON}" ]] || blocked "3DGRUT venv interpreter missing: ${DGRUT_PYTHON}"
}
