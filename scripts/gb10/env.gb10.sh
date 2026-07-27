#!/bin/bash
# Environment for running the Teammate 1 mapper on the GB10.
#
# The mapper needs numpy + PIL (preview geometry) and pxr (USD parser-level
# validation). Isaac Sim's bundled interpreter has the first two; pxr lives in
# the omni.usd.libs kit extension, which setup_python_env.sh does NOT put on
# PYTHONPATH -- hence the explicit entry below. usd-core has no aarch64 wheel,
# so this is the only pxr on the box (see docs/SSD_LAYOUT.md).
#
# Usage: source scripts/gb10/env.gb10.sh, then run:
# $FF_PY -m src.cli ...

ISAAC_ROOT=/home/dell/IsaacSim/_build/linux-aarch64/release
USD_LIBS="$ISAAC_ROOT/extscache/omni.usd.libs-1.0.3+6312fa25.la64.r.cp312"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export FF_PY="$ISAAC_ROOT/kit/python/bin/python3"
export PYTHONPATH="$REPO_ROOT:$USD_LIBS"
export LD_LIBRARY_PATH="$ISAAC_ROOT/kit/python/lib:$USD_LIBS/bin:$ISAAC_ROOT/kit:$ISAAC_ROOT/kit/kernel/plugins"
export FACTORYFLOW_WORK_ROOT="/home/dell/factoryflow/stage/mapper"
