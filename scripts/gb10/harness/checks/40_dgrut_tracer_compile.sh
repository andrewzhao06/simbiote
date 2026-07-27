#!/usr/bin/env bash
# Part 3 (reconstruction core): the 3DGUT tracer's CUDA/slang module compiles and loads.
#
# This is the capability the mapper actually needs. 3DGUT is the rasterisation path
# (Unscented Transform, handles the phone's rolling shutter and lens distortion);
# it is what config/mapper.gb10.env's FACTORYFLOW_DGRUT_COMMAND drives. The separate
# 3DGRT ray-tracing path needs OptiX and is NOT required by the mapper -- if only
# 3DGRT is broken, this check should still pass and 41_ should record the gap.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_dgrut_python

# First import triggers a JIT CUDA build that can take many minutes and a lot of RAM.
out=$(cd "${DGRUT_ROOT}" && "${DGRUT_PYTHON}" - <<'PY' 2>&1 | tail -5
import sys
try:
    import torch
    from threedgut_tracer import Tracer  # noqa: F401
except ImportError as exc:
    print(f"MISSING {type(exc).__name__}: {exc}")
    sys.exit(9)
except Exception as exc:
    print(f"BUILD_ERROR {type(exc).__name__}: {exc}")
    sys.exit(8)
print("OK threedgut_tracer imported and CUDA module loaded")
PY
)
code=${PIPESTATUS[0]:-$?}

case ${code} in
    0) pass "${out}" ;;
    9) blocked "3DGUT tracer not built yet -- ${out}" ;;
    8) fail "3DGUT tracer present but failed to build/load -- ${out}" ;;
    *) fail "3DGUT tracer check errored -- ${out}" ;;
esac
