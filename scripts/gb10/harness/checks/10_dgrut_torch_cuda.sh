#!/usr/bin/env bash
# Part 1 (core env): the 3DGRUT venv has a CUDA-capable PyTorch that sees the GB10.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_dgrut_python

out=$("${DGRUT_PYTHON}" - <<'PY' 2>&1
import sys
try:
    import torch
except Exception as exc:
    print(f"IMPORT_ERROR {exc}")
    sys.exit(9)

info = {
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "available": torch.cuda.is_available(),
}
if not info["available"]:
    print(f"NO_CUDA torch={info['torch']} cuda_build={info['cuda_build']}")
    sys.exit(8)

cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
# Actually exercise the GPU -- torch.cuda.is_available() can be true while every
# kernel launch fails because the wheel was built without sm_121 support.
a = torch.randn(512, 512, device="cuda")
result = (a @ a).sum().item()
print(f"OK torch={info['torch']} cuda={info['cuda_build']} device={name} sm_{cap[0]}{cap[1]} matmul={result:.2f}")
PY
)
code=$?

case ${code} in
    0) pass "${out}" ;;
    9) blocked "torch not installed in 3DGRUT venv -- ${out}" ;;
    8) fail "torch present but CUDA unavailable -- ${out}" ;;
    *) fail "torch CUDA check errored -- ${out}" ;;
esac
