#!/usr/bin/env bash
# Part 2 (native builds): tiny-cuda-nn and Kaolin are importable and GPU-functional.
# These are the two CUDA-13/aarch64 builds most likely to break.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_dgrut_python

out=$("${DGRUT_PYTHON}" - <<'PY' 2>&1
import sys

missing, broken, ok = [], [], []

try:
    import torch
except Exception as exc:
    print(f"NO_TORCH {exc}")
    sys.exit(9)

try:
    import tinycudann as tcnn
    enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 4,
                            "n_features_per_level": 2, "log2_hashmap_size": 15,
                            "base_resolution": 16, "per_level_scale": 1.5}).cuda()
    y = enc(torch.rand(64, 3, device="cuda"))
    ok.append(f"tinycudann(out={tuple(y.shape)})")
except ImportError as exc:
    missing.append(f"tinycudann({exc})")
except Exception as exc:
    broken.append(f"tinycudann({type(exc).__name__}: {exc})")

try:
    import kaolin
    from kaolin.ops.mesh import index_vertices_by_faces
    verts = torch.rand(1, 4, 3, device="cuda")
    faces = torch.tensor([[0, 1, 2], [1, 2, 3]], device="cuda", dtype=torch.long)
    fv = index_vertices_by_faces(verts, faces)
    ok.append(f"kaolin {kaolin.__version__}(out={tuple(fv.shape)})")
except ImportError as exc:
    missing.append(f"kaolin({exc})")
except Exception as exc:
    broken.append(f"kaolin({type(exc).__name__}: {exc})")

if broken:
    print("BROKEN " + "; ".join(broken) + (" | ok: " + ", ".join(ok) if ok else ""))
    sys.exit(8)
if missing:
    print("MISSING " + "; ".join(missing) + (" | ok: " + ", ".join(ok) if ok else ""))
    sys.exit(9)
print("OK " + ", ".join(ok))
PY
)
code=$?

case ${code} in
    0) pass "${out}" ;;
    9) blocked "native extensions not installed yet -- ${out}" ;;
    8) fail "native extensions installed but failing on GPU -- ${out}" ;;
    *) fail "native extension check errored -- ${out}" ;;
esac
