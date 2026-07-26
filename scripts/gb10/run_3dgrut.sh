#!/usr/bin/env bash
# Train 3DGUT from COLMAP sparse output and export a USDZ scene layer.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <capture-dir> <colmap-dir> <depth-dir> <output-dir>" >&2
  exit 2
fi

CAPTURE="$(readlink -f "$1")"
COLMAP_DIR="$(readlink -f "$2")"
DEPTH_DIR="$(readlink -f "$3")"
OUTPUT="$(readlink -m "$4")"
DGRUT_ROOT="${DGRUT_ROOT:?Set DGRUT_ROOT to the 3dgrut repository}"
DGRUT_PYTHON="${DGRUT_PYTHON:-$DGRUT_ROOT/.venv/bin/python}"
ITERATIONS="${DGRUT_ITERATIONS:-7000}"

[[ -d "$COLMAP_DIR/images" && -d "$COLMAP_DIR/sparse/0" ]] || {
  echo "COLMAP image/sparse layout is incomplete: $COLMAP_DIR" >&2
  exit 1
}
[[ -f "$DEPTH_DIR/exports/mini_npz/results.npz" ]] || {
  echo "Depth output missing: $DEPTH_DIR/exports/mini_npz/results.npz" >&2
  exit 1
}
[[ -f "$DGRUT_ROOT/train.py" ]] || {
  echo "DGRUT_ROOT is not a 3dgrut checkout: $DGRUT_ROOT" >&2
  exit 1
}

mkdir -p "$OUTPUT"
RUNS_DIR="$OUTPUT/runs"
EXPERIMENT="simbiote_scene"

(
  cd "$DGRUT_ROOT"
  "$DGRUT_PYTHON" train.py \
    --config-name apps/colmap_3dgut.yaml \
    "path=$COLMAP_DIR" \
    "out_dir=$RUNS_DIR" \
    "experiment_name=$EXPERIMENT" \
    "n_iterations=$ITERATIONS" \
    "export_usd.enabled=true"
)

USDZ="$(find "$RUNS_DIR/$EXPERIMENT" -type f \( -name '*.usdz' -o -name '*.usd' -o -name '*.usda' \) \
  | sort | tail -n 1)"
[[ -n "$USDZ" ]] || {
  echo "3DGRUT completed without producing a USD/USDZ export" >&2
  exit 1
}
cp "$USDZ" "$OUTPUT/reconstruction$(basename "$USDZ" | sed -E 's/.*(\.usd[acz]?)$/\1/')"
