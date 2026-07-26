#!/usr/bin/env bash
# Run Depth Anything 3 against Stray Scanner RGB video.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <capture-dir> <colmap-dir> <model-dir> <output-dir>" >&2
  exit 2
fi

CAPTURE="$(readlink -f "$1")"
COLMAP_DIR="$(readlink -f "$2")"
MODEL_DIR="$(readlink -f "$3")"
OUTPUT="$(readlink -m "$4")"
DA3_BIN="${DA3_BIN:-da3}"

[[ -f "$CAPTURE/rgb.mp4" ]] || {
  echo "Missing $CAPTURE/rgb.mp4" >&2
  exit 1
}
[[ -f "$MODEL_DIR/model.safetensors" ]] || {
  echo "Missing Depth Anything checkpoint in $MODEL_DIR" >&2
  exit 1
}
[[ -d "$COLMAP_DIR/sparse/0" ]] || {
  echo "COLMAP output missing sparse/0: $COLMAP_DIR" >&2
  exit 1
}
command -v "$DA3_BIN" >/dev/null

mkdir -p "$OUTPUT"
"$DA3_BIN" auto "$CAPTURE/rgb.mp4" \
  --model-dir "$MODEL_DIR" \
  --export-dir "$OUTPUT" \
  --export-format mini_npz \
  --fps 10

[[ -f "$OUTPUT/exports/mini_npz/results.npz" ]] || {
  echo "Depth Anything did not export results.npz" >&2
  exit 1
}
