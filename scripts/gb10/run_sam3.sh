#!/usr/bin/env bash
# Produce the mapper detections.json contract from an official SAM 3 checkpoint.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <capture-dir> <geometry-usd> <sam3-model-dir> <output-dir>" >&2
  exit 2
fi

CAPTURE="$(readlink -f "$1")"
GEOMETRY="$(readlink -f "$2")"
MODEL_DIR="$(readlink -f "$3")"
OUTPUT="$(readlink -m "$4")"
SAM3_ROOT="${SAM3_ROOT:?Set SAM3_ROOT to the sam3 repository}"
SAM3_PYTHON="${SAM3_PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -f "$CAPTURE/rgb.mp4" ]] || {
  echo "Missing $CAPTURE/rgb.mp4" >&2
  exit 1
}
[[ -f "$GEOMETRY" ]] || {
  echo "Reconstruction geometry missing: $GEOMETRY" >&2
  exit 1
}
CHECKPOINT="$MODEL_DIR/sam3.pt"
[[ -f "$CHECKPOINT" ]] || {
  echo "Missing SAM 3 checkpoint: $CHECKPOINT" >&2
  exit 1
}
[[ -f "$SAM3_ROOT/sam3/model_builder.py" ]] || {
  echo "SAM3_ROOT is not a sam3 checkout: $SAM3_ROOT" >&2
  exit 1
}

mkdir -p "$OUTPUT"
PYTHONPATH="$SAM3_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$SAM3_PYTHON" \
  "$SCRIPT_DIR/sam3_labels.py" \
  --capture "$CAPTURE" \
  --geometry "$GEOMETRY" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT/detections.json"
