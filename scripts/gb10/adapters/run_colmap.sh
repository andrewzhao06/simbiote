#!/usr/bin/env bash
# Build a COLMAP sparse reconstruction from a Stray Scanner RGB video.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <capture-dir> <output-dir>" >&2
  exit 2
fi

CAPTURE="$(readlink -f "$1")"
OUTPUT="$(readlink -m "$2")"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
COLMAP_BIN="${COLMAP_BIN:-colmap}"

[[ -f "$CAPTURE/rgb.mp4" ]] || {
  echo "Missing $CAPTURE/rgb.mp4" >&2
  exit 1
}
command -v "$FFMPEG_BIN" >/dev/null
command -v "$COLMAP_BIN" >/dev/null

mkdir -p "$OUTPUT/images"

# The source video frames become the COLMAP image set. Existing frames are
# retained to make retries resumable; delete OUTPUT to start from scratch.
if ! compgen -G "$OUTPUT/images/*.png" >/dev/null; then
  "$FFMPEG_BIN" -hide_banner -loglevel warning -i "$CAPTURE/rgb.mp4" \
    -vsync 0 "$OUTPUT/images/%06d.png"
fi

"$COLMAP_BIN" automatic_reconstructor \
  --workspace_path "$OUTPUT" \
  --image_path "$OUTPUT/images" \
  --data_type individual \
  --quality medium \
  --use_gpu 1 \
  --dense 0

[[ -e "$OUTPUT/sparse/0/cameras.bin" || -e "$OUTPUT/sparse/0/cameras.txt" ]] || {
  echo "COLMAP did not create sparse/0/cameras.*" >&2
  exit 1
}
[[ -e "$OUTPUT/sparse/0/images.bin" || -e "$OUTPUT/sparse/0/images.txt" ]] || {
  echo "COLMAP did not create sparse/0/images.*" >&2
  exit 1
}
[[ -e "$OUTPUT/sparse/0/points3D.bin" || -e "$OUTPUT/sparse/0/points3D.txt" ]] || {
  echo "COLMAP did not create sparse/0/points3D.*" >&2
  exit 1
}
