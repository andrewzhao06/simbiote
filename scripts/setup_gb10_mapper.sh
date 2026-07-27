#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/ssd/mount [repo-dir]" >&2
  exit 2
fi

SSD_ROOT="$(readlink -f "$1")"
REPO_ROOT="$(readlink -f "${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}")"
WORK_ROOT="/var/factoryflow/stage/mapper"
AI_ROOT="$SSD_ROOT"

if [[ ! -d "$SSD_ROOT" ]]; then
  echo "SSD mount does not exist: $SSD_ROOT" >&2
  exit 1
fi

if [[ -d "$SSD_ROOT/AI" ]]; then
  AI_ROOT="$SSD_ROOT/AI"
fi

echo "SSD:  $SSD_ROOT"
echo "AI:   $AI_ROOT"
echo "Repo: $REPO_ROOT"
uname -a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

sudo install -d -m 0775 "$WORK_ROOT"
sudo chown "$USER":"$(id -gn)" "$WORK_ROOT"
mkdir -p "$AI_ROOT/captures" "$AI_ROOT/assets"
chmod +x "$REPO_ROOT/scripts/gb10/"*.sh

CONFIG="$REPO_ROOT/config/mapper.gb10.toml"
cp "$REPO_ROOT/config/mapper.example.toml" "$CONFIG"
python3 - "$CONFIG" "$AI_ROOT" <<'PY'
from pathlib import Path
import sys

config = Path(sys.argv[1])
root = sys.argv[2]
text = config.read_text()
replacements = {
    "/mnt/factoryflow-ssd/AI": root,
    "/usr/local/bin/colmap": "colmap",
}
for source, target in replacements.items():
    text = text.replace(source, target)
config.write_text(text)
PY

ENV_FILE="$REPO_ROOT/config/mapper.gb10.env"
cat > "$ENV_FILE" <<EOF
# Source this before running simbiote-map in production mode.
export FACTORYFLOW_MODE=production
export DGRUT_ROOT="$AI_ROOT/repos/3dgrut"
export SAM3_ROOT="$AI_ROOT/repos/sam3"
# Point these at their separate CUDA-enabled environments after installation.
export DGRUT_PYTHON="\$DGRUT_ROOT/.venv/bin/python"
export DA3_BIN="da3"
export SAM3_PYTHON="python"
export FACTORYFLOW_COLMAP_COMMAND="$REPO_ROOT/scripts/gb10/run_colmap.sh {capture} {output}"
export FACTORYFLOW_DEPTH_COMMAND="$REPO_ROOT/scripts/gb10/run_depth.sh {capture} {colmap} {checkpoint} {output}"
export FACTORYFLOW_DGRUT_COMMAND="$REPO_ROOT/scripts/gb10/run_3dgrut.sh {capture} {colmap} {depth} {output}"
export FACTORYFLOW_SAM3_COMMAND="$REPO_ROOT/scripts/gb10/run_sam3.sh {capture} {geometry} {checkpoint} {output}"
EOF

if command -v uv >/dev/null 2>&1; then
  (cd "$REPO_ROOT" && uv sync --extra dev --extra usd)
else
  echo "uv is not installed; install it from the vendored/offline package before continuing." >&2
  exit 1
fi

echo
echo "Created $CONFIG"
echo "Created $ENV_FILE"
echo "Next:"
echo "  1. Copy the scan into '$AI_ROOT/captures/'."
echo "  2. Confirm model paths; this setup uses the SSD's existing AI/models and AI/repos layout."
echo "  3. Source '$ENV_FILE'."
echo "  4. Run: '$REPO_ROOT/scripts/gb10/preflight_mapper.sh' '$AI_ROOT' '$REPO_ROOT'"
echo "  5. Run: uv run simbiote-map --config '$CONFIG' doctor"
