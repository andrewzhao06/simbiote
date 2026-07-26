#!/usr/bin/env bash
# Verify the GB10 mapper environment before spending time on reconstruction.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <ai-root-on-ssd> <repo-root>" >&2
  exit 2
fi

AI_ROOT="$(readlink -f "$1")"
REPO_ROOT="$(readlink -f "$2")"
failures=0
warnings=0

pass() {
  printf 'PASS  %s\n' "$1"
}

fail() {
  printf 'FAIL  %s\n' "$1" >&2
  failures=$((failures + 1))
}

warn() {
  printf 'WARN  %s\n' "$1" >&2
  warnings=$((warnings + 1))
}

require_file() {
  if [[ -f "$2" ]]; then
    pass "$1: $2"
  else
    fail "$1 missing: $2"
  fi
}

require_dir() {
  if [[ -d "$2" ]]; then
    pass "$1: $2"
  else
    fail "$1 missing: $2"
  fi
}

require_command() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "command available: $1"
  else
    fail "command missing: $1"
  fi
}

if [[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
  pass "ARM64 architecture: $(uname -m)"
else
  fail "Expected GB10 ARM64, found $(uname -m)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  pass "NVIDIA driver is available"
else
  fail "nvidia-smi is unavailable"
fi

require_dir "SSD AI root" "$AI_ROOT"
require_file "Depth Anything checkpoint" "$AI_ROOT/models/depth-anything/model.safetensors"
require_file "SAM 3 checkpoint" "$AI_ROOT/models/sam3/sam3.pt"
require_dir "3DGRUT checkout" "$AI_ROOT/repos/3dgrut"
require_dir "Depth Anything checkout" "$AI_ROOT/repos/Depth-Anything-3"
require_dir "SAM 3 checkout" "$AI_ROOT/repos/sam3"
require_dir "COLMAP checkout" "$AI_ROOT/repos/colmap"
require_dir "Capture directory" "$AI_ROOT/captures"

ISAAC_PACK="$AI_ROOT/assets/isaac-5.1"
HOSPITAL_USD="$ISAAC_PACK/Isaac/Environments/Hospital/hospital.usd"
if [[ -f "$HOSPITAL_USD" ]]; then
  pass "Hospital fallback asset"
else
  warn "Hospital fallback asset missing: $HOSPITAL_USD"
fi

# The pack shipped without the /NVIDIA tree, so the hospital's dome-light sky
# silently failed to load. Cheap to check, annoying to diagnose in a viewport.
if [[ -f "$ISAAC_PACK/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr" ]]; then
  pass "Isaac pack /NVIDIA sky assets"
else
  warn "Isaac pack is missing /NVIDIA/Assets/Skies — the hospital dome light will not render"
fi

require_command ffmpeg
require_command colmap
require_command uv
require_file "3DGRUT Python" "$AI_ROOT/repos/3dgrut/.venv/bin/python"

for script in run_colmap.sh run_depth.sh run_3dgrut.sh run_sam3.sh; do
  if [[ -x "$REPO_ROOT/scripts/gb10/$script" ]]; then
    pass "Executable adapter: $script"
  else
    fail "Adapter is not executable: $REPO_ROOT/scripts/gb10/$script"
  fi
done

if [[ $failures -gt 0 ]]; then
  printf '\nPreflight failed: %d required check(s) need attention.\n' "$failures" >&2
  exit 1
fi

if [[ $warnings -gt 0 ]]; then
  printf '\nPreflight passed with %d warning(s).\n' "$warnings"
else
  printf '\nPreflight passed.\n'
fi
