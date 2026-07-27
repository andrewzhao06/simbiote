#!/usr/bin/env bash
# Hand-teleoperate the Ridgeback+Franka through hospital.usd.
#
# Starts both halves of the split described in simbiote/teleop/action_bridge.py:
#   1. Isaac Sim (its own bundled python)  -- the hospital + robot window
#   2. teleop     (the repo .venv)         -- camera, WiLoR, and the webcam window
#
# Usage:
#   scripts/gb10/run_teleop_hospital.sh http://172.16.11.185:4747/video/640x480
#   scripts/gb10/run_teleop_hospital.sh                 # uses $SIMBIOTE_CAMERA_URL
#
# Ctrl+C stops both.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ISAAC_PY="/home/dell/IsaacSim/_build/linux-aarch64/release/python.sh"
VENV_PY="$REPO/.venv/bin/python"
PORT="${SIMBIOTE_TELEOP_PORT:-47800}"
CAMERA_URL="${1:-${SIMBIOTE_CAMERA_URL:-}}"

if [[ -z "$CAMERA_URL" ]]; then
    echo "error: no camera given." >&2
    echo "  usage: $0 http://<iphone-ip>:4747/video/640x480" >&2
    echo "  (or export SIMBIOTE_CAMERA_URL first). See docs/TELEOP_IPHONE_CAMERA.md" >&2
    exit 2
fi

for path in "$ISAAC_PY" "$VENV_PY"; do
    [[ -x "$path" ]] || { echo "error: missing interpreter $path" >&2; exit 2; }
done

# Fail fast on an unreachable phone rather than after a multi-minute Isaac boot.
# The app rejects HEAD, so probe with a ranged GET.
echo "==> checking camera $CAMERA_URL"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 -r 0-1000 "$CAMERA_URL" || echo 000)
if [[ "$code" != "200" ]]; then
    echo "error: camera returned HTTP $code (want 200)." >&2
    echo "  Phone on the same Wi-Fi? App foregrounded? See docs/TELEOP_IPHONE_CAMERA.md" >&2
    exit 1
fi
echo "    camera OK"

ISAAC_PID=""
TELEOP_PID=""
cleanup() {
    echo
    echo "==> shutting down"
    [[ -n "$TELEOP_PID" ]] && kill "$TELEOP_PID" 2>/dev/null
    [[ -n "$ISAAC_PID" ]] && kill "$ISAAC_PID" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "==> starting Isaac Sim (hospital + Ridgeback/Franka). First boot takes a few minutes."
"$ISAAC_PY" scripts/gb10/teleop_hospital.py --port "$PORT" "${@:2}" &
ISAAC_PID=$!

# Isaac has to be bound before teleop starts firing, or the first datagrams are
# dropped into a closed port. Wait for the listener rather than sleeping blind.
echo "==> waiting for Isaac Sim to open udp://127.0.0.1:$PORT"
for _ in $(seq 1 900); do
    if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
        echo "error: Isaac Sim exited during startup" >&2
        exit 1
    fi
    if grep -qa "listening for teleop" /proc/"$ISAAC_PID"/fd/1 2>/dev/null; then break; fi
    # /proc fd inspection is unreliable across pipes; fall back to a port probe.
    if ss -lun 2>/dev/null | grep -q ":$PORT\b"; then break; fi
    sleep 1
done
echo "==> Isaac Sim is listening"

echo "==> starting teleop (camera + hand tracking + preview window)"
"$VENV_PY" scripts/teleop/run_demo.py \
    --sink udp --udp-port "$PORT" \
    --camera-url "$CAMERA_URL" \
    --backend wilor &
TELEOP_PID=$!

wait -n "$ISAAC_PID" "$TELEOP_PID"
