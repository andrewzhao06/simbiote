#!/usr/bin/env bash
# Long-horizon watch loop: run the check suite on an interval, record one line of
# history per run, and shout when something that used to PASS stops passing.
#
# Designed to be left running for hours on the GB10 box:
#   * ONE run at a time. Never concurrent -- an flock guarantees a single watcher,
#     and each cycle waits for run_checks.sh to finish before sleeping.
#   * MEMORY SAFE. 128 GB unified, no discrete VRAM: GPU allocations come out of
#     the same pool as the OS. This loop pre-checks MemAvailable and *raises*
#     run_checks.sh's own floor -- it never lowers it. When memory is tight it
#     backs off exponentially instead of hammering.
#   * BOUNDED DISK. Old report dirs are pruned, history.jsonl and the watch log
#     are rotated. Leaving this running does not fill the disk.
#
# Usage:
#   ./watch.sh                     # every 15 min, forever
#   ./watch.sh --interval 300      # every 5 min
#   ./watch.sh --once              # single cycle then exit (exit 1 on regression)
#   ./watch.sh --iterations 3
#   ./watch.sh --keep 10           # keep 10 report dirs
#   ./watch.sh --min-gb 30         # require 30 GB free (floor is 20, never lower)
#
# Env equivalents: WATCH_INTERVAL WATCH_KEEP WATCH_MIN_GB WATCH_MAX_HISTORY
#                  WATCH_LOG_MAX_BYTES WATCH_ON_REGRESSION
# WATCH_ON_REGRESSION, if set, is run as a shell command with the report stamp
# as $1 whenever a regression is detected (notification hook).

set -uo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${HARNESS_DIR}/reports"
HISTORY="${REPORTS_DIR}/history.jsonl"
WATCH_LOG="${REPORTS_DIR}/watch.log"
LOCK_FILE="${REPORTS_DIR}/.watch.lock"

INTERVAL="${WATCH_INTERVAL:-900}"
KEEP="${WATCH_KEEP:-24}"
# run_checks.sh refuses to start below 20 GB. We honour that as a hard floor.
HARD_FLOOR_GB=20
MIN_GB="${WATCH_MIN_GB:-${HARD_FLOOR_GB}}"
MAX_HISTORY="${WATCH_MAX_HISTORY:-2000}"
LOG_MAX_BYTES="${WATCH_LOG_MAX_BYTES:-$((5 * 1024 * 1024))}"
ITERATIONS=0          # 0 = forever
ONCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)   INTERVAL="$2"; shift 2 ;;
        --keep)       KEEP="$2"; shift 2 ;;
        --min-gb)     MIN_GB="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --once)       ONCE=1; ITERATIONS=1; shift ;;
        -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "watch: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

if (( MIN_GB < HARD_FLOOR_GB )); then
    echo "watch: --min-gb ${MIN_GB} is below the ${HARD_FLOOR_GB} GB floor; clamping." >&2
    MIN_GB=${HARD_FLOOR_GB}
fi
# Raise run_checks.sh's guard to match ours. Never lower it.
export MIN_AVAILABLE_GB="${MIN_GB}"

mkdir -p "${REPORTS_DIR}"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

available_gb() { awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo; }

# ---- single-watcher lock -----------------------------------------------------
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "watch: another watcher already holds ${LOCK_FILE}; refusing to run two." >&2
    exit 2
fi
echo "$$" >&9

# ---- clean shutdown ----------------------------------------------------------
RUNNING=1
CHILD_PID=""
SLEEP_PID=""
shutdown() {
    RUNNING=0
    [[ -n "${SLEEP_PID}" ]] && kill "${SLEEP_PID}" 2>/dev/null
    if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        log "stopping: terminating in-flight run_checks (pid ${CHILD_PID})"
        kill -TERM "${CHILD_PID}" 2>/dev/null
        for _ in $(seq 1 30); do
            kill -0 "${CHILD_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${CHILD_PID}" 2>/dev/null
    fi
}
trap shutdown INT TERM
# The lock file is deliberately NOT removed on exit: unlinking a file you still
# hold an flock on races with the next watcher creating and locking a new inode.

# ---- housekeeping ------------------------------------------------------------
rotate_file() {  # rotate_file PATH MAX_BYTES
    local f="$1" max="$2" size
    [[ -f "${f}" ]] || return 0
    size=$(stat -c %s "${f}" 2>/dev/null || echo 0)
    if (( size > max )); then
        mv -f "${f}" "${f}.1"
        log "rotated ${f} (${size} bytes) -> ${f}.1"
    fi
}

rotate_history() {
    [[ -f "${HISTORY}" ]] || return 0
    local lines
    lines=$(wc -l < "${HISTORY}")
    if (( lines > MAX_HISTORY )); then
        # Keep the most recent half in place; the rest is archived (one generation).
        local keep=$(( MAX_HISTORY / 2 ))
        head -n $(( lines - keep )) "${HISTORY}" >> "${HISTORY}.1"
        tail -n "${keep}" "${HISTORY}" > "${HISTORY}.tmp" && mv -f "${HISTORY}.tmp" "${HISTORY}"
        # Bound the archive too, so hours of running cannot grow it without limit.
        rotate_file "${HISTORY}.1" $(( 20 * 1024 * 1024 ))
        log "rotated history (${lines} lines -> ${keep})"
    fi
}

prune_reports() {
    local stamped
    mapfile -t stamped < <(find "${REPORTS_DIR}" -maxdepth 1 -name '2*Z.json' -printf '%f\n' | sort -r)
    (( ${#stamped[@]} <= KEEP )) && return 0
    local i name stem
    for (( i = KEEP; i < ${#stamped[@]}; i++ )); do
        name="${stamped[$i]}"
        stem="${name%.json}"
        # Never delete whatever latest.json points at.
        [[ "$(readlink -f "${REPORTS_DIR}/latest.json" 2>/dev/null)" == "${REPORTS_DIR}/${name}" ]] && continue
        rm -rf -- "${REPORTS_DIR:?}/${stem}" "${REPORTS_DIR:?}/${name}"
    done
    log "pruned $(( ${#stamped[@]} - KEEP )) old report(s), keeping ${KEEP}"
}

append_history() {  # append_history STATUS RUN_EXIT AVAIL_GB DURATION TREND_JSON_FILE
    STATUS="$1" RUN_EXIT="$2" AVAIL="$3" DURATION="$4" TRENDF="$5" \
    HISTORY="${HISTORY}" REPORT_LINK="${REPORTS_DIR}/latest.json" \
    python3 - <<'PY'
import json, os, time

trend_path = os.environ["TRENDF"]
trend = {}
if trend_path and os.path.exists(trend_path) and os.path.getsize(trend_path):
    try:
        trend = json.load(open(trend_path))
    except json.JSONDecodeError:
        trend = {}

summary, stamp = {}, None
link = os.environ["REPORT_LINK"]
if os.path.exists(link):
    try:
        rep = json.load(open(link))
        summary = rep.get("summary", {})
        stamp = rep.get("timestamp")
    except (json.JSONDecodeError, OSError):
        pass

changes = [
    {"check": c["check"], "from": c["previous"], "to": c["current"], "category": c["category"]}
    for c in trend.get("changes", [])
]
row = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "report": stamp,
    "status": os.environ["STATUS"],
    "run_exit": int(os.environ["RUN_EXIT"]),
    "available_gb": int(os.environ["AVAIL"]),
    "duration_s": int(os.environ["DURATION"]),
    "summary": summary,
    "changed": changes,
    "regressions": [r["check"] for r in trend.get("regressions", [])],
}
with open(os.environ["HISTORY"], "a") as fh:
    fh.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps(row, sort_keys=True))
PY
}

# ---- one cycle ---------------------------------------------------------------
consecutive_skips=0
regressions_seen=0
cycle=0
LAST_STAMP=""

run_cycle() {
    local avail started duration run_exit trend_file trend_exit status
    avail=$(available_gb)

    if (( avail < MIN_GB )); then
        consecutive_skips=$(( consecutive_skips + 1 ))
        log "SKIP cycle: only ${avail} GB available, need >= ${MIN_GB} GB (backing off)"
        append_history "skipped-low-memory" 3 "${avail}" 0 "" >/dev/null
        return 0
    fi

    log "cycle ${cycle}: ${avail} GB available, running suite"
    started=$(date -u +%s)

    local run_out
    run_out=$(mktemp "${TMPDIR:-/tmp}/watch-run.XXXXXX")
    "${HARNESS_DIR}/run_checks.sh" >"${run_out}" 2>&1 &
    CHILD_PID=$!
    wait "${CHILD_PID}"
    run_exit=$?
    CHILD_PID=""
    duration=$(( $(date -u +%s) - started ))

    cat "${run_out}"
    rm -f "${run_out}"

    if (( RUNNING == 0 )); then
        log "interrupted mid-run; not recording a partial cycle"
        return 0
    fi

    if (( run_exit == 3 )); then
        # run_checks.sh aborted on its own memory guard; no report was written.
        consecutive_skips=$(( consecutive_skips + 1 ))
        log "run_checks aborted (exit 3) -- memory guard or no checks matched"
        append_history "aborted" "${run_exit}" "${avail}" "${duration}" "" >/dev/null
        return 0
    fi
    consecutive_skips=0

    # run_checks.sh names reports with 1-second granularity. Two runs inside the
    # same second overwrite one report and one log dir, which silently erases a
    # transition -- a PASS->FAIL can vanish. Detect and report it rather than
    # trusting a comparison we know is degenerate.
    local stamp
    stamp=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("timestamp",""))' \
            "${REPORTS_DIR}/latest.json" 2>/dev/null)
    if [[ -n "${stamp}" && "${stamp}" == "${LAST_STAMP}" ]]; then
        log "WARN: report stamp ${stamp} collided with the previous cycle -- run_checks.sh"
        log "WARN: overwrote it. This cycle's trend is unreliable; spacing the next run."
        append_history "stamp-collision" "${run_exit}" "${avail}" "${duration}" "" >/dev/null
        sleep 2
        return 0
    fi
    LAST_STAMP="${stamp}"

    trend_file=$(mktemp "${TMPDIR:-/tmp}/watch-trend.XXXXXX")
    "${HARNESS_DIR}/trend.py" --json >"${trend_file}" 2>/dev/null
    trend_exit=$?
    if (( trend_exit == 2 )); then
        log "trend: first report -- no baseline to compare against yet"
        : > "${trend_file}"
    fi

    status="ok"
    (( run_exit != 0 )) && status="failing"
    (( trend_exit == 1 )) && status="regression"

    local row
    row=$(append_history "${status}" "${run_exit}" "${avail}" "${duration}" "${trend_file}")
    log "history += ${row}"

    if (( trend_exit == 1 )); then
        regressions_seen=$(( regressions_seen + 1 ))
        echo
        echo "################################################################"
        echo "###  REGRESSION DETECTED  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
        echo "################################################################"
        "${HARNESS_DIR}/trend.py" --quiet
        echo "################################################################"
        echo
        if [[ -n "${WATCH_ON_REGRESSION:-}" ]]; then
            bash -c "${WATCH_ON_REGRESSION}" watch-hook \
                "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("current",""))' "${trend_file}" 2>/dev/null)" \
                || log "regression hook exited ${?}"
        fi
    else
        "${HARNESS_DIR}/trend.py" --quiet 2>/dev/null | sed -n '1,6p'
    fi

    rm -f "${trend_file}"

    # Keep STATUS.md honest after every run.
    "${HARNESS_DIR}/render_status.py" >/dev/null 2>&1 \
        || log "render_status failed (STATUS.md may be stale)"

    prune_reports
    rotate_history
    rotate_file "${WATCH_LOG}" "${LOG_MAX_BYTES}"
}

# ---- main loop ---------------------------------------------------------------
main() {
    log "watch started: pid $$, interval ${INTERVAL}s, keep ${KEEP} reports, min ${MIN_GB} GB"
    log "history: ${HISTORY}"
    while (( RUNNING )); do
        cycle=$(( cycle + 1 ))
        run_cycle
        (( RUNNING )) || break
        if (( ITERATIONS > 0 )) && (( cycle >= ITERATIONS )); then
            break
        fi
        # Exponential back-off while memory is tight, capped at 8x.
        local factor=1
        if (( consecutive_skips > 0 )); then
            factor=$(( 1 << (consecutive_skips < 3 ? consecutive_skips : 3) ))
            log "backing off ${factor}x due to ${consecutive_skips} low-memory skip(s)"
        fi
        local nap=$(( INTERVAL * factor ))
        log "sleeping ${nap}s"
        sleep "${nap}" &
        SLEEP_PID=$!
        wait "${SLEEP_PID}" 2>/dev/null
        SLEEP_PID=""
    done
    log "watch stopped after ${cycle} cycle(s); ${regressions_seen} regression event(s)"
    (( regressions_seen > 0 )) && return 1
    return 0
}

# Everything the loop prints is teed into a size-capped watch log. This uses a
# process substitution rather than a pipeline so that main() runs in *this*
# shell -- a pipeline would put it in a subshell, where the INT/TERM traps are
# reset to default and an in-flight run_checks would be orphaned on shutdown.
rotate_file "${WATCH_LOG}" "${LOG_MAX_BYTES}"
exec > >(tee -a "${WATCH_LOG}") 2>&1
main
rc=$?
exec >&- 2>&-
exit "${rc}"
