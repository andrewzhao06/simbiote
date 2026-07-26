#!/usr/bin/env bash
# Autonomous verification harness for the GB10 build-out.
#
# Every part of the long-horizon work owns one check in checks/. A check is any
# executable that prints human-readable detail on stdout and exits:
#
#   0  PASS     the capability works, verified by actually exercising it
#   1  FAIL     the capability is meant to work and does not -- a regression
#   2  BLOCKED  a prerequisite is missing; this part has not been built yet
#   3  SKIP     deliberately not applicable in this environment
#
# BLOCKED is the important one: it lets the harness track long-horizon progress.
# A part is "done" when its check flips BLOCKED -> PASS and stays there.
#
# Usage:
#   run_checks.sh                 # run everything
#   run_checks.sh 20 30           # run only checks whose name starts with 20 or 30
#   FAIL_ON_BLOCKED=1 run_checks.sh   # treat BLOCKED as failure (for release gating)

set -uo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKS_DIR="${HARNESS_DIR}/checks"
REPORTS_DIR="${HARNESS_DIR}/reports"
mkdir -p "${REPORTS_DIR}"

# Checks may boot Isaac Sim or compile CUDA. The box has no discrete VRAM -- GPU
# allocations come out of the same 128 GB as the OS. See docs/GB10_MEMORY_BUDGET.md.
MIN_AVAILABLE_GB="${MIN_AVAILABLE_GB:-20}"
available_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if (( available_gb < MIN_AVAILABLE_GB )); then
    echo "ABORT: only ${available_gb} GB available, need >= ${MIN_AVAILABLE_GB} GB." >&2
    echo "       Stop resident LLM servers first: docker stop \$(docker ps -q --filter name=vllm)" >&2
    exit 3
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
report="${REPORTS_DIR}/${stamp}.json"
log_dir="${REPORTS_DIR}/${stamp}"
mkdir -p "${log_dir}"

mapfile -t all_checks < <(find "${CHECKS_DIR}" -maxdepth 1 -type f -executable -printf '%f\n' | sort)

selected=()
if [[ $# -gt 0 ]]; then
    for check in "${all_checks[@]}"; do
        for prefix in "$@"; do
            [[ "${check}" == "${prefix}"* ]] && selected+=("${check}") && break
        done
    done
else
    selected=("${all_checks[@]}")
fi

if [[ ${#selected[@]} -eq 0 ]]; then
    echo "no checks matched" >&2
    exit 3
fi

pass=0; fail=0; blocked=0; skip=0
results_json=""

printf '%-34s %-9s %s\n' "CHECK" "RESULT" "DETAIL"
printf '%s\n' "-------------------------------------------------------------------------"

for check in "${selected[@]}"; do
    check_log="${log_dir}/${check}.log"
    started=$(date -u +%s)
    timeout --signal=TERM --kill-after=30s "${CHECK_TIMEOUT:-1800}" \
        "${CHECKS_DIR}/${check}" >"${check_log}" 2>&1
    code=$?
    elapsed=$(( $(date -u +%s) - started ))

    case ${code} in
        0) status="PASS";    ((pass++)) ;;
        1) status="FAIL";    ((fail++)) ;;
        2) status="BLOCKED"; ((blocked++)) ;;
        3) status="SKIP";    ((skip++)) ;;
        124|137) status="TIMEOUT"; ((fail++)) ;;
        *) status="FAIL";    ((fail++)) ;;
    esac

    # Last non-empty line of the check's output is its one-line summary.
    detail=$(grep -v '^[[:space:]]*$' "${check_log}" | tail -1 | cut -c1-90)
    printf '%-34s %-9s %s\n' "${check}" "${status}" "${detail}"

    escaped_detail=$(printf '%s' "${detail}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    results_json+="{\"check\":\"${check}\",\"status\":\"${status}\",\"exit_code\":${code},\"seconds\":${elapsed},\"detail\":${escaped_detail},\"log\":\"${check_log}\"},"
done

results_json="[${results_json%,}]"
cat > "${report}" <<EOF
{
  "timestamp": "${stamp}",
  "available_gb_at_start": ${available_gb},
  "summary": {"pass": ${pass}, "fail": ${fail}, "blocked": ${blocked}, "skip": ${skip}},
  "results": ${results_json}
}
EOF

printf '%s\n' "-------------------------------------------------------------------------"
echo "PASS ${pass}  FAIL ${fail}  BLOCKED ${blocked}  SKIP ${skip}"
echo "report: ${report}"
echo "logs:   ${log_dir}/"

ln -sfn "${report}" "${REPORTS_DIR}/latest.json"

(( fail > 0 )) && exit 1
if [[ "${FAIL_ON_BLOCKED:-0}" == "1" ]] && (( blocked > 0 )); then
    exit 1
fi
exit 0
