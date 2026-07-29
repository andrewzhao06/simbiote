#!/usr/bin/env bash
# Part 5 (handoff): a mapper-produced stage opens in Isaac Sim and meets the Step 2
# contract -- navigable floor, graspable object, colliders, metres.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

validator="${REPO_ROOT}/scripts/gb10/mapper/validate_usd_isaac.sh"
[[ -x "${validator}" ]] || blocked "Isaac validator missing: ${validator}"

# Prefer the newest real (non-proxy) stage; fall back to the proxy scene.
# *.geometry.usda is the point-cloud sidecar the scene references, not a stage
# of its own -- it carries no floor or graspable prim and would always fail the
# contract. Skip it or the newest-first walk lands on it and reports a bogus
# regression whenever a real reconstruction is published here.
scene=""
for candidate in $(ls -t "${STAGE_ROOT}"/*.usda "${STAGE_ROOT}"/*.usd 2>/dev/null); do
    case "${candidate}" in
        *.geometry.usda | *.geometry.usd) continue ;;
    esac
    scene="${candidate}"
    grep -q 'factoryflow:proxy = true' "${candidate}" 2>/dev/null || break
done

[[ -n "${scene}" ]] || blocked "no mapper stage found under ${STAGE_ROOT}"

out=$("${validator}" "${scene}" 2>&1)
code=$?

summary=$(printf '%s\n' "${out}" | grep -E '^(navigable floor|graspable objects)' | tr '\n' ' ')

case ${code} in
    0) pass "$(basename "${scene}"): ${summary}" ;;
    1) fail "$(basename "${scene}") violates the Step 2 contract -- $(printf '%s\n' "${out}" | grep '^ERROR' | tr '\n' ';')" ;;
    3) skip "Isaac validation deferred: memory preflight refused" ;;
    *) fail "Isaac validator crashed (exit ${code})" ;;
esac
