#!/usr/bin/env bash
# Part 4 (integration): the mapper's own preflight passes end to end.
# This is the gate on running the pipeline in production mode.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

preflight="${REPO_ROOT}/scripts/gb10/preflight_mapper.sh"
[[ -x "${preflight}" ]] || blocked "preflight script missing: ${preflight}"

out=$("${preflight}" "${AI_ROOT}" "${REPO_ROOT}" 2>&1)
code=$?

failures=$(printf '%s\n' "${out}" | grep -c '^FAIL' || true)

if (( code == 0 )); then
    passes=$(printf '%s\n' "${out}" | grep -c '^PASS' || true)
    pass "mapper preflight: ${passes} checks passed, 0 failed"
fi

detail=$(printf '%s\n' "${out}" | grep '^FAIL' | tr '\n' ';' | cut -c1-200)
blocked "mapper preflight: ${failures} check(s) outstanding -- ${detail}"
