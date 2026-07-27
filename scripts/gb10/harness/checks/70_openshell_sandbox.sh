#!/usr/bin/env bash
# Part 6 (event stack): OpenShell sandbox is installed and running.
# Placeholder contract -- the OpenClaw/NemoClaw/OpenShell agent owns tightening this
# into a real policy assertion (Landlock ABI level, egress denial, write scoping).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

binary="/opt/openshell/bin/openshell-sandbox"
[[ -x "${binary}" ]] || blocked "openshell-sandbox not installed at ${binary}"

if ! pgrep -f openshell-sandbox >/dev/null 2>&1; then
    fail "openshell-sandbox installed but not running"
fi

version=$("${binary}" --version 2>&1 | head -1 || echo "unknown")
landlock_abi=$(grep -c . /sys/kernel/security/landlock/* 2>/dev/null | head -1 || echo "")

pass "openshell-sandbox running (${version})${landlock_abi:+, landlock present}"
