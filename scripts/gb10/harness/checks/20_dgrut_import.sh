#!/usr/bin/env bash
# Part 1 (core env): the threedgrut package itself imports from the venv.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_dgrut_python

out=$(cd "${DGRUT_ROOT}" && "${DGRUT_PYTHON}" - <<'PY' 2>&1
import sys
try:
    import threedgrut
except Exception as exc:
    print(f"IMPORT_ERROR {type(exc).__name__}: {exc}")
    sys.exit(9)
print(f"OK threedgrut from {threedgrut.__file__}")
PY
)
code=$?

case ${code} in
    0) pass "${out}" ;;
    9) blocked "threedgrut not importable -- ${out}" ;;
    *) fail "threedgrut import errored -- ${out}" ;;
esac
