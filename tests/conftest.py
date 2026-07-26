"""Shared test fixtures.

`pybullet` has no Windows wheels on PyPI (see README's "Known issues" /
requirements.txt) -- tests that need real physics use `require_pybullet` so
the suite reports SKIPPED (not an error, not a silent pass) on machines
where it isn't installed, and runs for real on Linux/macOS/the GB10.
"""

from __future__ import annotations

import pytest


def _has_pybullet() -> bool:
    try:
        import pybullet  # noqa: F401
    except ImportError:
        return False
    return True


HAS_PYBULLET = _has_pybullet()

require_pybullet = pytest.mark.skipif(
    not HAS_PYBULLET,
    reason="pybullet not installed (no Windows wheel on PyPI -- see README/requirements.txt)",
)
