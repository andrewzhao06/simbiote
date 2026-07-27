"""Shared test fixtures for the whole suite -- lives at the repo root so it's
visible to every domain under `tests/` (pytest's
conftest.py scoping cascades down from the rootdir to every subdirectory).

`pybullet` has no Windows wheels on PyPI (see README's "Known issues" /
requirements.txt) -- tests that need real physics use `require_pybullet` so
the suite reports SKIPPED (not an error, not a silent pass) on machines
where it isn't installed, and runs for real on Linux/macOS/the GB10.

`scene` / `fake_llm` / `tools` / `stage` are used across `tests/agentic/*`.
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


@pytest.fixture
def scene():
    from simbiote.agentic.scene_query import SceneGraph, load_scene

    return load_scene()


@pytest.fixture
def fake_llm(scene):
    from simbiote.agentic.llm_backend import FakeBackend

    return FakeBackend(scene)


@pytest.fixture
def tools(scene):
    from simbiote.agentic.robot_tools import RobotTools, StubBackend

    return RobotTools(scene, StubBackend())


@pytest.fixture
def stage(tmp_path):
    """An isolated staging dir, so tests never touch /var/simbiote/stage."""
    return tmp_path / "stage"
