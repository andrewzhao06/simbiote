from __future__ import annotations

import pytest

from factoryflow.agentic.llm_backend import FakeBackend
from factoryflow.agentic.robot_tools import RobotTools, StubBackend
from factoryflow.agentic.scene_query import SceneGraph, load_scene


@pytest.fixture
def scene() -> SceneGraph:
    return load_scene()


@pytest.fixture
def fake_llm(scene: SceneGraph) -> FakeBackend:
    return FakeBackend(scene)


@pytest.fixture
def tools(scene: SceneGraph) -> RobotTools:
    return RobotTools(scene, StubBackend())


@pytest.fixture
def stage(tmp_path):
    """An isolated staging dir, so tests never touch /var/factoryflow/stage."""
    return tmp_path / "stage"
