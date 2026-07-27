"""`spawn_robot(spec)` — the OpenClaw tool call Part 1's orchestration table
attributes to Step 2 (`| Spawn robot | spawn_robot(spec) | Loads the 4-wheel
+ arm reference robot into the scene | Part 5 (Step 2) |`) but which §5.4's
build-spec file tree didn't previously list a home for. This is that home.

Today: loads the PyBullet stand-in into a fresh scene. Tomorrow: swap the
body of `_spawn_pybullet` for an Isaac Sim stage-loading call (`Create >
Robots > Wheeled Robots > Clearpath > Ridgeback Franka`, or the scripted
`omni.isaac.core` equivalent) behind the same `spawn_robot()` signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simbiote.robot.robot_config import RIDGEBACK_FRANKA_CONFIG, STAND_IN_CONFIG, RobotConfig, get_config
from simbiote.sim_env import pybullet_scene as scene


@dataclass
class SpawnedRobot:
    """Handle returned to the OpenClaw orchestrator / other tasks."""

    robot_id: int
    physics_client: int
    config: RobotConfig
    handles: "scene.RobotHandles"


def spawn_robot(spec: str | RobotConfig = "stand_in", gui: bool = False, fixed_base: bool = False) -> SpawnedRobot:
    """OpenClaw tool: `spawn_robot(spec)`.

    `spec` is either a `RobotConfig` (e.g. `RIDGEBACK_FRANKA_CONFIG`) or a
    registry name string (`"stand_in"` / `"ridgeback_franka"`, see
    `robot_config.get_config`). Loads the 4-wheel + arm reference robot into
    a fresh scene and returns a handle other tools/tasks can use.
    """
    config = spec if isinstance(spec, RobotConfig) else get_config(spec)

    if config.engine != "pybullet":
        raise NotImplementedError(
            f"spawn_robot: engine '{config.engine}' isn't wired up yet in this laptop-tier build "
            "-- tomorrow's Isaac Sim swap goes here (see module docstring)."
        )

    client = scene.connect(gui=gui)
    scene.load_ground_plane(client)
    robot_id = scene.load_robot(config, client, fixed_base=fixed_base)
    handles = scene.RobotHandles.build(config, client, robot_id)
    return SpawnedRobot(robot_id=robot_id, physics_client=client, config=config, handles=handles)


def despawn(robot: SpawnedRobot) -> None:
    scene.disconnect(robot.physics_client)
