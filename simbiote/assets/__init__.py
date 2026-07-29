"""Data files shipped with the package, and the one place their root is resolved.

These are *inside* `simbiote/` rather than at the repository root on purpose:
`robot_config`, `wheelchair_task`, `scene_query` and `wilor_backend` all load
them at runtime, so a wheel that omitted them would import fine and then fail
at the first `loadURDF`. Keeping them under the package makes them ordinary
package data (see `[tool.setuptools.package-data]` in `pyproject.toml`).

Layout:

    robots/       stand-in mobile-manipulator URDF for laptop-tier physics
    wheelchair/   the stretch-goal wheelchair URDF
    scenes/       hand-written scene graphs used until a real scan exists
    mano/         (not in git) chumpy-free MANO pickle for the WiLoR backend,
                  written by `scripts/gb10/teleop/dechumpify_mano.py`
"""

from __future__ import annotations

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent

ROBOTS_DIR = ASSETS_DIR / "robots"
SCENES_DIR = ASSETS_DIR / "scenes"
WHEELCHAIR_DIR = ASSETS_DIR / "wheelchair"
MANO_DIR = ASSETS_DIR / "mano"

STAND_IN_ROBOT_URDF = ROBOTS_DIR / "stand_in_robot.urdf"
WHEELCHAIR_URDF = WHEELCHAIR_DIR / "wheelchair.urdf"
HOSPITAL_SCENE_GRAPH = SCENES_DIR / "hospital_scene_graph.json"

__all__ = [
    "ASSETS_DIR",
    "HOSPITAL_SCENE_GRAPH",
    "MANO_DIR",
    "ROBOTS_DIR",
    "SCENES_DIR",
    "STAND_IN_ROBOT_URDF",
    "WHEELCHAIR_DIR",
    "WHEELCHAIR_URDF",
]
