"""Typed contracts shared by every mapper stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True, slots=True)
class Pose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True, slots=True)
class Frame:
    frame_id: int
    timestamp: float
    pose: Pose
    intrinsics: CameraIntrinsics
    depth_path: Path
    confidence_path: Path


@dataclass(frozen=True, slots=True)
class ImuSample:
    timestamp: float
    acceleration: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]


@dataclass(slots=True)
class CaptureBundle:
    root: Path
    video_path: Path
    static_intrinsics: tuple[float, ...]
    frames: list[Frame]
    imu_samples: list[ImuSample]
    has_media: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(slots=True)
class SceneNode:
    node_id: str
    label: str
    kind: str
    bounds: BoundingBox
    confidence: float = 1.0
    source_frame_ids: list[int] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneEdge:
    source: str
    relation: str
    target: str


@dataclass(slots=True)
class SceneGraph:
    schema_version: str = "1.0"
    coordinate_system: str = "Y_UP_RIGHT_HANDED_METERS"
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
