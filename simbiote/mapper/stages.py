"""Mapper stages and production-tool adapter boundary."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from simbiote.mapper.config import MapperConfig
from simbiote.mapper.models import BoundingBox, CaptureBundle, SceneGraph, SceneNode


class StageError(RuntimeError):
    """A pipeline stage failed or did not produce its contract artifact."""


@dataclass(frozen=True, slots=True)
class RefinedCapture:
    capture: CaptureBundle
    artifact_dir: Path
    trajectory_bounds: BoundingBox


@dataclass(frozen=True, slots=True)
class DepthCompletedCapture:
    refined: RefinedCapture
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class ReconstructedScene:
    depth: DepthCompletedCapture
    artifact_dir: Path
    geometry_path: Path | None


def _trajectory_bounds(capture: CaptureBundle) -> BoundingBox:
    xyz = [(frame.pose.x, frame.pose.y, frame.pose.z) for frame in capture.frames]
    minimum = tuple(min(values[index] for values in xyz) for index in range(3))
    maximum = tuple(max(values[index] for values in xyz) for index in range(3))
    size = tuple(max(maximum[index] - minimum[index], 0.5) for index in range(3))
    center = tuple((minimum[index] + maximum[index]) / 2 for index in range(3))
    return BoundingBox(center=center, size=size)


def _run_adapter(env_name: str, variables: dict[str, Path], cwd: Path) -> None:
    template = os.getenv(env_name)
    if not template:
        raise StageError(
            f"Production mode requires {env_name}; see config/mapper.example.toml"
        )
    command = template.format(**{key: str(value) for key, value in variables.items()})
    result = subprocess.run(shlex.split(command), cwd=cwd, check=False)
    if result.returncode:
        raise StageError(f"{env_name} command failed with exit code {result.returncode}")


def _rotate(
    point: tuple[float, float, float], quaternion: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    px, py, pz = point
    qx, qy, qz, qw = quaternion
    dot = qx * px + qy * py + qz * pz
    cross = (qy * pz - qz * py, qz * px - qx * pz, qx * py - qy * px)
    q_norm = qx * qx + qy * qy + qz * qz + qw * qw
    if q_norm == 0:
        raise StageError("Odometry contains a zero-length quaternion")
    scale = 2 / q_norm
    return (
        px + scale * (qw * cross[0] + qx * dot - q_norm * px),
        py + scale * (qw * cross[1] + qy * dot - q_norm * py),
        pz + scale * (qw * cross[2] + qz * dot - q_norm * pz),
    )


def _preview_points(
    capture: CaptureBundle, config: MapperConfig
) -> list[tuple[float, float, float]]:
    import numpy as np
    from PIL import Image

    max_frames = 12
    stride = max(math.ceil(len(capture.frames) / max_frames), 1)
    points: list[tuple[float, float, float]] = []
    for frame in capture.frames[::stride]:
        depth = np.asarray(Image.open(frame.depth_path), dtype=np.float32)
        confidence = np.asarray(Image.open(frame.confidence_path))
        if depth.ndim != 2 or confidence.shape != depth.shape:
            raise StageError(f"Invalid depth/confidence dimensions for frame {frame.frame_id}")
        valid = (depth > 0) & (confidence >= config.confidence_threshold)
        for row in range(0, depth.shape[0], 8):
            for column in range(0, depth.shape[1], 8):
                if not valid[row, column]:
                    continue
                distance = float(depth[row, column]) * config.depth_scale_meters
                local = (
                    (column - frame.intrinsics.cx) / frame.intrinsics.fx * distance,
                    (row - frame.intrinsics.cy) / frame.intrinsics.fy * distance,
                    distance,
                )
                rotated = _rotate(
                    local,
                    (
                        frame.pose.qx,
                        frame.pose.qy,
                        frame.pose.qz,
                        frame.pose.qw,
                    ),
                )
                points.append(
                    (
                        rotated[0] + frame.pose.x,
                        rotated[1] + frame.pose.y,
                        rotated[2] + frame.pose.z,
                    )
                )
    if not points:
        raise StageError("No confidence-filtered LiDAR points available for local preview")
    return points


def _write_preview_geometry(
    capture: CaptureBundle, config: MapperConfig, output: Path
) -> Path:
    points = _preview_points(capture, config)
    serialized = ",\n        ".join(
        f"({x:.8g}, {y:.8g}, {z:.8g})" for x, y, z in points
    )
    geometry = output / "lidar_preview.usda"
    geometry.write_text(
        f"""#usda 1.0
(
    defaultPrim = "LiDARPreview"
    metersPerUnit = 1
    upAxis = "Y"
)

def Points "LiDARPreview"
{{
    point3f[] points = [
        {serialized}
    ]
    float[] widths = [{", ".join("0.015" for _ in points)}]
}}
""",
        encoding="utf-8",
    )
    return geometry


def refine_poses(
    capture: CaptureBundle, config: MapperConfig, artifact_dir: Path
) -> RefinedCapture:
    output = artifact_dir / "01_colmap"
    output.mkdir(parents=True, exist_ok=True)
    bounds = _trajectory_bounds(capture)
    manifest = {
        "mode": config.mode,
        "capture": str(capture.root),
        "frame_count": len(capture.frames),
        "trajectory_bounds": asdict(bounds),
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if config.mode == "production":
        _run_adapter(
            "SIMBIOTE_COLMAP_COMMAND",
            {"capture": capture.root, "output": output},
            output,
        )
    return RefinedCapture(capture=capture, artifact_dir=output, trajectory_bounds=bounds)


def complete_depth(
    refined: RefinedCapture, config: MapperConfig, artifact_dir: Path
) -> DepthCompletedCapture:
    output = artifact_dir / "02_depth"
    output.mkdir(parents=True, exist_ok=True)
    if config.mode == "production":
        _run_adapter(
            "SIMBIOTE_DEPTH_COMMAND",
            {
                "capture": refined.capture.root,
                "colmap": refined.artifact_dir,
                "checkpoint": config.depth_checkpoint,
                "output": output,
            },
            output,
        )
        if not any(output.iterdir()):
            raise StageError("Depth adapter produced no completed-depth artifacts")
    else:
        (output / "PROXY_MODE.txt").write_text(
            "Original LiDAR depth is passed through in proxy mode.\n", encoding="utf-8"
        )
    return DepthCompletedCapture(refined=refined, artifact_dir=output)


def reconstruct(
    depth: DepthCompletedCapture, config: MapperConfig, artifact_dir: Path
) -> ReconstructedScene:
    output = artifact_dir / "03_reconstruction"
    output.mkdir(parents=True, exist_ok=True)
    geometry: Path | None = None
    if config.mode == "production":
        _run_adapter(
            "SIMBIOTE_DGRUT_COMMAND",
            {
                "capture": depth.refined.capture.root,
                "colmap": depth.refined.artifact_dir,
                "depth": depth.artifact_dir,
                "output": output,
                "dgrut": config.dgrut_root,
            },
            output,
        )
        candidates = (
            list(output.glob("*.usd"))
            + list(output.glob("*.usda"))
            + list(output.glob("*.usdc"))
            + list(output.glob("*.usdz"))
        )
        if not candidates:
            raise StageError(
                "3DGRUT adapter must export a .usd/.usda/.usdc/.usdz scene layer"
            )
        geometry = candidates[0]
    elif config.mode == "preview":
        geometry = _write_preview_geometry(depth.refined.capture, config, output)
    else:
        (output / "PROXY_MODE.txt").write_text(
            "No Gaussian optimization was run. USD will contain contract-test proxies.\n",
            encoding="utf-8",
        )
    return ReconstructedScene(depth=depth, artifact_dir=output, geometry_path=geometry)


def label_scene(
    scene: ReconstructedScene, config: MapperConfig, artifact_dir: Path
) -> list[SceneNode]:
    output = artifact_dir / "04_semantic"
    output.mkdir(parents=True, exist_ok=True)
    detections_path = output / "detections.json"
    if config.mode == "production":
        _run_adapter(
            "SIMBIOTE_SAM3_COMMAND",
            {
                "capture": scene.depth.refined.capture.root,
                "geometry": scene.geometry_path or Path(""),
                "checkpoint": config.sam3_checkpoint,
                "output": output,
            },
            output,
        )
        if not detections_path.exists():
            raise StageError("SAM 3 adapter must produce detections.json")
        raw = json.loads(detections_path.read_text(encoding="utf-8"))
        return [
            SceneNode(
                node_id=str(item["node_id"]),
                label=str(item["label"]),
                kind=str(item["kind"]),
                confidence=float(item.get("confidence", 1.0)),
                bounds=BoundingBox(
                    center=tuple(float(v) for v in item["bounds"]["center"]),
                    size=tuple(float(v) for v in item["bounds"]["size"]),
                ),
                source_frame_ids=[int(v) for v in item.get("source_frame_ids", [])],
            )
            for item in raw["nodes"]
        ]

    bounds = scene.depth.refined.trajectory_bounds
    floor_size = (max(bounds.size[0] + 2.0, 3.0), 0.05, max(bounds.size[2] + 2.0, 3.0))
    nodes = [
        SceneNode(
            node_id="navigable_floor",
            label="navigable floor",
            kind="floor",
            confidence=1.0,
            bounds=BoundingBox(
                center=(bounds.center[0], bounds.center[1] - 1.0, bounds.center[2]),
                size=floor_size,
            ),
        ),
        SceneNode(
            node_id="proxy_graspable",
            label="proxy graspable object",
            kind="object",
            confidence=1.0,
            bounds=BoundingBox(
                center=(bounds.center[0] + 0.5, bounds.center[1] - 0.7, bounds.center[2]),
                size=(0.25, 0.25, 0.25),
            ),
        ),
    ]
    detections_path.write_text(
        json.dumps(
            {"proxy": config.mode == "proxy", "nodes": [asdict(node) for node in nodes]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return nodes


def build_scene_graph(
    nodes: list[SceneNode], scene: ReconstructedScene, artifact_dir: Path
) -> SceneGraph:
    for node in nodes:
        if node.kind == "floor":
            node.attributes.update(
                {
                    "is_navigable": True,
                    "collision_enabled": True,
                    "friction_static": 0.9,
                    "friction_dynamic": 0.8,
                }
            )
        elif node.kind == "object":
            node.attributes.update(
                {
                    "is_graspable": True,
                    "grasp_type": "parallel_jaw",
                    "mass_kg": 0.25,
                    "collision_enabled": True,
                    "friction_static": 0.6,
                    "friction_dynamic": 0.5,
                }
            )
    graph = SceneGraph(
        nodes=nodes,
        metadata={
            "source_capture": str(scene.depth.refined.capture.root),
            "geometry_path": str(scene.geometry_path) if scene.geometry_path else None,
            "proxy": scene.geometry_path is None,
        },
    )
    output = artifact_dir / "05_scenegraph"
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene_graph.json").write_text(
        json.dumps(graph.to_dict(), indent=2), encoding="utf-8"
    )
    return graph
