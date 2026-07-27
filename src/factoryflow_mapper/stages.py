"""Mapper stages and production-tool adapter boundary."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from factoryflow_mapper.config import MapperConfig
from factoryflow_mapper.models import (
    BoundingBox,
    CaptureBundle,
    Frame,
    SceneGraph,
    SceneNode,
)


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
    # Measured from the reconstructed geometry when there is any. Proxy runs
    # leave these None and fall back to the trajectory heuristic.
    scene_bounds: BoundingBox | None = None
    floor_bounds: BoundingBox | None = None


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
    """Rotate a point by a quaternion: p' = p + 2*qw*(qv X p) + 2*qv X (qv X p).

    The vector triple product expands to `qv*(qv.p) - p*|qv|^2`, so the term
    subtracted from `p` is scaled by the norm of the *vector part* only. Using
    the full quaternion norm there (which folds in qw^2) is not a rotation at
    all -- it negates the vector for the identity quaternion and does not
    preserve length. See Gagan/tests/test_stages.py.
    """
    px, py, pz = point
    qx, qy, qz, qw = quaternion
    norm = qx * qx + qy * qy + qz * qz + qw * qw
    if norm == 0:
        raise StageError("Odometry contains a zero-length quaternion")
    # Normalise so drift in the exported ARKit quaternions cannot rescale points.
    inverse = 1 / math.sqrt(norm)
    qx, qy, qz, qw = qx * inverse, qy * inverse, qz * inverse, qw * inverse

    dot = qx * px + qy * py + qz * pz
    cross = (qy * pz - qz * py, qz * px - qx * pz, qx * py - qy * px)
    vector_norm = qx * qx + qy * qy + qz * qz
    return (
        px + 2 * (qw * cross[0] + qx * dot - vector_norm * px),
        py + 2 * (qw * cross[1] + qy * dot - vector_norm * py),
        pz + 2 * (qw * cross[2] + qz * dot - vector_norm * pz),
    )


PREVIEW_MAX_FRAMES = 120
PREVIEW_PIXEL_STRIDE = 2
PREVIEW_VOXEL_METRES = 0.02


def unproject_frame(
    frame: Frame,
    config: MapperConfig,
    rgb_width: int,
    *,
    pixel_stride: int = 1,
    mask: "object | None" = None,
):
    """Unproject one frame's LiDAR depth map into the ARKit world frame.

    Shared by the preview reconstruction and the SAM 3 adapter so a semantic
    mask lands in exactly the same coordinates as the point cloud it labels.

    `mask` is an optional boolean array at *RGB* resolution (what SAM 3
    returns); it is point-sampled down onto the depth raster. Returns an
    (N, 3) float array of world-space points.
    """
    import numpy as np
    from PIL import Image

    depth = np.asarray(Image.open(frame.depth_path), dtype=np.float32)
    confidence = np.asarray(Image.open(frame.confidence_path))
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise StageError(f"Invalid depth/confidence dimensions for frame {frame.frame_id}")

    # odometry.csv reports intrinsics for the full-resolution RGB frame
    # (1920x1440 on a Pro-line iPhone), but the LiDAR depth maps are 256x192 --
    # the ~7.5x gap called out in SCAN_MAP.md 4.3. Unprojecting depth-map pixel
    # coordinates with RGB-scale focal lengths collapses the cloud into a
    # needle along the view axis, so rescale intrinsics to the depth raster.
    scale = depth.shape[1] / rgb_width
    fx = frame.intrinsics.fx * scale
    fy = frame.intrinsics.fy * scale
    cx = frame.intrinsics.cx * scale
    cy = frame.intrinsics.cy * scale

    step = max(int(pixel_stride), 1)
    depth = depth[::step, ::step] * config.depth_scale_meters
    confidence = confidence[::step, ::step]
    rows, columns = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]] * step

    valid = (depth > 0) & (confidence >= config.confidence_threshold)
    if mask is not None:
        selection = np.asarray(mask)
        if selection.ndim != 2:
            raise StageError("Semantic mask must be a 2-D boolean array")
        # Nearest-neighbour the RGB-resolution mask onto the depth raster.
        row_index = (
            np.clip(
                (rows * selection.shape[0]) // (depth.shape[0] * step), 0,
                selection.shape[0] - 1,
            )
        )
        column_index = (
            np.clip(
                (columns * selection.shape[1]) // (depth.shape[1] * step), 0,
                selection.shape[1] - 1,
            )
        )
        valid &= selection[row_index, column_index].astype(bool)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    distance = depth[valid]
    # Stray Scanner poses map an OpenCV-style camera frame (+X right,
    # +Y down, +Z forward) into the ARKit gravity-aligned world frame.
    local = np.stack(
        [
            (columns[valid] - cx) / fx * distance,
            (rows[valid] - cy) / fy * distance,
            distance,
        ],
        axis=1,
    )

    qx, qy, qz, qw = frame.pose.qx, frame.pose.qy, frame.pose.qz, frame.pose.qw
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise StageError("Odometry contains a zero-length quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    origin = np.array([frame.pose.x, frame.pose.y, frame.pose.z])
    return local @ rotation.T + origin


def rgb_width(capture: CaptureBundle, config: MapperConfig) -> int:
    """Width of the RGB frames the odometry intrinsics were measured against."""
    command = [
        str(config.ffprobe_binary),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "default=nw=1:nk=1",
        str(capture.video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    except OSError:
        pass
    # ffprobe is optional for preview mode. Phone principal points sit within a
    # pixel or two of the frame centre, so 2*cx recovers the width well enough
    # to keep the depth/RGB ratio right.
    return int(round(2 * capture.frames[0].intrinsics.cx))


def _preview_points(
    capture: CaptureBundle, config: MapperConfig
) -> "list[tuple[float, float, float]]":
    import numpy as np

    width = rgb_width(capture, config)
    stride = max(math.ceil(len(capture.frames) / PREVIEW_MAX_FRAMES), 1)
    chunks: list[object] = []
    for frame in capture.frames[::stride]:
        points = unproject_frame(
            frame, config, width, pixel_stride=PREVIEW_PIXEL_STRIDE
        )
        if len(points):
            chunks.append(points)

    if not chunks:
        raise StageError("No confidence-filtered LiDAR points available for local preview")

    cloud = np.concatenate(chunks)
    # Overlapping frames re-observe the same surfaces; collapsing to a voxel
    # grid keeps the cloud uniform and the exported .usda a sane size.
    _, unique = np.unique(
        np.floor(cloud / PREVIEW_VOXEL_METRES).astype(np.int64), axis=0, return_index=True
    )
    cloud = cloud[np.sort(unique)]
    return [tuple(float(value) for value in point) for point in cloud]


FLOOR_SLAB_METRES = 0.08


def _measure_cloud(
    points: "list[tuple[float, float, float]]",
) -> tuple[BoundingBox, BoundingBox | None]:
    """Scene bounds, plus the floor slab found inside them.

    The capture frame is gravity-aligned (ARKit, +Y up), so the floor is the
    densest horizontal slab in the lower half of the cloud -- ceilings and
    table tops are dense too, hence the restriction. Its X/Z extent is the
    navigable region Step 2's costmap is built from, so it is measured from
    the floor points themselves rather than from the whole scene.
    """
    import numpy as np

    cloud = np.asarray(points, dtype=np.float64)
    minimum, maximum = cloud.min(axis=0), cloud.max(axis=0)
    scene = BoundingBox(
        center=tuple((minimum + maximum) / 2),
        size=tuple(np.maximum(maximum - minimum, 0.5)),
    )

    heights = cloud[:, 1]
    midpoint = (heights.min() + heights.max()) / 2
    lower = heights[heights <= midpoint]
    if lower.size < 100:
        return scene, None

    bins = max(int((lower.max() - lower.min()) / FLOOR_SLAB_METRES), 1)
    counts, edges = np.histogram(lower, bins=bins)
    peak = int(counts.argmax())
    height = (edges[peak] + edges[peak + 1]) / 2

    on_floor = cloud[np.abs(heights - height) <= FLOOR_SLAB_METRES]
    if on_floor.shape[0] < 100:
        return scene, None
    floor_min, floor_max = on_floor.min(axis=0), on_floor.max(axis=0)
    floor = BoundingBox(
        center=(
            float((floor_min[0] + floor_max[0]) / 2),
            float(height),
            float((floor_min[2] + floor_max[2]) / 2),
        ),
        size=(
            float(max(floor_max[0] - floor_min[0], 0.5)),
            0.05,
            float(max(floor_max[2] - floor_min[2], 0.5)),
        ),
    )
    return scene, floor


def _write_preview_geometry(
    points: "list[tuple[float, float, float]]", output: Path
) -> Path:
    serialized = ",\n        ".join(
        f"({x:.5g}, {y:.5g}, {z:.5g})" for x, y, z in points
    )
    extent = [
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    ]
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
    float3[] extent = [({", ".join(f"{v:.5g}" for v in extent[0])}), \
({", ".join(f"{v:.5g}" for v in extent[1])})]
    point3f[] points = [
        {serialized}
    ]
    float[] widths = [0.02] (
        interpolation = "constant"
    )
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
            "FACTORYFLOW_COLMAP_COMMAND",
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
            "FACTORYFLOW_DEPTH_COMMAND",
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
    scene_bounds: BoundingBox | None = None
    floor_bounds: BoundingBox | None = None
    if config.mode == "production":
        _run_adapter(
            "FACTORYFLOW_DGRUT_COMMAND",
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
        points = _preview_points(depth.refined.capture, config)
        geometry = _write_preview_geometry(points, output)
        scene_bounds, floor_bounds = _measure_cloud(points)
        (output / "cloud_stats.json").write_text(
            json.dumps(
                {
                    "points": len(points),
                    "scene_bounds": asdict(scene_bounds),
                    "floor_bounds": asdict(floor_bounds) if floor_bounds else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        (output / "PROXY_MODE.txt").write_text(
            "No Gaussian optimization was run. USD will contain contract-test proxies.\n",
            encoding="utf-8",
        )
    return ReconstructedScene(
        depth=depth,
        artifact_dir=output,
        geometry_path=geometry,
        scene_bounds=scene_bounds,
        floor_bounds=floor_bounds,
    )


def label_scene(
    scene: ReconstructedScene, config: MapperConfig, artifact_dir: Path
) -> list[SceneNode]:
    output = artifact_dir / "04_semantic"
    output.mkdir(parents=True, exist_ok=True)
    detections_path = output / "detections.json"
    # Production always requires SAM 3 (_run_adapter raises if it is unset).
    # Preview may opt in: semantic labelling only needs posed RGB + depth, not
    # the COLMAP/3DGRUT reconstruction, so real objects are available on the
    # LiDAR preview path too.
    if config.mode == "production" or os.getenv("FACTORYFLOW_SAM3_COMMAND"):
        _run_adapter(
            "FACTORYFLOW_SAM3_COMMAND",
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

    if scene.floor_bounds is not None:
        # Preview/production: the floor was measured off the reconstructed
        # cloud, so it lines up with the geometry Step 2 navigates.
        floor_bounds = scene.floor_bounds
        confidence = 0.8
    else:
        # Proxy: no geometry exists, so fall back to the trajectory. A handheld
        # sweep is carried roughly a metre above the floor.
        bounds = scene.depth.refined.trajectory_bounds
        floor_bounds = BoundingBox(
            center=(bounds.center[0], bounds.center[1] - 1.0, bounds.center[2]),
            size=(
                max(bounds.size[0] + 2.0, 3.0),
                0.05,
                max(bounds.size[2] + 2.0, 3.0),
            ),
        )
        confidence = 1.0

    cube = 0.25
    nodes = [
        SceneNode(
            node_id="navigable_floor",
            label="navigable floor",
            kind="floor",
            confidence=confidence,
            bounds=floor_bounds,
        ),
        SceneNode(
            node_id="proxy_graspable",
            label="proxy graspable object",
            kind="object",
            confidence=1.0,
            bounds=BoundingBox(
                # Resting on the floor, not floating above it.
                center=(
                    floor_bounds.center[0] + 0.5,
                    floor_bounds.center[1] + floor_bounds.size[1] / 2 + cube / 2,
                    floor_bounds.center[2],
                ),
                size=(cube, cube, cube),
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
