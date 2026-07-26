"""Strict parser for confirmed Stray Scanner capture bundles."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from simbiote.mapper.models import (
    CameraIntrinsics,
    CaptureBundle,
    Frame,
    ImuSample,
    Pose,
)


class CaptureValidationError(ValueError):
    """Raised when a capture bundle cannot be safely matched frame-by-frame."""


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CaptureValidationError(f"{path.name} has no header")
        return list(reader)


def _pick(row: dict[str, str], *names: str) -> str:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name.lower())
        if value not in (None, ""):
            return value
    raise CaptureValidationError(f"Missing required column; expected one of {names}")


def _number(row: dict[str, str], *names: str) -> float:
    try:
        value = float(_pick(row, *names))
    except ValueError as exc:
        raise CaptureValidationError(f"Invalid numeric value in column {names[0]}") from exc
    if not math.isfinite(value):
        raise CaptureValidationError(f"Non-finite value in column {names[0]}")
    return value


def _read_intrinsic_matrix(path: Path) -> tuple[float, ...]:
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            for item in row:
                item = item.strip()
                if item:
                    try:
                        values.append(float(item))
                    except ValueError:
                        continue
    if len(values) < 9:
        raise CaptureValidationError("camera_matrix.csv must contain at least 9 numbers")
    return tuple(values[-9:])


def load_capture_bundle(
    path: str | Path, *, allow_metadata_only: bool = False
) -> CaptureBundle:
    root = Path(path).expanduser().resolve()
    metadata_files = [
        root / "camera_matrix.csv",
        root / "odometry.csv",
        root / "imu.csv",
    ]
    missing_metadata = [item.name for item in metadata_files if not item.exists()]
    if missing_metadata:
        raise CaptureValidationError(
            f"Capture is missing required metadata: {', '.join(missing_metadata)}"
        )

    media_files = [
        root / "depth",
        root / "confidence",
        root / "rgb.mp4",
    ]
    missing_media = [item.name for item in media_files if not item.exists()]
    has_media = not missing_media
    if missing_media and not allow_metadata_only:
        raise CaptureValidationError(
            "Capture is missing media required for reconstruction: "
            f"{', '.join(missing_media)}. Export the full Stray Scanner bundle, "
            "or use --allow-metadata-only only for proxy contract testing."
        )

    frames: list[Frame] = []
    warnings: list[str] = []
    if missing_media:
        warnings.append(
            "Metadata-only capture: missing "
            f"{', '.join(missing_media)}; no reconstruction or semantic labeling is possible."
        )
    seen: set[int] = set()
    for row in _rows(root / "odometry.csv"):
        frame_id = int(_number(row, "frame", "frame_id", "index"))
        if frame_id in seen:
            raise CaptureValidationError(f"Duplicate odometry frame {frame_id}")
        seen.add(frame_id)
        depth = root / "depth" / f"{frame_id:06d}.png"
        confidence = root / "confidence" / f"{frame_id:06d}.png"
        if has_media and (not depth.exists() or not confidence.exists()):
            warnings.append(f"Dropped frame {frame_id}: depth/confidence pair is incomplete")
            continue

        frames.append(
            Frame(
                frame_id=frame_id,
                timestamp=_number(row, "timestamp", "time"),
                pose=Pose(
                    x=_number(row, "x"),
                    y=_number(row, "y"),
                    z=_number(row, "z"),
                    qx=_number(row, "qx"),
                    qy=_number(row, "qy"),
                    qz=_number(row, "qz"),
                    qw=_number(row, "qw"),
                ),
                intrinsics=CameraIntrinsics(
                    fx=_number(row, "fx"),
                    fy=_number(row, "fy"),
                    cx=_number(row, "cx"),
                    cy=_number(row, "cy"),
                ),
                depth_path=depth,
                confidence_path=confidence,
            )
        )

    if not frames:
        raise CaptureValidationError("No usable odometry frames found")
    frames.sort(key=lambda item: item.timestamp)

    imu: list[ImuSample] = []
    for row in _rows(root / "imu.csv"):
        imu.append(
            ImuSample(
                timestamp=_number(row, "timestamp", "time"),
                acceleration=(
                    _number(row, "a_x", "ax"),
                    _number(row, "a_y", "ay"),
                    _number(row, "a_z", "az"),
                ),
                angular_velocity=(
                    _number(row, "alpha_x", "gyro_x", "gx"),
                    _number(row, "alpha_y", "gyro_y", "gy"),
                    _number(row, "alpha_z", "gyro_z", "gz"),
                ),
            )
        )
    imu.sort(key=lambda item: item.timestamp)

    if len(imu) < len(frames):
        warnings.append("IMU sample count is lower than retained RGB frame count")

    return CaptureBundle(
        root=root,
        video_path=root / "rgb.mp4",
        static_intrinsics=_read_intrinsic_matrix(root / "camera_matrix.csv"),
        frames=frames,
        imu_samples=imu,
        has_media=has_media,
        warnings=warnings,
    )
