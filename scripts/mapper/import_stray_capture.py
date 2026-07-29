"""Copy a Stray Scanner export into the obvious phone-scan upload folder."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REQUIRED_METADATA = ("camera_matrix.csv", "odometry.csv", "imu.csv")
REQUIRED_MEDIA = ("rgb.mp4", "depth", "confidence")
REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
DEFAULT_DESTINATION = REPO_ROOT / "UPLOAD_PHONE_SCANS_HERE"
#: The prepared SSD's capture folder. Taken from the same environment variable
#: the mapper config uses (`FACTORYFLOW_SSD_ROOT`, pointing at `<SSD>/AI`), so
#: `--ssd` works on the GB10 rather than only on the one Windows box whose
#: drive letter used to be hardcoded here.
SSD_ROOT_ENV = "FACTORYFLOW_SSD_ROOT"


def ssd_destination() -> Path:
    root = os.getenv(SSD_ROOT_ENV)
    if not root:
        raise ValueError(
            f"--ssd needs {SSD_ROOT_ENV} set to the SSD's AI/ directory "
            "(source config/mapper.gb10.env), or pass --ssd-destination."
        )
    return Path(root).expanduser() / "captures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a Stray Scanner export into UPLOAD_PHONE_SCANS_HERE "
            f"(optional --ssd also copies to ${SSD_ROOT_ENV}/captures)."
        )
    )
    parser.add_argument("source", type=Path, help="Folder exported by Stray Scanner")
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Upload folder (default: ./UPLOAD_PHONE_SCANS_HERE)",
    )
    parser.add_argument(
        "--name",
        help="Name for the copied capture; defaults to the source folder name",
    )
    parser.add_argument(
        "--ssd",
        action="store_true",
        help=f"Also copy the capture to ${SSD_ROOT_ENV}/captures",
    )
    parser.add_argument(
        "--ssd-destination",
        type=Path,
        help=f"Explicit SSD capture folder, overriding ${SSD_ROOT_ENV}",
    )
    parser.add_argument(
        "--allow-metadata-only",
        action="store_true",
        help="Permit a metadata-only export for proxy testing",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not name:
        raise ValueError("Capture name must contain at least one letter or number")
    return name


def validate_source(source: Path, *, allow_metadata_only: bool) -> None:
    if not source.is_dir():
        raise ValueError(f"Stray Scanner export folder does not exist: {source}")
    missing = [name for name in REQUIRED_METADATA if not (source / name).is_file()]
    if not allow_metadata_only:
        missing.extend(name for name in REQUIRED_MEDIA if not (source / name).exists())
    if missing:
        mode = "metadata-only " if allow_metadata_only else ""
        raise ValueError(
            f"Not a valid {mode}Stray Scanner export; missing: {', '.join(missing)}"
        )


def copy_capture(source: Path, destination: Path, name: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    if target.exists():
        raise FileExistsError(f"Capture already exists: {target}")
    shutil.copytree(source, target)
    return target


def main() -> int:
    args = parse_args()
    try:
        source = args.source.expanduser().resolve()
        validate_source(source, allow_metadata_only=args.allow_metadata_only)
        name = safe_name(args.name or source.name)
        # Resolve the SSD target before the first copy, so a misconfigured
        # --ssd fails before half the work is done rather than after.
        ssd_target_dir = (
            (args.ssd_destination.expanduser() if args.ssd_destination else ssd_destination())
            if args.ssd
            else None
        )

        target = copy_capture(source, args.destination.expanduser().resolve(), name)
        print(f"Uploaded capture to: {target}")
        print(f'Validate it with: python -m simbiote.mapper.cli ingest "{target}"')

        if ssd_target_dir is not None:
            print(f"Also copied to SSD: {copy_capture(source, ssd_target_dir, name)}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
