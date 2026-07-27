"""Copy a Stray Scanner export into the obvious phone-scan upload folder."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REQUIRED_METADATA = ("camera_matrix.csv", "odometry.csv", "imu.csv")
REQUIRED_MEDIA = ("rgb.mp4", "depth", "confidence")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPO_ROOT / "UPLOAD_PHONE_SCANS_HERE"
SSD_DESTINATION = Path(r"D:\AI\captures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a Stray Scanner export into UPLOAD_PHONE_SCANS_HERE "
            "(optional --ssd also copies to D:\\AI\\captures)."
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
        help=f"Also copy the capture to {SSD_DESTINATION}",
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
    source = args.source.expanduser().resolve()
    validate_source(source, allow_metadata_only=args.allow_metadata_only)
    name = safe_name(args.name or source.name)

    target = copy_capture(source, args.destination.expanduser().resolve(), name)
    print(f"Uploaded capture to: {target}")
    print(f'Validate it with: python -m src.cli ingest "{target}"')

    if args.ssd:
        ssd_target = copy_capture(source, SSD_DESTINATION, name)
        print(f"Also copied to SSD: {ssd_target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
