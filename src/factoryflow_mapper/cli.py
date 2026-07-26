"""Command-line interface for FactoryFlow Mapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from factoryflow_mapper.config import MapperConfig
from factoryflow_mapper.doctor import run_doctor, serialize_checks
from factoryflow_mapper.ingest import load_capture_bundle
from factoryflow_mapper.pipeline import run_pipeline
from factoryflow_mapper.usd import validate_map


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="factoryflow-map")
    parser.add_argument("--config", type=Path, help="Mapper TOML config")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Check GB10 mapper dependencies")

    ingest = commands.add_parser("ingest", help="Validate a Stray Scanner capture")
    ingest.add_argument("capture", type=Path)
    ingest.add_argument(
        "--allow-metadata-only",
        action="store_true",
        help="Accept pose/IMU-only export for proxy testing; cannot reconstruct a scene",
    )

    run = commands.add_parser("run", help="Run scan-to-USD pipeline")
    run.add_argument("--capture", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--allow-proxy",
        action="store_true",
        help="Accept contract-test proxy geometry (never use for final handoff)",
    )

    validate = commands.add_parser("validate", help="Validate Step 2 USD contract")
    validate.add_argument("usd", type=Path)
    validate.add_argument("--allow-proxy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = MapperConfig.load(args.config)
    try:
        if args.command == "doctor":
            checks = run_doctor(config)
            print(json.dumps(serialize_checks(checks), indent=2))
            return int(any(check.required and not check.ok for check in checks))
        if args.command == "ingest":
            capture = load_capture_bundle(
                args.capture, allow_metadata_only=args.allow_metadata_only
            )
            print(
                json.dumps(
                    {
                        "capture": str(capture.root),
                        "frames": len(capture.frames),
                        "imu_samples": len(capture.imu_samples),
                        "has_media": capture.has_media,
                        "warnings": capture.warnings,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            result = run_pipeline(
                args.capture, args.out, config, allow_proxy=args.allow_proxy
            )
            print(
                json.dumps(
                    {
                        "usd": str(result.usd_path),
                        "artifacts": str(result.artifacts_path),
                        "validation": result.validation.to_dict(),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "validate":
            report = validate_map(args.usd, allow_proxy=args.allow_proxy)
            print(json.dumps(report.to_dict(), indent=2))
            return int(not report.valid)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
