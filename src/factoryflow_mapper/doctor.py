"""GB10 readiness checks for the mapper only."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from factoryflow_mapper.config import MapperConfig


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _executable(path: Path) -> bool:
    return path.is_file() or shutil.which(str(path)) is not None


def _nonempty(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(path.iterdir()))


def _checkpoint(path: Path, filename: str) -> bool:
    return path.is_file() or (path / filename).is_file()


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or result.stderr).strip()


def run_doctor(config: MapperConfig) -> list[Check]:
    production = config.mode == "production"
    checks = [
        Check(
            "architecture",
            platform.machine().lower() in {"aarch64", "arm64"},
            platform.machine(),
            required=False,
        ),
        Check("SSD root", config.ssd_root.is_dir(), str(config.ssd_root)),
        Check(
            "work root writable",
            config.work_root.exists() and os.access(config.work_root, os.W_OK),
            str(config.work_root),
        ),
        Check(
            "COLMAP",
            _executable(config.colmap_binary),
            str(config.colmap_binary),
            required=production,
        ),
        Check(
            "3DGRUT",
            _nonempty(config.dgrut_root),
            str(config.dgrut_root),
            required=production,
        ),
        Check(
            "depth checkpoint",
            _checkpoint(config.depth_checkpoint, "model.safetensors"),
            str(config.depth_checkpoint),
            required=production,
        ),
        Check(
            "SAM 3 checkpoint",
            _checkpoint(config.sam3_checkpoint, "sam3.pt"),
            str(config.sam3_checkpoint),
            required=production,
        ),
        Check(
            "hospital fallback",
            config.hospital_usd.is_file(),
            str(config.hospital_usd),
            required=production,
        ),
        Check(
            "COLMAP adapter",
            bool(os.getenv("FACTORYFLOW_COLMAP_COMMAND")),
            "FACTORYFLOW_COLMAP_COMMAND",
            required=production,
        ),
        Check(
            "depth adapter",
            bool(os.getenv("FACTORYFLOW_DEPTH_COMMAND")),
            "FACTORYFLOW_DEPTH_COMMAND",
            required=production,
        ),
        Check(
            "3DGRUT adapter",
            bool(os.getenv("FACTORYFLOW_DGRUT_COMMAND")),
            "FACTORYFLOW_DGRUT_COMMAND",
            required=production,
        ),
        Check(
            "SAM 3 adapter",
            bool(os.getenv("FACTORYFLOW_SAM3_COMMAND")),
            "FACTORYFLOW_SAM3_COMMAND",
            required=production,
        ),
    ]
    gpu = _command_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    checks.append(Check("NVIDIA GPU", bool(gpu), gpu or "nvidia-smi unavailable", production))
    return checks


def serialize_checks(checks: list[Check]) -> list[dict[str, object]]:
    return [asdict(check) for check in checks]
