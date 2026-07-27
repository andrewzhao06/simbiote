"""Configuration loaded from TOML and environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MapperConfig:
    ssd_root: Path
    work_root: Path
    models_root: Path
    tools_root: Path
    colmap_binary: Path
    ffprobe_binary: Path
    depth_checkpoint: Path
    sam3_checkpoint: Path
    dgrut_root: Path
    hospital_usd: Path
    mode: str = "proxy"
    depth_scale_meters: float = 0.001
    confidence_threshold: int = 1

    @classmethod
    def load(cls, path: Path | None = None) -> MapperConfig:
        raw: dict[str, object] = {}
        if path:
            with path.open("rb") as handle:
                raw = tomllib.load(handle)

        paths = raw.get("paths", {})
        pipeline = raw.get("pipeline", {})
        if not isinstance(paths, dict) or not isinstance(pipeline, dict):
            raise ValueError("Config sections [paths] and [pipeline] must be TOML tables")

        def value(name: str, default: str = "") -> str:
            env_name = f"FACTORYFLOW_{name.upper()}"
            return os.getenv(env_name, str(paths.get(name, default)))

        ssd_root = Path(value("ssd_root", "/mnt/factoryflow-ssd/AI")).expanduser()
        models_root = Path(value("models_root", str(ssd_root / "models"))).expanduser()
        tools_root = Path(value("tools_root", str(ssd_root / "tools"))).expanduser()
        work_root = Path(value("work_root", "/var/factoryflow/stage/mapper")).expanduser()

        mode = os.getenv("FACTORYFLOW_MODE", str(pipeline.get("mode", "proxy")))
        if mode not in {"proxy", "preview", "production"}:
            raise ValueError("pipeline.mode must be 'proxy', 'preview', or 'production'")

        return cls(
            ssd_root=ssd_root,
            work_root=work_root,
            models_root=models_root,
            tools_root=tools_root,
            colmap_binary=Path(value("colmap_binary", "colmap")),
            ffprobe_binary=Path(value("ffprobe_binary", "ffprobe")),
            depth_checkpoint=Path(
                value("depth_checkpoint", str(models_root / "depth-anything"))
            ),
            sam3_checkpoint=Path(value("sam3_checkpoint", str(models_root / "sam3"))),
            dgrut_root=Path(value("dgrut_root", str(tools_root / "3dgrut"))),
            hospital_usd=Path(
                value(
                    "hospital_usd",
                    # Inside the Isaac asset pack, not a symlinked copy: USD
                    # resolves the scene's relative ./Props references against
                    # the link's own directory.
                    str(
                        ssd_root
                        / "assets/isaac-5.1/Isaac/Environments/Hospital/hospital.usd"
                    ),
                )
            ),
            mode=mode,
            depth_scale_meters=float(pipeline.get("depth_scale_meters", 0.001)),
            confidence_threshold=int(pipeline.get("confidence_threshold", 1)),
        )
