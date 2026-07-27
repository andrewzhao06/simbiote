"""End-to-end orchestration for the Teammate 1 mapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from factoryflow_mapper.config import MapperConfig
from factoryflow_mapper.ingest import load_capture_bundle
from factoryflow_mapper.stages import (
    build_scene_graph,
    complete_depth,
    label_scene,
    reconstruct,
    refine_poses,
)
from factoryflow_mapper.usd import ValidationReport, export_usd, validate_map


@dataclass(frozen=True, slots=True)
class PipelineResult:
    usd_path: Path
    artifacts_path: Path
    validation: ValidationReport


def run_pipeline(
    capture_path: str | Path,
    out_path: str | Path,
    config: MapperConfig,
    *,
    allow_proxy: bool = False,
) -> PipelineResult:
    capture = load_capture_bundle(
        capture_path, allow_metadata_only=config.mode == "proxy"
    )
    run_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    artifacts = config.work_root / run_name
    artifacts.mkdir(parents=True, exist_ok=False)

    refined = refine_poses(capture, config, artifacts)
    depth = complete_depth(refined, config, artifacts)
    reconstruction = reconstruct(depth, config, artifacts)
    labels = label_scene(reconstruction, config, artifacts)
    graph = build_scene_graph(labels, reconstruction, artifacts)
    usd_path = export_usd(graph, out_path)
    validation = validate_map(usd_path, allow_proxy=allow_proxy)

    manifest = {
        "capture": str(Path(capture_path).resolve()),
        "output": str(usd_path),
        "mode": config.mode,
        "warnings": capture.warnings,
        "validation": validation.to_dict(),
    }
    (artifacts / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if not validation.valid:
        raise RuntimeError(
            "Map failed Step 2 contract validation: " + "; ".join(validation.errors)
        )
    return PipelineResult(usd_path, artifacts, validation)
