from pathlib import Path

import numpy as np
from PIL import Image

from simbiote.mapper.config import MapperConfig
from simbiote.mapper.pipeline import run_pipeline
from simbiote.mapper.usd import validate_map


def _config(tmp_path: Path, *, mode: str = "proxy") -> MapperConfig:
    return MapperConfig(
        ssd_root=tmp_path,
        work_root=tmp_path / "work",
        models_root=tmp_path / "models",
        tools_root=tmp_path / "tools",
        colmap_binary=Path("colmap"),
        ffprobe_binary=Path("ffprobe"),
        depth_checkpoint=tmp_path / "models/depth",
        sam3_checkpoint=tmp_path / "models/sam3",
        dgrut_root=tmp_path / "tools/3dgrut",
        hospital_usd=tmp_path / "hospital.usd",
        mode=mode,
    )


def test_proxy_pipeline_exports_step2_contract(capture_dir: Path, tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.work_root.mkdir()
    output = tmp_path / "mapped.usda"

    result = run_pipeline(capture_dir, output, config, allow_proxy=True)

    assert result.usd_path == output.resolve()
    assert result.validation.valid
    assert output.with_suffix(".scene_graph.json").is_file()
    text = output.read_text(encoding="utf-8")
    assert "PhysicsCollisionAPI" in text
    assert "factoryflow:is_navigable = true" in text
    assert "factoryflow:is_graspable = true" in text


def test_proxy_is_rejected_for_production_handoff(
    capture_dir: Path, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    config.work_root.mkdir()
    output = tmp_path / "mapped.usda"
    run_pipeline(capture_dir, output, config, allow_proxy=True)

    report = validate_map(output)

    assert not report.valid
    assert not report.checks["non_proxy"]


def test_proxy_pipeline_accepts_metadata_only_capture(
    capture_dir: Path, tmp_path: Path
) -> None:
    (capture_dir / "rgb.mp4").unlink()
    for directory in ("depth", "confidence"):
        for image in (capture_dir / directory).glob("*.png"):
            image.unlink()
        (capture_dir / directory).rmdir()
    config = _config(tmp_path)
    config.work_root.mkdir()

    result = run_pipeline(capture_dir, tmp_path / "metadata-only.usda", config, allow_proxy=True)

    assert result.validation.valid


def test_preview_pipeline_exports_lidar_geometry(capture_dir: Path, tmp_path: Path) -> None:
    for frame_id in (0, 2):
        Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(
            capture_dir / "depth" / f"{frame_id:06d}.png"
        )
        Image.fromarray(np.full((4, 4), 2, dtype=np.uint8)).save(
            capture_dir / "confidence" / f"{frame_id:06d}.png"
        )
    config = _config(tmp_path, mode="preview")
    config.work_root.mkdir()
    output = tmp_path / "preview.usda"

    result = run_pipeline(capture_dir, output, config)

    assert result.validation.valid
    text = output.read_text(encoding="utf-8")
    # Typeless so the referenced Points type composes through -- an authored
    # Xform type would win and leave the prim unrenderable (empty world bound).
    assert 'def "ReconstructedGeometry"' in text
    assert 'def Xform "ReconstructedGeometry"' not in text
    preview = next(result.artifacts_path.glob("03_reconstruction/lidar_preview.usda"))
    assert 'def Points "LiDARPreview"' in preview.read_text(encoding="utf-8")
