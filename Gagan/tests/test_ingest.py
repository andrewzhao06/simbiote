from pathlib import Path

import pytest

from Gagan.fixtures import capture_dir
from simbiote.mapper.ingest import CaptureValidationError, load_capture_bundle


def test_ingest_matches_files_by_frame_id(capture_dir: Path) -> None:
    capture = load_capture_bundle(capture_dir)

    assert [frame.frame_id for frame in capture.frames] == [0, 2]
    assert len(capture.imu_samples) == 2
    assert capture.frames[1].depth_path.name == "000002.png"
    assert capture.static_intrinsics == (
        1000.0,
        0.0,
        960.0,
        0.0,
        1000.0,
        720.0,
        0.0,
        0.0,
        1.0,
    )


def test_ingest_rejects_missing_required_bundle_file(capture_dir: Path) -> None:
    (capture_dir / "rgb.mp4").unlink()

    with pytest.raises(CaptureValidationError, match="missing media"):
        load_capture_bundle(capture_dir)


def test_ingest_accepts_metadata_only_capture_for_proxy_testing(capture_dir: Path) -> None:
    (capture_dir / "rgb.mp4").unlink()
    for directory in ("depth", "confidence"):
        for image in (capture_dir / directory).glob("*.png"):
            image.unlink()
        (capture_dir / directory).rmdir()

    capture = load_capture_bundle(capture_dir, allow_metadata_only=True)

    assert not capture.has_media
    assert [frame.frame_id for frame in capture.frames] == [0, 2]
    assert "Metadata-only capture" in capture.warnings[0]
