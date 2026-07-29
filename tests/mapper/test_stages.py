"""Geometry regression tests for the reconstruction stages.

Both cases here were live bugs that produced a plausible-looking but
geometrically meaningless point cloud from a real Stray Scanner capture.
"""

from __future__ import annotations

import math

import pytest

from simbiote.mapper.stages import StageError, _rotate, _run_adapter


def _reference_rotate(point, quaternion):
    """Rotate via an explicit rotation matrix, independent of _rotate."""
    qx, qy, qz, qw = quaternion
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    matrix = (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )
    return tuple(sum(row[i] * point[i] for i in range(3)) for row in matrix)


IDENTITY = (0.0, 0.0, 0.0, 1.0)
HALF = math.sqrt(0.5)


class TestRotate:
    def test_identity_quaternion_leaves_the_point_alone(self):
        # Regression: the previous implementation returned (-1, -2, -3) here,
        # because it subtracted |q|^2 * p instead of |q_vector|^2 * p.
        assert _rotate((1.0, 2.0, 3.0), IDENTITY) == pytest.approx((1.0, 2.0, 3.0))

    def test_quarter_turn_about_y_maps_x_axis_onto_negative_z(self):
        rotated = _rotate((1.0, 0.0, 0.0), (0.0, HALF, 0.0, HALF))
        assert rotated == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)

    def test_quarter_turn_about_x_maps_y_axis_onto_z(self):
        rotated = _rotate((0.0, 1.0, 0.0), (HALF, 0.0, 0.0, HALF))
        assert rotated == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)

    @pytest.mark.parametrize(
        "quaternion",
        [
            (0.5, 0.5, 0.5, 0.5),
            (-0.13589308, 0.01881028, -0.9905318, -0.0050859754),  # real capture frame
            (0.9796523, -0.0110650705, -0.19823788, 0.02933473),
            (0.1, 0.2, 0.3, 0.9),
        ],
    )
    def test_preserves_length(self, quaternion):
        """A rotation is an isometry; the old formula was not."""
        point = (0.4, -1.3, 2.7)
        expected = math.dist((0.0, 0.0, 0.0), point)
        actual = math.dist((0.0, 0.0, 0.0), _rotate(point, quaternion))
        assert actual == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize(
        "quaternion",
        [
            IDENTITY,
            (0.5, 0.5, 0.5, 0.5),
            (-0.13589308, 0.01881028, -0.9905318, -0.0050859754),
            (0.1, 0.2, 0.3, 0.9),
        ],
    )
    def test_matches_an_independent_rotation_matrix(self, quaternion):
        point = (0.5, 0.3, 2.0)
        assert _rotate(point, quaternion) == pytest.approx(
            _reference_rotate(point, quaternion), abs=1e-9
        )

    def test_unnormalised_quaternion_does_not_rescale_the_point(self):
        point = (0.5, 0.3, 2.0)
        scaled = tuple(3.0 * value for value in (0.1, 0.2, 0.3, 0.9))
        assert _rotate(point, scaled) == pytest.approx(
            _rotate(point, (0.1, 0.2, 0.3, 0.9)), abs=1e-9
        )

    def test_zero_quaternion_is_rejected(self):
        with pytest.raises(StageError):
            _rotate((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))


class TestRunAdapter:
    """The production adapter boundary: an env-var command template plus paths.

    Nothing else in the suite reaches this branch, and every path it receives
    is user-supplied (SSD mount, capture folder, repo checkout).
    """

    @staticmethod
    def _echo_adapter(tmp_path):
        """A script that writes its argv, one entry per line, to argv.txt."""
        script = tmp_path / "adapter with space.sh"
        script.write_text(
            '#!/bin/sh\nfor a in "$@"; do echo "$a"; done > "$(dirname "$0")/argv.txt"\n'
        )
        script.chmod(0o755)
        return script

    def test_paths_containing_spaces_reach_the_adapter_intact(self, tmp_path, monkeypatch):
        """Regression: the template was formatted *then* split, so a space in
        any substituted path silently became an argument boundary and the
        adapter was handed a truncated path that does not exist."""
        script = self._echo_adapter(tmp_path)
        capture = tmp_path / "my capture"
        capture.mkdir()
        monkeypatch.setenv("FF_TEST_ADAPTER", f"'{script}' {{capture}} {{output}}")

        _run_adapter("FF_TEST_ADAPTER", {"capture": capture, "output": tmp_path}, tmp_path)

        assert (tmp_path / "argv.txt").read_text().splitlines() == [
            str(capture),
            str(tmp_path),
        ]

    def test_unset_command_is_a_stage_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FF_TEST_ADAPTER", raising=False)
        with pytest.raises(StageError, match="Production mode requires"):
            _run_adapter("FF_TEST_ADAPTER", {"capture": tmp_path}, tmp_path)

    def test_unknown_placeholder_is_a_stage_error_not_a_keyerror(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FF_TEST_ADAPTER", "/bin/true {nope}")
        with pytest.raises(StageError, match="placeholder"):
            _run_adapter("FF_TEST_ADAPTER", {"capture": tmp_path}, tmp_path)

    def test_nonzero_exit_is_a_stage_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FF_TEST_ADAPTER", "/bin/sh -c 'exit 3'")
        with pytest.raises(StageError, match="exit code 3"):
            _run_adapter("FF_TEST_ADAPTER", {}, tmp_path)


class TestDepthIntrinsicsRegistration:
    """The LiDAR raster is ~7.5x smaller than the RGB frame the intrinsics
    describe (SCAN_MAP.md 4.3). Unprojection must rescale, or the cloud
    collapses toward the view axis."""

    def test_preview_cloud_spans_the_depth_field_of_view(self, tmp_path):
        import numpy as np
        from PIL import Image

        from simbiote.mapper.config import MapperConfig
        from simbiote.mapper.models import (
            CameraIntrinsics,
            CaptureBundle,
            Frame,
            Pose,
        )
        from simbiote.mapper.stages import _preview_points

        depth_width, depth_height = 256, 192
        rgb_width = 1920
        # A flat wall three metres straight ahead, filling the frame.
        depth = np.full((depth_height, depth_width), 3000, dtype=np.uint16)
        confidence = np.full((depth_height, depth_width), 2, dtype=np.uint8)
        Image.fromarray(depth).save(tmp_path / "depth.png")
        Image.fromarray(confidence).save(tmp_path / "confidence.png")

        frame = Frame(
            frame_id=0,
            timestamp=0.0,
            pose=Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            intrinsics=CameraIntrinsics(fx=1433.9, fy=1433.9, cx=961.3, cy=725.4),
            depth_path=tmp_path / "depth.png",
            confidence_path=tmp_path / "confidence.png",
        )
        capture = CaptureBundle(
            root=tmp_path,
            video_path=tmp_path / "rgb.mp4",  # absent: exercises the 2*cx fallback
            static_intrinsics=(1433.9, 0, 961.3, 0, 1433.9, 725.4, 0, 0, 1),
            frames=[frame],
            imu_samples=[],
            has_media=True,
        )
        config = MapperConfig.load(None)
        points = _preview_points(capture, config)

        # At 3 m, a 1920-px-wide frame with fx=1433.9 spans
        # 2 * 3 * (960/1433.9) ~= 4.0 m horizontally. The wall must too.
        span_x = max(p[0] for p in points) - min(p[0] for p in points)
        span_y = max(p[1] for p in points) - min(p[1] for p in points)
        assert span_x == pytest.approx(4.0, rel=0.05)
        assert span_y == pytest.approx(3.0, rel=0.05)
        # Every point sits on the wall, 3 m along +Z.
        assert all(p[2] == pytest.approx(3.0, abs=1e-6) for p in points)
        assert rgb_width / depth_width == pytest.approx(7.5)
