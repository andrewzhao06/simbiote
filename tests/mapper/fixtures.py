from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    capture = tmp_path / "capture"
    (capture / "depth").mkdir(parents=True)
    (capture / "confidence").mkdir()
    (capture / "camera_matrix.csv").write_text(
        "1000,0,960\n0,1000,720\n0,0,1\n", encoding="utf-8"
    )
    (capture / "odometry.csv").write_text(
        "timestamp,frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy\n"
        "0.0,0,0,1,0,0,0,0,1,1000,1000,960,720\n"
        "0.1,2,2,1,1,0,0,0,1,1000,1000,960,720\n",
        encoding="utf-8",
    )
    (capture / "imu.csv").write_text(
        "timestamp,a_x,a_y,a_z,alpha_x,alpha_y,alpha_z\n"
        "0.0,0,9.81,0,0,0,0\n"
        "0.05,0,9.81,0,0,0,0\n",
        encoding="utf-8",
    )
    for frame_id in (0, 2):
        (capture / "depth" / f"{frame_id:06d}.png").write_bytes(b"png")
        (capture / "confidence" / f"{frame_id:06d}.png").write_bytes(b"png")
    (capture / "rgb.mp4").write_bytes(b"video")
    return capture
