"""Produce Simbiote's semantic detections contract from a SAM 3 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def first_pose(capture: Path) -> tuple[int, tuple[float, float, float]]:
    with (capture / "odometry.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    normalized = {key.strip().lower(): value for key, value in row.items()}
    frame_id = int(normalized["frame"])
    return frame_id, tuple(float(normalized[axis]) for axis in ("x", "y", "z"))


def lidar_depth_meters(path: Path) -> float:
    if not path.is_file():
        return 1.0
    values = np.asarray(Image.open(path)).astype(np.float32)
    values = values[values > 0]
    if not len(values):
        return 1.0
    value = float(median(values.tolist()))
    return value / 1000 if value > 20 else value


def first_video_frame(video: Path) -> Image.Image:
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "frame.png"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(image_path),
            ],
            check=True,
        )
        return Image.open(image_path).convert("RGB").copy()


def tensor_values(value: object) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).astype(float).tolist()


def main() -> int:
    args = parse_args()
    frame_id, pose = first_pose(args.capture)
    image = first_video_frame(args.capture / "rgb.mp4")
    depth = lidar_depth_meters(args.capture / "depth" / f"{frame_id:06d}.png")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), device="cuda")
    processor = Sam3Processor(model)
    prompt_list = [
        item.strip()
        for item in os.getenv(
            "SAM3_OBJECT_PROMPTS", "tray,bottle,box,small movable object"
        ).split(",")
        if item.strip()
    ]
    nodes = [
        {
            "node_id": "navigable_floor",
            "label": "navigable floor",
            "kind": "floor",
            "confidence": 1.0,
            "bounds": {
                "center": [pose[0], pose[1] - 1.0, pose[2]],
                "size": [6.0, 0.05, 6.0],
            },
            "source_frame_ids": [frame_id],
        }
    ]
    for prompt in prompt_list:
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=prompt)
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        for index, (box, score) in enumerate(zip(boxes, scores, strict=False)):
            confidence = tensor_values(score)[0]
            if confidence < 0.5:
                continue
            x0, y0, x1, y1 = tensor_values(box)
            width = max((x1 - x0) / image.width * depth, 0.05)
            height = max((y1 - y0) / image.height * depth, 0.05)
            nodes.append(
                {
                    "node_id": f"{prompt.replace(' ', '_')}_{index}",
                    "label": prompt,
                    "kind": "object",
                    "confidence": confidence,
                    "bounds": {
                        "center": [pose[0], pose[1] - 0.7, pose[2] + depth],
                        "size": [width, max(height, 0.05), width],
                    },
                    "source_frame_ids": [frame_id],
                }
            )
    if len(nodes) == 1:
        raise RuntimeError(
            "SAM 3 found no graspable object. Change SAM3_OBJECT_PROMPTS and retry."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "nodes": nodes,
                "metadata": {
                    "geometry": str(args.geometry),
                    "frame_id": frame_id,
                    "depth_meters": depth,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
