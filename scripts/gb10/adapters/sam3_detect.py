"""SAM 3 semantic-labelling adapter (FACTORYFLOW_SAM3_COMMAND).

Runs Meta's SAM 3 over posed RGB frames from a Stray Scanner capture, projects
each mask into 3D with the capture's own LiDAR depth and ARKit poses, merges
detections of the same object seen from several frames, and writes the
`detections.json` that `simbiote.mapper.stages.label_scene` consumes.

SAM 3 does detection and segmentation from a text phrase in one model, so the
prompts below are the whole configuration -- no class list, no fine-tuning.

Wire it up with:

    export FACTORYFLOW_SAM3_COMMAND="$FF_PY <this file> \\
        --capture {capture} --geometry {geometry} \\
        --checkpoint {checkpoint} --output {output}"

The projection deliberately reuses `stages.unproject_frame`, so a labelled
object lands in exactly the same coordinates as the point cloud it sits in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from simbiote.mapper.config import MapperConfig
from simbiote.mapper.ingest import load_capture_bundle
from simbiote.mapper.stages import rgb_width, unproject_frame

# Hospital-flavoured defaults: things a mobile manipulator would be asked to
# pick up, plus the surfaces they rest on. Override with --object-prompts.
DEFAULT_OBJECT_PROMPTS = (
    "backpack",
    "bag",
    "bottle",
    "box",
    "cardboard box",
    "chair",
    "cup",
    "laptop",
    "monitor",
    "keyboard",
    "book",
    "tray",
)
DEFAULT_FLOOR_PROMPT = "floor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--geometry", type=Path, default=None, help="Reconstructed layer (unused; the "
        "adapter projects with the capture's own depth so it matches exactly)"
    )
    parser.add_argument("--config", type=Path, default=None, help="Mapper TOML")
    parser.add_argument(
        "--object-prompts",
        default=";".join(DEFAULT_OBJECT_PROMPTS),
        help="Semicolon-separated SAM 3 text prompts for graspable objects",
    )
    parser.add_argument("--floor-prompt", default=DEFAULT_FLOOR_PROMPT)
    parser.add_argument(
        "--frames", type=int, default=12, help="RGB frames to sample across the sweep"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.5, help="SAM 3 confidence floor"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=60,
        help="Discard a detection whose mask projects to fewer LiDAR points than this",
    )
    parser.add_argument(
        "--merge-radius",
        type=float,
        default=0.4,
        help="Metres; detections of one label closer than this are one object",
    )
    parser.add_argument(
        "--max-object-size",
        type=float,
        default=2.0,
        help="Metres; reject blobs bigger than this as over-segmentation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sam3-root",
        type=Path,
        default=Path("/home/dell/AI/repos/sam3"),
        help="SAM 3 checkout to import from",
    )
    return parser.parse_args()


EXTRACT_HEIGHT = 768


def usable_frame_count(video: Path, timestamps: np.ndarray) -> int:
    """How many video frames provably line up 1:1 with odometry rows.

    SCAN_MAP.md 4.2 warns that Stray Scanner has historically dropped frames,
    and this capture really is short one: 926 encoded frames against 927
    odometry rows. Position-based pairing would then attach the wrong pose to
    every frame after the drop and silently misplace every labelled object, so
    check the pairing against presentation timestamps instead of assuming it.

    Video PTS restart at zero, so compare against odometry timestamps measured
    from their own first sample. Returns the length of the verified prefix.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode:
        raise RuntimeError(f"ffprobe failed on {video.name}: {probe.stderr.strip()}")
    pts = np.array([
        float(line.strip().rstrip(",")) for line in probe.stdout.splitlines()
        if line.strip().rstrip(",")
    ])
    if len(pts) == 0:
        raise RuntimeError(f"{video.name} contains no decodable video frames")

    relative = timestamps - timestamps[0]
    count = min(len(pts), len(relative))
    interval = float(np.median(np.diff(relative))) if len(relative) > 1 else 0.0
    if not interval or count < 2:
        return count

    # The container's declared frame rate is not always the rate the capture
    # ran at. scan-0855cbfdfd is 30 Hz content tagged 60 fps: its PTS advance
    # at exactly half the odometry rate, so raw drift reaches 10 s across a
    # 21 s capture even though the frames pair 1:1 (confirmed by correlating
    # RGB edges against the depth maps -- the match peaks at the 1:1 index at
    # every probe, never at the half index). A wrong tag rescales the whole
    # PTS axis and cannot reorder frames, so put PTS on the odometry timebase
    # before looking for drops. Only do this when the frame counts already say
    # the streams are 1:1; if the video really did carry twice the frames, the
    # counts would differ ~2x and rescaling would hide a genuine mismatch.
    video_interval = float(np.median(np.diff(pts))) if len(pts) > 1 else 0.0
    if video_interval > 0 and abs(len(pts) - len(relative)) <= 0.05 * len(relative):
        pts = pts * (interval / video_interval)

    drift = pts[:count] - relative[:count]

    # Both clocks free-run, so a few us per frame of skew accumulates over a
    # long capture (~5 us/frame here, 18 ms across d6724d7d8d's 60 s). That is
    # a straight line and harms nothing; subtract it before judging.
    index = np.arange(count)
    slope, intercept = np.polyfit(index, drift, 1)
    residual = drift - (slope * index + intercept)

    # What actually breaks index pairing is a *persistent* shift: drop a frame
    # and every later frame inherits the offset. Stray Scanner's ordinary
    # jitter instead bumps a single frame by one interval and recovers on the
    # next (40 such blips in d6724d7d8d, all self-correcting). Comparing the
    # median of the window behind each frame with the window ahead of it keeps
    # the original guard's intent while ignoring blips that heal.
    window = max(5, round(0.5 / interval))  # ~half a second of frames
    if count > 2 * window:
        windows = np.lib.stride_tricks.sliding_window_view(residual, window)
        medians = np.median(windows, axis=1)          # medians[i] covers [i, i+window)
        shifts = medians[window:] - medians[:-window]  # ahead-of-i minus behind-i
        worst = int(np.argmax(np.abs(shifts)))
        if abs(shifts[worst]) > interval / 2:
            raise RuntimeError(
                f"{video.name} desynchronises from odometry at frame "
                f"{worst + window} (pairing shifts by {shifts[worst]:.4f}s and "
                f"stays shifted; one frame is {interval:.4f}s). Frames were "
                "dropped mid-stream; pairing by position would attach the wrong "
                "pose to everything after that point."
            )

    if len(pts) < len(relative):
        print(
            f"[sam3] note: {video.name} is truncated -- {len(pts)} frames for "
            f"{len(relative)} odometry rows. The first {count} align to within "
            f"{np.abs(residual).max():.4f}s of a straight line, so the tail was "
            "simply never encoded; ignoring the trailing rows."
        )
    return count


def extract_frames(video: Path, into: Path) -> list[Path]:
    """Decode rgb.mp4 to numbered stills, one per video frame.

    Decoded in a single pass rather than with a `select='eq(n,X)+...'` filter,
    which silently emitted one fewer still than requested and left no way to
    tell which index went missing.

    Downscaled on the way out: SAM 3 resizes internally, and the mask is
    point-sampled onto a 256x192 depth raster regardless, so full 1920x1440
    stills would cost disk for no accuracy.
    """
    command = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"scale=-2:{EXTRACT_HEIGHT}",
        "-vsync", "0", "-q:v", "3",
        str(into / "frame_%05d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")
    return sorted(into.glob("frame_*.jpg"))


FLOOR_SLAB = 0.10


def load_geometry_points(geometry: Path | None) -> np.ndarray | None:
    """Points from the reconstructed layer, or None if it is unreadable.

    Only used to widen the floor's extent, so a missing or non-point-based
    geometry layer (e.g. a 3DGRUT mesh) degrades to SAM 3's own extent rather
    than failing the run.
    """
    if geometry is None or not Path(geometry).is_file():
        return None
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(geometry))
        for prim in stage.Traverse():
            points = UsdGeom.Points(prim)
            if points:
                raw = points.GetPointsAttr().Get()
                if raw:
                    return np.asarray([tuple(p) for p in raw], dtype=np.float64)
    except Exception as exc:
        print(f"[sam3] note: could not read {geometry}: {exc}")
    return None


def robust_bounds(points: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Axis-aligned box from the 5th-95th percentile, so mask bleed onto the
    wall behind an object does not inflate it."""
    low = np.percentile(points, 5, axis=0)
    high = np.percentile(points, 95, axis=0)
    centre = (low + high) / 2
    size = np.maximum(high - low, 0.05)
    return tuple(float(v) for v in centre), tuple(float(v) for v in size)


def merge_detections(raw: list[dict], radius: float) -> list[dict]:
    """One physical object seen from several frames should be one node."""
    merged: list[dict] = []
    for detection in sorted(raw, key=lambda item: -item["score"]):
        for existing in merged:
            if existing["label"] != detection["label"]:
                continue
            if float(np.linalg.norm(
                np.array(existing["centre"]) - np.array(detection["centre"])
            )) <= radius:
                existing["points"] = np.vstack([existing["points"], detection["points"]])
                existing["centre"], existing["size"] = robust_bounds(existing["points"])
                existing["score"] = max(existing["score"], detection["score"])
                existing["frames"].extend(detection["frames"])
                break
        else:
            merged.append(dict(detection))
    return merged


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = MapperConfig.load(args.config)
    capture = load_capture_bundle(args.capture)
    width = rgb_width(capture, config)

    usable = usable_frame_count(
        capture.video_path,
        np.array([frame.timestamp for frame in capture.frames]),
    )
    paired = list(capture.frames[:usable])

    # Sample evenly across the sweep so objects are seen from several angles.
    count = min(args.frames, len(paired))
    sampled = [
        paired[index] for index in np.linspace(0, len(paired) - 1, count).astype(int)
    ]
    print(
        f"[sam3] {len(capture.frames)} odometry frames, {usable} paired with RGB, "
        f"sampling {len(sampled)}"
    )

    if args.sam3_root:
        sys.path.insert(0, str(args.sam3_root))
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    checkpoint = args.checkpoint
    if checkpoint.is_dir():
        checkpoint = checkpoint / "sam3.pt"
    print(f"[sam3] loading {checkpoint} on {args.device}")
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint), load_from_HF=False, device=args.device
    )
    processor = Sam3Processor(model, confidence_threshold=args.min_score)

    object_prompts = [p.strip() for p in args.object_prompts.split(";") if p.strip()]
    prompts = [(args.floor_prompt, "floor")] + [(p, "object") for p in object_prompts]

    raw: list[dict] = []
    with tempfile.TemporaryDirectory() as scratch:
        stills = extract_frames(capture.video_path, Path(scratch))
        if len(stills) < usable:
            raise RuntimeError(
                f"ffmpeg wrote {len(stills)} stills but ffprobe counted {usable} "
                "verified frames"
            )
        # SAM 3's weights are bfloat16; without the autocast context the first
        # matmul dies on "mat1 and mat2 must have the same dtype".
        for frame in sampled:
            # frame_id is odometry's own `frame` column, which indexes rgb.mp4.
            image = Image.open(stills[frame.frame_id]).convert("RGB")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)
                per_prompt = []
                for prompt, kind in prompts:
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(prompt=prompt, state=state)
                    per_prompt.append((
                        prompt,
                        kind,
                        state["masks"].squeeze(1).float().cpu().numpy(),
                        state["scores"].float().cpu().numpy(),
                    ))
            for prompt, kind, masks, scores in per_prompt:
                for mask, score in zip(masks, scores, strict=True):
                    if score < args.min_score:
                        continue
                    mask = mask > 0.5
                    points = unproject_frame(frame, config, width, mask=mask)
                    if len(points) < args.min_points:
                        continue
                    centre, size = robust_bounds(points)
                    if kind == "object" and max(size) > args.max_object_size:
                        continue
                    raw.append({
                        "label": prompt,
                        "kind": kind,
                        "score": float(score),
                        "centre": centre,
                        "size": size,
                        "points": points,
                        "frames": [frame.frame_id],
                    })

    print(f"[sam3] {len(raw)} raw detections above score {args.min_score}")
    merged = merge_detections(raw, args.merge_radius)

    floors = [d for d in merged if d["kind"] == "floor"]
    objects = [d for d in merged if d["kind"] == "object"]
    print(f"[sam3] merged into {len(floors)} floor region(s), {len(objects)} object(s)")

    nodes: list[dict] = []
    if floors:
        # Every floor detection is the same physical plane; pool them and flatten
        # to a thin slab at its median height rather than trusting mask bleed up
        # the walls for the vertical extent.
        pooled = np.vstack([d["points"] for d in floors])
        height = float(np.median(pooled[:, 1]))
        slab = pooled[np.abs(pooled[:, 1] - height) <= FLOOR_SLAB]
        centre, size = robust_bounds(slab if len(slab) >= args.min_points else pooled)

        # SAM 3 only sees the floor inside the handful of frames sampled, so its
        # extent under-covers the room. The height is the trustworthy part
        # (semantics); take the extent from every reconstructed point lying in
        # that same slab, which is the whole navigable area Step 2 plans over.
        cloud = load_geometry_points(args.geometry)
        if cloud is not None:
            on_plane = cloud[np.abs(cloud[:, 1] - height) <= FLOOR_SLAB]
            if len(on_plane) >= args.min_points:
                wide_centre, wide_size = robust_bounds(on_plane)
                print(
                    f"[sam3] floor extent widened from SAM 3's "
                    f"{size[0]:.1f}x{size[2]:.1f} m to the reconstruction's "
                    f"{wide_size[0]:.1f}x{wide_size[2]:.1f} m at y={height:.2f}"
                )
                centre, size = wide_centre, wide_size

        nodes.append({
            "node_id": "navigable_floor",
            "label": "navigable floor",
            "kind": "floor",
            "confidence": max(d["score"] for d in floors),
            "bounds": {
                "center": [centre[0], height, centre[2]],
                "size": [size[0], 0.05, size[2]],
            },
            "source_frame_ids": sorted({f for d in floors for f in d["frames"]}),
        })

    seen: dict[str, int] = {}
    for detection in sorted(objects, key=lambda item: -item["score"]):
        label = detection["label"]
        seen[label] = seen.get(label, 0) + 1
        identifier = label.replace(" ", "_")
        if seen[label] > 1:
            identifier = f"{identifier}_{seen[label]}"
        nodes.append({
            "node_id": identifier,
            "label": label,
            "kind": "object",
            "confidence": detection["score"],
            "bounds": {
                "center": list(detection["centre"]),
                "size": list(detection["size"]),
            },
            "source_frame_ids": sorted(set(detection["frames"])),
        })

    detections = args.output / "detections.json"
    detections.write_text(
        json.dumps({"proxy": False, "nodes": nodes}, indent=2), encoding="utf-8"
    )
    for node in nodes:
        centre = node["bounds"]["center"]
        size = node["bounds"]["size"]
        print(
            f"  {node['kind']:6s} {node['node_id']:24s} conf {node['confidence']:.2f} "
            f"at ({centre[0]:+.2f}, {centre[1]:+.2f}, {centre[2]:+.2f}) "
            f"size ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})"
        )

    if not floors:
        print("[sam3] ERROR: no floor detected; Step 2 needs a navigable region",
              file=sys.stderr)
        return 1
    if not objects:
        print("[sam3] ERROR: no graspable objects detected. Widen --object-prompts "
              "or lower --min-score; the scene may genuinely contain none.",
              file=sys.stderr)
        return 1
    print(f"[sam3] wrote {detections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
