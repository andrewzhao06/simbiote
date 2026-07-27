"""WiLoR hand-pose backend -- the GB10/aarch64 path.

`hand_tracking.HandTracker` uses MediaPipe, which has **no linux-aarch64
wheel** (upstream ships macOS/Windows arm64 only), so it cannot run on the
GB10 at all. WiLoR is the backend the spec always intended for the GB10
anyway: a ViT that does detection + MANO mesh recovery end-to-end, with much
better finger articulation than MediaPipe.

This module keeps WiLoR behind the exact same contract as MediaPipe --
`get_hand_landmarks(frame) -> Optional[HandLandmarks]`, 21 landmarks,
normalized to [0, 1] in image space -- so `ik_bridge.py` and
`teleop_session.py` are backend-agnostic (spec 6.4).

Why the landmark indices line up for free: WiLoR's MANO wrapper remaps its
joints through `mano_to_openpose`, and the OpenPose hand ordering is the same
convention MediaPipe uses -- 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
13-16 ring, 17-20 pinky. So WRIST/THUMB_TIP/INDEX_TIP/MIDDLE_MCP in
`hand_tracking.py` address the same joints under either backend.

Model weights are the ones already staged in /home/dell/AI/models/wilor;
override with SIMBIOTE_WILOR_DIR. The MANO pickle must be the chumpy-free
copy produced by scripts/gb10/dechumpify_mano.py (chumpy does not import on
Python 3.12), which lives in assets/mano/ by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from simbiote.teleop.hand_tracking import HandLandmarks, NUM_LANDMARKS, WRIST

DEFAULT_WILOR_DIR = Path(os.environ.get("SIMBIOTE_WILOR_DIR", "/home/dell/AI/models/wilor"))
DEFAULT_MANO_DIR = Path(
    os.environ.get(
        "SIMBIOTE_MANO_DIR",
        str(Path(__file__).resolve().parents[2] / "assets" / "mano"),
    )
)

# Re-running the YOLO hand detector on every frame roughly halves throughput.
# Between detections we re-derive the crop box from the previous frame's
# keypoints, which is stable as long as the hand doesn't teleport.
DEFAULT_DETECT_INTERVAL = 5

# How much padding to add around the previous frame's keypoints when reusing
# them as the next crop box (fraction of the keypoint bounding-box size).
TRACK_BOX_MARGIN = 0.6


class WiLoRHandTracker:
    """WiLoR-backed hand tracker. Construct once, call repeatedly.

    Mirrors `hand_tracking.HandTracker`'s interface exactly.
    """

    def __init__(
        self,
        wilor_dir: Optional[Path] = None,
        mano_dir: Optional[Path] = None,
        device: Optional[str] = None,
        min_detection_confidence: float = 0.3,
        detect_interval: int = DEFAULT_DETECT_INTERVAL,
        rescale_factor: float = 2.0,
    ):
        import torch

        self.wilor_dir = Path(wilor_dir or DEFAULT_WILOR_DIR)
        self.mano_dir = Path(mano_dir or DEFAULT_MANO_DIR)
        self.min_detection_confidence = min_detection_confidence
        self.detect_interval = max(1, detect_interval)
        self.rescale_factor = rescale_factor

        self._check_assets()

        # WiLoR is a loose research repo, not a package: it imports its own
        # modules as top-level `wilor.*`, so its root has to be on sys.path.
        import sys

        if str(self.wilor_dir) not in sys.path:
            sys.path.insert(0, str(self.wilor_dir))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model, self.model_cfg = self._load_model()
        self.model = self.model.to(self.device).eval()

        from ultralytics import YOLO

        self._detector = YOLO(str(self.wilor_dir / "pretrained_models" / "detector.pt"))

        self._frame_index = 0
        self._last_box: Optional[np.ndarray] = None  # xyxy in full-image pixels
        self._last_is_right: float = 1.0

    # -- setup ---------------------------------------------------------------

    def _check_assets(self) -> None:
        checkpoint = self.wilor_dir / "pretrained_models" / "wilor_final.ckpt"
        detector = self.wilor_dir / "pretrained_models" / "detector.pt"
        mano = self.mano_dir / "MANO_RIGHT.pkl"

        for path, hint in [
            (checkpoint, "WiLoR checkpoint"),
            (detector, "WiLoR hand detector"),
            (
                mano,
                "chumpy-free MANO pickle -- generate it with "
                "`python scripts/gb10/dechumpify_mano.py --out assets/mano/MANO_RIGHT.pkl`",
            ),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"missing {hint}: {path}")

    def _load_model(self):
        """Reimplements WiLoR's `load_wilor` with absolute paths.

        Upstream's `load_wilor()` hardcodes a relative './mano_data/', so it
        only works if the process cwd happens to be the WiLoR checkout. We
        also repoint MANO at our de-chumpified pickle.
        """

        import torch
        from wilor.configs import get_config
        from wilor.models import WiLoR

        cfg_path = self.wilor_dir / "pretrained_models" / "model_config.yaml"
        cfg = get_config(str(cfg_path), update_cachedir=True)
        cfg.defrost()
        if "vit" in cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in cfg.MODEL:
            cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        cfg.MANO.DATA_DIR = str(self.mano_dir) + os.sep
        cfg.MANO.MODEL_PATH = str(self.mano_dir) + os.sep
        cfg.MANO.MEAN_PARAMS = str(self.wilor_dir / "mano_data" / "mano_mean_params.npz")
        cfg.freeze()

        model = WiLoR.load_from_checkpoint(
            str(self.wilor_dir / "pretrained_models" / "wilor_final.ckpt"),
            strict=False,
            cfg=cfg,
            map_location=torch.device("cpu"),
        )
        return model, cfg

    # -- detection -----------------------------------------------------------

    def _detect_box(self, img_rgb: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
        """Run YOLO and return (xyxy box, is_right) for the best hand, or None."""

        detections = self._detector(
            img_rgb, conf=self.min_detection_confidence, verbose=False, iou=0.5
        )[0]
        if len(detections) == 0:
            return None

        best, best_conf = None, -1.0
        for det in detections:
            conf = float(det.boxes.conf.cpu().numpy().reshape(-1)[0])
            if conf > best_conf:
                best_conf = conf
                # detector class: 0 = left hand, 1 = right hand
                is_right = float(det.boxes.cls.cpu().numpy().reshape(-1)[0])
                best = (det.boxes.xyxy.cpu().numpy().reshape(-1)[:4].astype(np.float32), is_right)
        return best

    def _box_from_keypoints(self, keypoints_px: np.ndarray) -> np.ndarray:
        """Re-derive a crop box from the previous frame's keypoints."""

        x0, y0 = keypoints_px.min(axis=0)
        x1, y1 = keypoints_px.max(axis=0)
        pad_x = (x1 - x0) * TRACK_BOX_MARGIN * 0.5
        pad_y = (y1 - y0) * TRACK_BOX_MARGIN * 0.5
        return np.array([x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y], dtype=np.float32)

    # -- inference -----------------------------------------------------------

    def get_hand_landmarks(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """frame: a BGR image as read from camera_source.FrameSource.read().

        Returns None if no hand is found in this frame.
        """

        import torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset

        img_rgb = frame[:, :, ::-1]
        height, width = frame.shape[:2]

        # Detect on an interval; reuse the tracked box on the frames between,
        # and always fall back to a fresh detection if tracking has no box.
        due_for_detection = (self._frame_index % self.detect_interval == 0) or self._last_box is None
        self._frame_index += 1

        if due_for_detection:
            found = self._detect_box(img_rgb)
            if found is None:
                self._last_box = None
                return None
            box, is_right = found
            self._last_box, self._last_is_right = box, is_right
        else:
            box, is_right = self._last_box, self._last_is_right

        dataset = ViTDetDataset(
            self.model_cfg,
            img_rgb,
            box[None, :],
            np.array([is_right], dtype=np.float32),
            rescale_factor=self.rescale_factor,
        )
        batch = torch.utils.data.default_collate([dataset[0]])
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        with torch.no_grad():
            out = self.model(batch)

        keypoints_2d = out["pred_keypoints_2d"][0].float().cpu().numpy()  # (21, 2), crop space
        keypoints_3d = out["pred_keypoints_3d"][0].float().cpu().numpy()  # (21, 3), root-relative m

        # The dataset horizontally flips left hands into the model's
        # right-hand canonical frame, so undo that on the way out.
        mirror = 2.0 * float(is_right) - 1.0
        keypoints_2d[:, 0] *= mirror
        keypoints_3d[:, 0] *= mirror

        # Crop-normalized -> full-image pixels -> [0, 1] normalized.
        box_center = batch["box_center"].float().cpu().numpy()[0]
        box_size = float(batch["box_size"].float().cpu().numpy()[0])
        keypoints_px = keypoints_2d * box_size + box_center

        # Feed the tracker for the frames that skip detection.
        self._last_box = self._box_from_keypoints(keypoints_px)

        points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        points[:, 0] = keypoints_px[:, 0] / width
        points[:, 1] = keypoints_px[:, 1] / height
        points[:, 2] = self._relative_depth(keypoints_2d, keypoints_3d)

        handedness = "Right" if is_right >= 0.5 else "Left"
        confidence = 1.0

        return HandLandmarks(points=points, handedness=handedness, confidence=confidence)

    @staticmethod
    def _relative_depth(keypoints_2d: np.ndarray, keypoints_3d: np.ndarray) -> np.ndarray:
        """Root-relative z, rescaled to match MediaPipe's z convention.

        MediaPipe reports z on roughly the same scale as normalized x/y, with
        negative meaning closer to the camera. WiLoR's z is root-relative
        metres, so scale it by the 2D/3D size ratio of the same hand to land
        in comparable units.
        """

        span_2d = float(np.linalg.norm(keypoints_2d.max(axis=0) - keypoints_2d.min(axis=0)))
        span_3d = float(np.linalg.norm(keypoints_3d.max(axis=0) - keypoints_3d.min(axis=0)))
        scale = span_2d / span_3d if span_3d > 1e-9 else 0.0
        return (keypoints_3d[:, 2] - keypoints_3d[WRIST, 2]) * scale

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._last_box = None

    def __enter__(self) -> "WiLoRHandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
