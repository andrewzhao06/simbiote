"""Exports to ONNX/TorchScript for Steps 3 & 4 — spec §5.4/§5.7:

    "You produce for Step 3 (teleop) and Step 4 (agentic): exported
    ONNX/TorchScript checkpoints, callable via navigate_to(location_id) /
    pick_up(object_id)"

    python -m simbiote.training.export_policy --checkpoint checkpoints/nav_ppo.pt --format onnx --out checkpoints/nav_policy.onnx

Exports a deterministic, obs -> action_mean inference graph only (no
sampling, no value head) -- exactly what `robot_iface/skills.py` and Steps
3/4 need to call at deploy time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from simbiote.training.policy_net import ActorCriticMLP


class InferenceOnlyPolicy(nn.Module):
    """Wraps `ActorCriticMLP.actor_mean`, clamped to the action bounds --
    the one op ONNX/TorchScript actually needs to export cleanly."""

    def __init__(self, policy: ActorCriticMLP):
        super().__init__()
        self.actor_mean = policy.actor_mean
        low = policy.meta.act_low
        high = policy.meta.act_high
        self.register_buffer("act_low", torch.as_tensor(low, dtype=torch.float32) if low is not None else None, persistent=False)
        self.register_buffer("act_high", torch.as_tensor(high, dtype=torch.float32) if high is not None else None, persistent=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.actor_mean(obs)
        if self.act_low is not None and self.act_high is not None:
            action = torch.clamp(action, self.act_low, self.act_high)
        return action


def export_policy(checkpoint: str | Path, out_path: str | Path, fmt: str = "onnx") -> Path:
    policy = ActorCriticMLP.load(checkpoint)
    policy.eval()
    inference_model = InferenceOnlyPolicy(policy)
    inference_model.eval()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_obs = torch.zeros(1, policy.meta.obs_dim, dtype=torch.float32)

    if fmt == "onnx":
        torch.onnx.export(
            inference_model,
            dummy_obs,
            str(out_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
    elif fmt == "torchscript":
        scripted = torch.jit.trace(inference_model, dummy_obs)
        scripted.save(str(out_path))
    else:
        raise ValueError(f"Unknown export format '{fmt}', expected 'onnx' or 'torchscript'")

    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX/TorchScript for Steps 3 & 4.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript"])
    parser.add_argument("--out", type=str, required=True)
    return parser


def main(argv=None) -> Path:
    args = build_arg_parser().parse_args(argv)
    out_path = export_policy(args.checkpoint, args.out, args.format)
    print(f"[export_policy] wrote {args.format} policy to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
