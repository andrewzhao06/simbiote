"""GB10-only stretch — spec §5.3/§5.4: teacher (privileged-state) -> vision
student (ResNet-18 laptop-tier stand-in here; Theia/ResNet-50 on the GB10)
distillation.

Not part of the core acceptance tests (§5.4) or Tier 1/2 fallbacks (§7) --
only attempt this once nav+grasp are solid and rehearsed, exactly like the
wheelchair stretch task. Kept dependency-light (no torchvision requirement)
by using a small hand-rolled CNN student instead of importing ResNet-18,
since the point tonight is validating the distillation *loop* shape, not
the exact backbone -- swap in Theia/ResNet-50 on the GB10 by changing
`VisionStudent` alone.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from simbiote.training.policy_net import ActorCriticMLP

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


class VisionStudent(nn.Module):
    """Tiny CNN standing in for ResNet-18 (laptop) / Theia or ResNet-50 (GB10).

    Same input/output contract either way: an RGB image in, an action mean
    out -- swapping the backbone tomorrow doesn't change any caller.
    """

    def __init__(self, act_dim: int, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, act_dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(images))


def distill(
    teacher: ActorCriticMLP,
    student: VisionStudent,
    render_fn: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    steps: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    out_path: str | Path | None = None,
) -> Path:
    """`render_fn(batch_size) -> (images, privileged_obs)` is provided by the
    caller (a rollout/replay source that can render both a camera frame and
    the matching privileged state for the same simulated moment). The
    teacher's action on `privileged_obs` becomes the regression target for
    the student's action on `images` -- standard teacher-student behavior
    cloning, not RL.
    """
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(steps):
        images, privileged_obs = render_fn(batch_size)
        with torch.no_grad():
            target_actions, _, _ = teacher.act(privileged_obs, deterministic=True)

        pred_actions = student(images)
        loss = loss_fn(pred_actions, target_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out_path = Path(out_path) if out_path else DEFAULT_CHECKPOINT_DIR / "vision_student.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "act_dim": student.head[-1].out_features,
            "image_size": student.image_size,
            "state_dict": student.state_dict(),
        },
        out_path,
    )
    return out_path


def make_synthetic_render_fn(obs_dim: int, act_dim: int, image_size: int = 64, seed: int = 0):
    """A synthetic `render_fn` for tonight's smoke test -- random images
    paired with random privileged obs of the right shape. Real GB10 usage
    replaces this with an Isaac Sim camera-render + ground-truth-state pair.
    """
    generator = torch.Generator().manual_seed(seed)

    def render_fn(batch_size: int):
        images = torch.rand(batch_size, 3, image_size, image_size, generator=generator)
        privileged_obs = torch.rand(batch_size, obs_dim, generator=generator) * 2 - 1
        return images, privileged_obs

    return render_fn
