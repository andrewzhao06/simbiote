import torch

from simbiote.training.distill_student import VisionStudent, distill, make_synthetic_render_fn
from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta


def test_vision_student_forward_shape():
    student = VisionStudent(act_dim=3, image_size=64)
    images = torch.randn(4, 3, 64, 64)
    out = student(images)
    assert out.shape == (4, 3)


def test_distill_produces_loadable_checkpoint(tmp_path):
    obs_dim, act_dim = 5, 2
    teacher = ActorCriticMLP(PolicyMeta(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=(16, 16)))
    student = VisionStudent(act_dim=act_dim, image_size=32)
    render_fn = make_synthetic_render_fn(obs_dim=obs_dim, act_dim=act_dim, image_size=32)

    out_path = tmp_path / "student.pt"
    result_path = distill(teacher, student, render_fn, steps=5, batch_size=4, out_path=out_path)

    assert result_path == out_path
    assert out_path.exists()
    checkpoint = torch.load(out_path, weights_only=False)
    assert checkpoint["act_dim"] == act_dim
    assert checkpoint["image_size"] == 32
