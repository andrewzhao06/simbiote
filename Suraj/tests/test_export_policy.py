import numpy as np
import torch

from simbiote.training.export_policy import export_policy
from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta


def _make_and_save(tmp_path, obs_dim=6, act_dim=3):
    meta = PolicyMeta(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=(16, 16), act_low=(-1.0,) * act_dim, act_high=(1.0,) * act_dim)
    model = ActorCriticMLP(meta)
    path = tmp_path / "policy.pt"
    model.save(path)
    return model, path


def test_export_onnx_matches_torch_inference(tmp_path):
    import onnxruntime as ort

    model, ckpt_path = _make_and_save(tmp_path)
    onnx_path = export_policy(ckpt_path, tmp_path / "policy.onnx", fmt="onnx")
    assert onnx_path.exists()

    obs = np.random.RandomState(0).uniform(-1, 1, size=(5, 6)).astype(np.float32)
    with torch.no_grad():
        torch_action, _, _ = model.act(torch.as_tensor(obs), deterministic=True)

    session = ort.InferenceSession(str(onnx_path))
    onnx_action = session.run(None, {"observation": obs})[0]

    np.testing.assert_allclose(torch_action.numpy(), onnx_action, atol=1e-4)


def test_export_torchscript_matches_torch_inference(tmp_path):
    model, ckpt_path = _make_and_save(tmp_path)
    ts_path = export_policy(ckpt_path, tmp_path / "policy.ts", fmt="torchscript")
    assert ts_path.exists()

    obs = torch.randn(4, 6)
    with torch.no_grad():
        torch_action, _, _ = model.act(obs, deterministic=True)

    scripted = torch.jit.load(str(ts_path))
    with torch.no_grad():
        scripted_action = scripted(obs)

    torch.testing.assert_close(torch_action, scripted_action, atol=1e-5, rtol=1e-5)


def test_export_action_bounds_enforced(tmp_path):
    """The exported graph should clamp to act_low/act_high even for
    out-of-range obs that would otherwise push the mean past the bounds."""
    import onnxruntime as ort

    model, ckpt_path = _make_and_save(tmp_path, obs_dim=4, act_dim=2)
    onnx_path = export_policy(ckpt_path, tmp_path / "clamped.onnx", fmt="onnx")

    obs = (np.random.RandomState(1).uniform(-1, 1, size=(20, 4)) * 50).astype(np.float32)
    session = ort.InferenceSession(str(onnx_path))
    action = session.run(None, {"observation": obs})[0]

    assert np.all(action >= -1.0 - 1e-4)
    assert np.all(action <= 1.0 + 1e-4)
