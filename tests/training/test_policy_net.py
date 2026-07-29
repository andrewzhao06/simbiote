import torch

from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta


def make_model(obs_dim=6, act_dim=3):
    meta = PolicyMeta(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(32, 32),
        act_low=(-1,) * act_dim,
        act_high=(1,) * act_dim,
    )
    return ActorCriticMLP(meta)


def test_forward_shapes():
    model = make_model()
    obs = torch.randn(4, 6)
    mean, std, value = model.forward(obs)
    assert mean.shape == (4, 3)
    assert std.shape == (4, 3)
    assert value.shape == (4,)


def test_act_respects_bounds():
    model = make_model()
    obs = torch.randn(16, 6) * 10  # large obs -> possibly large raw action before clipping
    action, log_prob, value = model.act(obs, deterministic=False)
    assert action.shape == (16, 3)
    assert torch.all(action >= -1.0 - 1e-5)
    assert torch.all(action <= 1.0 + 1e-5)
    assert log_prob.shape == (16,)
    assert value.shape == (16,)


def test_act_deterministic_matches_mean_before_clip():
    model = make_model()
    obs = torch.zeros(1, 6)
    mean, _, _ = model.forward(obs)
    action, _, _ = model.act(obs, deterministic=True)
    expected = torch.clamp(mean, -1.0, 1.0)
    assert torch.allclose(action, expected, atol=1e-6)


def test_evaluate_actions_gradients_flow():
    model = make_model()
    obs = torch.randn(8, 6)
    actions = torch.randn(8, 3)
    log_prob, _entropy, value = model.evaluate_actions(obs, actions)
    loss = -log_prob.mean() + value.mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_save_and_load_roundtrip(tmp_path):
    model = make_model(obs_dim=5, act_dim=2)
    path = tmp_path / "policy.pt"
    model.save(path)

    loaded = ActorCriticMLP.load(path)
    assert loaded.meta.obs_dim == 5
    assert loaded.meta.act_dim == 2

    obs = torch.randn(3, 5)
    a1, _, _ = model.act(obs, deterministic=True)
    a2, _, _ = loaded.act(obs, deterministic=True)
    assert torch.allclose(a1, a2, atol=1e-6)


def test_warm_start_preserves_weights_when_reused():
    """bc_pretrain.py fine-tunes an existing policy_net in place -- confirm
    that loading it back out still reflects those (not fresh-random) weights."""
    model = make_model()
    before = model.actor_mean[0].weight.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    obs = torch.randn(32, 6)
    target = torch.randn(32, 3)
    for _ in range(5):
        pred = model.actor_mean(obs)
        loss = ((pred - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = model.actor_mean[0].weight
    assert not torch.allclose(before, after)
