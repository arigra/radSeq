import pytest
import torch
from src.diffusion import GaussianDiffusion


def test_schedule_monotone():
    d = GaussianDiffusion(timesteps=1000)
    ab = d.alphas_bar
    assert ab.shape == (1000,)
    assert (ab[1:] <= ab[:-1] + 1e-8).all()
    assert ab[0] > 0.99 and ab[-1] < 0.01


def test_q_sample_pred_x0_roundtrip():
    torch.manual_seed(0)
    d = GaussianDiffusion(timesteps=1000)
    x0 = torch.randn(2, 16, 64, 64)
    t = torch.tensor([100, 900])
    eps = torch.randn_like(x0)
    xt = d.q_sample(x0, t, eps)
    x0_hat = d.pred_x0(xt, t, eps)
    assert torch.allclose(x0, x0_hat, atol=1e-4)


def test_ddim_sample_shape_and_finite():
    torch.manual_seed(0)
    d = GaussianDiffusion(timesteps=1000)

    def dummy_model(xt, t, cond=None):
        return torch.zeros_like(xt)

    out = d.ddim_sample(dummy_model, (1, 16, 64, 64), torch.device("cpu"), steps=10)
    assert out.shape == (1, 16, 64, 64)
    assert torch.isfinite(out).all()


def test_p_sample_loop_shape_and_finite():
    torch.manual_seed(0)
    d = GaussianDiffusion(timesteps=50)

    def dummy_model(xt, t, cond=None):
        return torch.zeros_like(xt)

    out = d.p_sample_loop(dummy_model, (1, 4, 16, 16), torch.device("cpu"))
    assert out.shape == (1, 4, 16, 16)
    assert torch.isfinite(out).all()


def test_x0_clamp_defaults_to_the_historical_guard_rail():
    assert GaussianDiffusion(timesteps=10).x0_clamp == (-4.0, 4.0)


def test_x0_clamp_can_be_disabled_or_widened():
    big = torch.tensor([[-9.0, 9.0]])
    assert torch.equal(
        GaussianDiffusion(10, x0_clamp=None).clamp_x0(big), big)
    assert GaussianDiffusion(10).clamp_x0(big).abs().max().item() == 4.0
    assert GaussianDiffusion(10, x0_clamp=8).clamp_x0(big).abs().max().item() == 8.0


def test_parse_x0_clamp_forms():
    from src.diffusion import parse_x0_clamp
    assert parse_x0_clamp(None) is None
    assert parse_x0_clamp("off") is None
    assert parse_x0_clamp(6) == (-6.0, 6.0)
    assert parse_x0_clamp([-3.0, 5.0]) == (-3.0, 5.0)
    with pytest.raises(ValueError):
        parse_x0_clamp([5.0, -3.0])
    with pytest.raises(ValueError):
        parse_x0_clamp("sometimes")


def test_smoothness_weight_modes_have_opposite_schedules():
    d = GaussianDiffusion(timesteps=1000)
    t = torch.tensor([0, 500, 999])
    high_noise_heavy = d.loss_weight(t, "one_minus_alpha_bar")
    low_noise_heavy = d.loss_weight(t, "alpha_bar")
    assert high_noise_heavy[0] < high_noise_heavy[-1]
    assert low_noise_heavy[0] > low_noise_heavy[-1]
    assert torch.allclose(d.loss_weight(t, "uniform"), torch.ones(3))
    assert torch.allclose(high_noise_heavy + low_noise_heavy, torch.ones(3), atol=1e-6)
    with pytest.raises(ValueError):
        d.loss_weight(t, "made_up")
