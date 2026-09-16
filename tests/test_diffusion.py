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


def test_v_target_and_inverse_are_consistent():
    """v-prediction must recover exactly the x0 and eps that produced xt."""
    d = GaussianDiffusion(timesteps=1000, parameterization="v")
    torch.manual_seed(0)
    x0 = torch.randn(4, 2, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.tensor([10, 300, 700, 999])
    xt = d.q_sample(x0, t, eps)
    v = d.v_target(x0, t, eps)
    eps_hat, x0_hat = d.to_eps_x0(v, xt, t)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
    assert torch.allclose(eps_hat, eps, atol=1e-4)


def test_eps_parameterization_is_unchanged():
    d = GaussianDiffusion(timesteps=1000)
    assert d.parameterization == "eps"
    torch.manual_seed(0)
    x0, t = torch.randn(2, 2, 8, 8), torch.tensor([100, 800])
    eps = torch.randn_like(x0)
    xt = d.q_sample(x0, t, eps)
    assert torch.equal(d.target(x0, t, eps), eps)
    e, x = d.to_eps_x0(eps, xt, t)
    assert torch.equal(e, eps) and torch.allclose(x, d.pred_x0(xt, t, eps))


def test_terminal_step_uses_data_mean_instead_of_amplified_eps_error():
    """At t=999 alpha_bar is 2.4e-9: pred_x0 multiplies eps error by ~2e4.

    A zero-eps model makes that error equal to xt itself, so the historical
    sampler turns pure noise into a +/-4 clamp pattern that survives to the end.
    With terminal_x0="mean" the zero-SNR step uses the data mean (0) instead.
    """
    def zero_eps(xt, t, cond=None):
        return torch.zeros_like(xt)

    shape, device = (2, 2, 8, 8), torch.device("cpu")
    torch.manual_seed(0)
    old = GaussianDiffusion(1000).ddim_sample(zero_eps, shape, device, steps=2)
    torch.manual_seed(0)
    new = GaussianDiffusion(1000, terminal_x0="mean").ddim_sample(
        zero_eps, shape, device, steps=2)
    assert old.abs().min() > 3.9                  # every pixel saturated
    assert new.abs().max() < 0.1                  # no injected clamp noise
    torch.manual_seed(0)
    anc = GaussianDiffusion(20, terminal_x0="mean").p_sample_loop(
        zero_eps, shape, device)
    assert torch.isfinite(anc).all()


def test_schedule_shift_divides_snr_by_shift_squared():
    """Resolution shift (Hoogeboom et al. 2023; SD3): SNR'(t) = SNR(t) / s^2."""
    base, shifted = GaussianDiffusion(1000), GaussianDiffusion(1000, schedule_shift=4.0)
    assert torch.equal(GaussianDiffusion(1000, schedule_shift=1.0).alphas_bar,
                       base.alphas_bar)
    t = torch.tensor([10, 300, 600, 900])
    assert torch.allclose(shifted.snr(t), base.snr(t) / 16.0, rtol=1e-3)
    assert (shifted.alphas_bar[1:] <= shifted.alphas_bar[:-1]).all()
    with pytest.raises(ValueError):
        GaussianDiffusion(10, schedule_shift=0.0)


def test_terminal_x0_rejects_unknown_modes():
    with pytest.raises(ValueError):
        GaussianDiffusion(10, terminal_x0="zero")


def test_min_snr_caps_the_easy_low_noise_steps():
    d = GaussianDiffusion(timesteps=1000)
    t = torch.tensor([0, 500, 999])
    flat = d.objective_weights(t, "none")
    capped = d.objective_weights(t, "min_snr", gamma=5.0)
    assert torch.allclose(flat, torch.ones(3))
    assert capped[0] < 0.01           # t=0 has enormous SNR, weight ~gamma/SNR
    assert capped[-1] == pytest.approx(1.0, abs=1e-3)   # t=999: SNR << gamma
    assert (capped[:-1] <= capped[1:]).all()          # non-decreasing in t
    assert capped[1] == pytest.approx(1.0, abs=1e-3)  # SNR(500) < gamma: uncapped
    with pytest.raises(ValueError):
        d.objective_weights(t, "bogus")


def test_diffusion_from_config_honours_every_sampling_setting():
    """Callers must not rebuild GaussianDiffusion by hand: the old copies in
    sample.generate silently dropped schedule_shift and terminal_x0."""
    from src.diffusion import diffusion_from_config
    d = diffusion_from_config({"timesteps": 1000, "parameterization": "v",
                               "schedule_shift": 4.0, "x0_clamp": "off",
                               "terminal_x0": "mean"})
    assert (d.parameterization, d.schedule_shift, d.x0_clamp, d.terminal_x0) == \
        ("v", 4.0, None, "mean")
    plain = diffusion_from_config({"timesteps": 1000})
    assert (plain.parameterization, plain.schedule_shift, plain.x0_clamp, plain.terminal_x0) == \
        ("eps", 1.0, (-4.0, 4.0), "model")
