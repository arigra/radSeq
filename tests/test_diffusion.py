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
