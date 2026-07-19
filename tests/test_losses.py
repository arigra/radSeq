import torch
from src.losses import diffusion_loss, smooth_loss


def test_diffusion_loss_zero_when_equal():
    e = torch.randn(2, 16, 64, 64)
    assert diffusion_loss(e, e).item() == 0.0
    assert diffusion_loss(e, torch.zeros_like(e)).item() > 0


def test_smooth_loss_zero_for_static():
    x = torch.randn(2, 1, 64, 64).repeat(1, 16, 1, 1)   # identical frames
    w = torch.ones(2)
    assert smooth_loss(x, w).item() < 1e-10


def test_smooth_loss_scales_with_weight():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64, 64)
    l1 = smooth_loss(x, torch.ones(2))
    l2 = smooth_loss(x, 2 * torch.ones(2))
    assert torch.allclose(l2, 2 * l1)


def test_smooth_loss_grad_flows():
    x = torch.randn(1, 16, 64, 64, requires_grad=True)
    smooth_loss(x, torch.ones(1)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
