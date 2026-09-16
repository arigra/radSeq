import torch

from src.research_losses import (cfar_curve_loss, range_doppler_residual,
                                 soft_cfar_map, target_support_loss)
from src.simulator import generate_sequences


def test_acceleration_aware_loss_rejects_false_motion_and_flat_maps():
    batch = generate_sequences(n=4, n_targets=1, target_class="steady",
                               snr_db=20, clutter=False, noise=False, seed=4)
    x = (batch["x"] - 35.1472) / 15.6529
    shifted = x.roll(4, dims=2)
    assert range_doppler_residual(x, batch) < range_doppler_residual(shifted, batch)
    assert target_support_loss(x, x, batch) == 0
    assert target_support_loss(torch.zeros_like(x), x, batch) > 1
    pred = shifted.clone().requires_grad_()
    loss = range_doppler_residual(pred, batch) + target_support_loss(pred, x, batch)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_soft_cfar_responds_to_clutter_corruption():
    x = generate_sequences(n=2, seq_len=4, seed=5)["x"]
    blurred = torch.nn.functional.avg_pool2d(
        x.flatten(0, 1)[:, None], 5, 1, 2)[:, 0].reshape_as(x)
    assert soft_cfar_map(x).shape == x.shape
    assert cfar_curve_loss(x, x) == 0
    assert cfar_curve_loss(blurred, x) > 0
    pred = blurred.clone().requires_grad_()
    cfar_curve_loss(pred, x).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
