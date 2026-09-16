"""Experimental radar-specific objectives for controlled ablation studies.

These are research candidates, not established improvements. They are kept
separate from the production training objective until generation experiments
show a benefit.
"""
import math

import torch
import torch.nn.functional as F

from src.losses import _windows


def _masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _target_mask(batch, device):
    n = batch["n_targets"].to(device)
    m = batch["traj"].shape[1]
    return (torch.arange(m, device=device)[None] < n[:, None]).float()


def _local_centroid(x, traj, half=2, temperature=0.25):
    win, origin = _windows(x, traj, half)
    size = 2 * half + 1
    weights = torch.softmax(win.reshape(*win.shape[:3], -1) / temperature,
                            dim=-1).reshape_as(win)
    idx = torch.arange(size, device=x.device, dtype=x.dtype)
    row = (weights.sum(-1) * idx).sum(-1)
    col = (weights.sum(-2) * idx).sum(-1)
    return origin + torch.stack((row, col), dim=-1)


def range_doppler_residual(x, batch, frame_interval=0.5, range_bin_m=3.0,
                           doppler_bin_mps=0.2496006389776358):
    """Squared residual of Δrange/Δtime = mean Doppler velocity.

    Uses soft target centroids within ground-truth windows. This admits a flat
    map as a degenerate solution; pair it with target_support_loss.
    """
    traj = batch["traj"].to(x.device)
    mask = _target_mask(batch, x.device)
    pos = _local_centroid(x, traj)
    rng = pos[..., 0] * range_bin_m
    vel = (pos[..., 1] - 32.0) * doppler_bin_mps
    residual = (rng[..., 1:] - rng[..., :-1]) / frame_interval
    residual -= (vel[..., 1:] + vel[..., :-1]) / 2
    return _masked_mean(residual.square().mean(-1), mask)


def target_support_loss(x_pred, x_true, batch, inner=1, outer=4):
    """Require at least the target/background contrast visible in real data.

    The contrast margin comes from each example, so a weak target under
    clutter is not forced to be brighter than it actually is. Constant maps
    cannot satisfy this term when a target is visible in the reference.
    """
    traj = batch["traj"].to(x_pred.device)
    mask = _target_mask(batch, x_pred.device)
    pred, _ = _windows(x_pred, traj, outer)
    true, _ = _windows(x_true.to(x_pred.device), traj, outer)
    radius = torch.arange(-outer, outer + 1, device=x_pred.device)
    rr, cc = torch.meshgrid(radius, radius, indexing="ij")
    foreground = (rr.abs() <= inner) & (cc.abs() <= inner)
    background = (rr.abs() > inner) | (cc.abs() > inner)

    def contrast(win):
        peak = win[..., foreground].amax(-1)
        back = win[..., background].mean(-1)
        return peak - back

    needed = contrast(true).detach().clamp(min=0, max=4)
    deficit = F.relu(needed - contrast(pred))
    return _masked_mean(deficit.square().mean(-1), mask)


def soft_cfar_map(x_db, pfa=1e-3, num_train=4, num_guard=2,
                  temperature=0.5):
    """Differentiable CA-CFAR detection probability on dB-magnitude maps.

    Convert amplitude dB to linear power before averaging training cells.
    The existing hard detector is defined on linear power; applying it to dB
    values would be physically incorrect.
    """
    if x_db.ndim != 4:
        raise ValueError("x_db must have shape (B,L,N,K)")
    if not 0 < pfa < 1 or temperature <= 0:
        raise ValueError("pfa and temperature must be positive, pfa < 1")
    pad = num_train + num_guard
    width = 2 * pad + 1
    guard = 2 * num_guard + 1
    n_ref = width * width - guard * guard
    alpha = n_ref * (pfa ** (-1 / n_ref) - 1)
    kernel = torch.ones((width, width), device=x_db.device, dtype=x_db.dtype)
    kernel[pad - num_guard:pad + num_guard + 1,
           pad - num_guard:pad + num_guard + 1] = 0
    power = torch.exp(x_db.clamp(max=100) * (math.log(10) / 10))
    B, L, H, W = x_db.shape
    ref_sum = F.conv2d(power.reshape(B * L, 1, H, W),
                       kernel[None, None], padding=pad)
    ref_count = F.conv2d(torch.ones_like(power).reshape(B * L, 1, H, W),
                         kernel[None, None], padding=pad)
    ref_mean = ref_sum / ref_count.clamp_min(1)
    log_ratio = (torch.log(power.reshape(B * L, 1, H, W).clamp_min(1e-20))
                 - torch.log(ref_mean.clamp_min(1e-20)) - math.log(alpha))
    return torch.sigmoid(log_ratio / temperature).reshape(B, L, H, W)


def cfar_curve_loss(x_pred_db, x_true_db, pfas=(1e-2, 1e-3, 1e-4)):
    """Match detector occupancy and spatiotemporal moments across Pfa values.

    Correlated heavy-tailed clutter violates CA-CFAR's exponential-noise
    assumption. The target is the *empirical* curve on real/reference maps.
    """
    def moments(det):
        return torch.stack((det.mean(), det.square().mean(),
                            (det[:, 1:] * det[:, :-1]).mean(),
                            (det[..., 1:] * det[..., :-1]).mean()))

    losses = []
    for pfa in pfas:
        pred = moments(soft_cfar_map(x_pred_db, pfa))
        with torch.no_grad():
            true = moments(soft_cfar_map(x_true_db, pfa))
        losses.append((pred - true).square().mean())
    return torch.stack(losses).mean()
