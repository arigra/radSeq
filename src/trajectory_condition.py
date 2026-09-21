"""Trajectory condition maps for the conditional DiT.

A requested scene is a list of targets, each with a (range, Doppler) bin per
frame and a class. It is drawn as 4 channels per frame, aligned with the RD
map: a Gaussian blob per target in its class's channel (0 steady,
1 Swerling-1, 2 extended), and a presence channel of ones meaning "a
condition was given". All-zero channels are the "no condition" input used
for classifier-free guidance.
"""
import torch
import torch.nn.functional as F

N_CLASSES = 3
COND_CHANNELS = N_CLASSES + 1


def render_condition(traj, n_targets, cls, n_range=64, n_doppler=64, sigma=1.0):
    """traj (B, M, L, 2) bins [range, Doppler]; n_targets (B,); cls (B, M).

    Returns (B, L, 4, n_range, n_doppler) with values in [0, 1].
    """
    B, M, L, _ = traj.shape
    device = traj.device
    r = torch.arange(n_range, device=device, dtype=torch.float32).view(1, 1, 1, n_range, 1)
    d = torch.arange(n_doppler, device=device, dtype=torch.float32).view(1, 1, 1, 1, n_doppler)
    tr = traj[..., 0].float().view(B, M, L, 1, 1)
    td = traj[..., 1].float().view(B, M, L, 1, 1)
    blobs = torch.exp(-((r - tr) ** 2 + (d - td) ** 2) / (2 * sigma ** 2))  # (B, M, L, N, K)
    real = torch.arange(M, device=device)[None] < n_targets.to(device)[:, None]  # (B, M)
    onehot = F.one_hot(cls.to(device).long(), N_CLASSES).float()            # (B, M, 3)
    weight = (onehot * real[..., None]).view(B, M, N_CLASSES, 1, 1, 1)
    per_class = (blobs[:, :, None] * weight).amax(dim=1)                   # (B, 3, L, N, K)
    presence = torch.ones(B, 1, L, n_range, n_doppler, device=device)
    return torch.cat([per_class, presence], dim=1).permute(0, 2, 1, 3, 4).contiguous()


def drop_condition(cond, p, generator=None):
    """Zero the whole condition (all channels, all frames) for a fraction p of sequences."""
    if p <= 0:
        return cond
    keep = torch.rand(cond.shape[0], device=cond.device, generator=generator) >= p
    return cond * keep.view(-1, 1, 1, 1, 1).to(cond.dtype)


# RADIal labels give vehicle positions only (no classes), so the condition is
# one blob channel plus the presence plane that classifier-free guidance uses to
# tell "no vehicles" apart from "condition dropped".
VEHICLE_COND_CHANNELS = 2


def render_vehicle_condition(traj, present, n_range=512, n_doppler=256, sigma=1.0):
    """traj (B, M, L, 2) [range, Doppler] bins; present (B, M, L) bool.

    Returns (B, L, 2, n_range, n_doppler). The Doppler distance wraps, because
    the Doppler axis is circular (unshifted FFT order: bin 255 neighbours 0).
    """
    B, M, L, _ = traj.shape
    device = traj.device
    r = torch.arange(n_range, device=device, dtype=torch.float32).view(1, 1, 1, n_range, 1)
    d = torch.arange(n_doppler, device=device, dtype=torch.float32).view(1, 1, 1, 1, n_doppler)
    tr = traj[..., 0].float().view(B, M, L, 1, 1)
    td = traj[..., 1].float().view(B, M, L, 1, 1)
    dd = torch.remainder(d - td + n_doppler / 2, n_doppler) - n_doppler / 2
    blobs = torch.exp(-((r - tr) ** 2 + dd ** 2) / (2 * sigma ** 2))
    blobs = blobs * present.to(device).view(B, M, L, 1, 1).float()
    vehicles = blobs.amax(dim=1, keepdim=True) if M else torch.zeros(
        B, 1, L, n_range, n_doppler, device=device)
    presence = torch.ones(B, 1, L, n_range, n_doppler, device=device)
    return torch.cat([vehicles, presence], dim=1).permute(0, 2, 1, 3, 4).contiguous()
