"""Training losses: base DDPM loss + physics regularizers (spec §3)."""
import torch


def diffusion_loss(eps, eps_hat):
    return ((eps - eps_hat) ** 2).mean()


def smooth_loss(x0_hat, weight):
    """Temporal smoothness on predicted clean frames, weighted by
    omega_t = 1 - alphas_bar[t] (heavier late in the reverse process)."""
    diff = x0_hat[:, 1:] - x0_hat[:, :-1]                 # (B, L-1, N, K)
    per_seq = diff.pow(2).mean(dim=(1, 2, 3))             # (B,)
    return (weight * per_seq).mean()


def _windows(x0_hat, traj, half):
    """Gather (2*half+1)^2 windows around GT bins with edge clamping.
    x0_hat: (B, L, N, K); traj: (B, M, L, 2) -> windows (B, M, L, w, w),
    plus integer window origins (B, M, L, 2)."""
    B, L, N, K = x0_hat.shape
    M = traj.shape[1]
    w = 2 * half + 1
    r0 = traj[..., 0].round().long().clamp(half, N - 1 - half) - half   # (B, M, L)
    c0 = traj[..., 1].round().long().clamp(half, K - 1 - half) - half
    dr = torch.arange(w, device=x0_hat.device)
    rows = (r0[..., None] + dr).clamp(0, N - 1)                        # (B, M, L, w)
    cols = (c0[..., None] + dr).clamp(0, K - 1)
    bi = torch.arange(B, device=x0_hat.device)[:, None, None, None, None]
    li = torch.arange(L, device=x0_hat.device)[None, None, :, None, None]
    win = x0_hat[bi, li, rows[..., :, None], cols[..., None, :]]       # (B, M, L, w, w)
    return win, torch.stack([r0, c0], dim=-1).float()


def soft_positions(x0_hat, traj, half=2, temp=1.0):
    win, origin = _windows(x0_hat, traj, half)
    B, M, L, w, _ = win.shape
    p = torch.softmax(win.reshape(B, M, L, -1) / temp, dim=-1).reshape(B, M, L, w, w)
    idx = torch.arange(w, device=x0_hat.device).float()
    er = (p.sum(dim=-1) * idx).sum(dim=-1)                             # (B, M, L)
    ec = (p.sum(dim=-2) * idx).sum(dim=-1)
    return origin + torch.stack([er, ec], dim=-1)


def doppler_centroids(x0_hat, traj, half=2):
    win, origin = _windows(x0_hat, traj, half)
    w = win.shape[-1]
    inten = torch.relu(win - win.amin(dim=(-2, -1), keepdim=True)) + 1e-8
    idx = torch.arange(w, device=x0_hat.device).float()
    ec = (inten.sum(dim=-2) * idx).sum(dim=-1) / inten.sum(dim=(-2, -1))
    return origin[..., 1] + ec                                          # (B, M, L)


def traj_consistency_loss(pos, mask):
    """pos: (B, M, L, 2); mask: (B, M) 1 for real targets."""
    acc = pos[:, :, 2:] - 2 * pos[:, :, 1:-1] + pos[:, :, :-2]
    per = acc.pow(2).sum(dim=-1).mean(dim=-1)                           # (B, M)
    return (per * mask).sum() / mask.sum().clamp(min=1)


def doppler_consistency_loss(cent, mask):
    """cent: (B, M, L); mask: (B, M)."""
    d = cent[:, :, 1:] - cent[:, :, :-1]
    per = d.pow(2).mean(dim=-1)
    return (per * mask).sum() / mask.sum().clamp(min=1)


def traj_loss_from_batch(x0_hat, batch, tr_cfg, device):
    traj = batch["traj"].to(device)
    n = batch["n_targets"].to(device)
    mask = (torch.arange(traj.shape[1], device=device)[None] < n[:, None]).float()
    pos = soft_positions(x0_hat, traj)
    cent = doppler_centroids(x0_hat, traj)
    return (tr_cfg["lambda_traj"] * traj_consistency_loss(pos, mask)
            + tr_cfg["lambda_doppler"] * doppler_consistency_loss(cent, mask))
