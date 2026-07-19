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
