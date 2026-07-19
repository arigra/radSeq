"""DDPM with cosine schedule, epsilon-prediction; DDIM sampler for eval."""
import math

import torch


def _bc(v, x):
    """Broadcast (B,) schedule values over x's trailing dims."""
    return v.view(-1, *([1] * (x.dim() - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps=1000):
        self.T = timesteps
        t = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        abar = (f / f[0])
        betas = torch.clamp(1 - abar[1:] / abar[:-1], max=0.999)
        self.alphas_bar = torch.cumprod(1 - betas, dim=0).float()

    def _ab(self, t, x):
        return _bc(self.alphas_bar.to(x.device)[t], x)

    def q_sample(self, x0, t, eps):
        ab = self._ab(t, x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def pred_x0(self, xt, t, eps_hat):
        ab = self._ab(t, xt)
        return (xt - (1 - ab).sqrt() * eps_hat) / ab.sqrt()

    def loss_weight(self, t):
        return 1 - self.alphas_bar.to(t.device)[t]

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, steps=50, cond=None):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for i in range(steps):
            t = ts[i].repeat(shape[0])
            eps = model(x, t, cond)
            x0 = self.pred_x0(x, t, eps).clamp(-4, 4)
            if i == steps - 1:
                x = x0
            else:
                ab_next = self._ab(ts[i + 1].repeat(shape[0]), x)
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, cond=None):
        ab = self.alphas_bar.to(device)
        x = torch.randn(shape, device=device)
        for ti in reversed(range(self.T)):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            eps = model(x, t, cond)
            x0 = self.pred_x0(x, t, eps).clamp(-4, 4)
            if ti == 0:
                x = x0
            else:
                ab_t, ab_prev = ab[ti], ab[ti - 1]
                beta_t = 1 - ab_t / ab_prev
                mean = (ab_prev.sqrt() * beta_t * x0
                        + (1 - beta_t).sqrt() * (1 - ab_prev) * x) / (1 - ab_t)
                var = beta_t * (1 - ab_prev) / (1 - ab_t)
                x = mean + var.sqrt() * torch.randn_like(x)
        return x
