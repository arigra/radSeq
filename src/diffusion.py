"""DDPM with cosine schedule, epsilon-prediction; DDIM sampler for eval."""
import math

import torch


DEFAULT_X0_CLAMP = (-4.0, 4.0)


def parse_x0_clamp(value):
    """Normalize a config value into a (low, high) pair or None (no clamping).

    Accepts None / "off" / "none" / False to disable, a scalar v meaning
    (-v, v), or an explicit two-element sequence.
    """
    if value is None or value is False:
        return None
    if isinstance(value, str):
        if value.lower() in ("off", "none", "false"):
            return None
        raise ValueError(f"unknown x0_clamp {value!r}")
    if isinstance(value, (int, float)):
        v = float(abs(value))
        return (-v, v)
    low, high = (float(v) for v in value)
    if not low < high:
        raise ValueError(f"x0_clamp bounds must increase, got {value!r}")
    return (low, high)


def _bc(v, x):
    """Broadcast (B,) schedule values over x's trailing dims."""
    return v.view(-1, *([1] * (x.dim() - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps=1000, x0_clamp=DEFAULT_X0_CLAMP):
        self.T = timesteps
        self.x0_clamp = parse_x0_clamp(x0_clamp)
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

    def clamp_x0(self, x0):
        """Apply the configured x0 guard rail, or nothing when disabled."""
        if self.x0_clamp is None:
            return x0
        return x0.clamp(*self.x0_clamp)

    def loss_weight(self, t, mode="one_minus_alpha_bar"):
        """Per-sample weight for the auxiliary smoothness term.

        ``one_minus_alpha_bar`` (the original) peaks at high noise, which is
        where the timestep diagnostic located the synthesis failure.
        ``alpha_bar`` inverts it so the penalty applies while the model refines
        a scene it can already see; ``uniform`` removes the schedule entirely.
        """
        ab = self.alphas_bar.to(t.device)[t]
        if mode == "one_minus_alpha_bar":
            return 1 - ab
        if mode == "alpha_bar":
            return ab
        if mode == "uniform":
            return torch.ones_like(ab)
        raise ValueError(f"unknown smoothness weight mode {mode!r}")

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, steps=50, cond=None):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for i in range(steps):
            t = ts[i].repeat(shape[0])
            eps = model(x, t, cond)
            x0 = self.clamp_x0(self.pred_x0(x, t, eps))
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
            x0 = self.clamp_x0(self.pred_x0(x, t, eps))
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
