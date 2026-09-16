"""DDPM with cosine schedule, epsilon-prediction; DDIM sampler for eval."""
import math

import torch


DEFAULT_X0_CLAMP = (-4.0, 4.0)
# Below this alpha_bar the signal is unrecoverable (t=999 has 2.4e-9) and
# pred_x0 amplifies eps error by 1/sqrt(alpha_bar); see
# docs/notes/2026-09-14-terminal-step-diagnosis.md.
ZERO_SNR_ALPHA_BAR = 1e-6


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
    def __init__(self, timesteps=1000, x0_clamp=DEFAULT_X0_CLAMP,
                 parameterization="eps", terminal_x0="model", schedule_shift=1.0):
        self.T = timesteps
        self.x0_clamp = parse_x0_clamp(x0_clamp)
        if terminal_x0 not in ("model", "mean"):
            raise ValueError(f"unknown terminal_x0 {terminal_x0!r}")
        self.terminal_x0 = terminal_x0
        if parameterization not in ("eps", "v"):
            raise ValueError(f"unknown parameterization {parameterization!r}")
        self.parameterization = parameterization
        t = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        abar = (f / f[0])
        betas = torch.clamp(1 - abar[1:] / abar[:-1], max=0.999)
        abar = torch.cumprod(1 - betas, dim=0)
        if schedule_shift <= 0:
            raise ValueError("schedule_shift must be positive")
        # Resolution shift (Hoogeboom et al. 2023; SD3): SNR' = SNR / s^2. Large,
        # redundant inputs reveal global structure at low SNR, so shift noise up.
        self.schedule_shift = float(schedule_shift)
        if schedule_shift != 1.0:
            abar = abar / (abar + schedule_shift ** 2 * (1 - abar))
        self.alphas_bar = abar.float()

    def _ab(self, t, x):
        return _bc(self.alphas_bar.to(x.device)[t], x)

    def q_sample(self, x0, t, eps):
        ab = self._ab(t, x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def pred_x0(self, xt, t, eps_hat):
        ab = self._ab(t, xt)
        return (xt - (1 - ab).sqrt() * eps_hat) / ab.sqrt()

    def v_target(self, x0, t, eps):
        """v = sqrt(abar) * eps - sqrt(1 - abar) * x0 (Salimans & Ho, 2022)."""
        ab = self._ab(t, x0)
        return ab.sqrt() * eps - (1 - ab).sqrt() * x0

    def target(self, x0, t, eps):
        """What the network is trained to regress under this parameterization."""
        return eps if self.parameterization == "eps" else self.v_target(x0, t, eps)

    def to_eps_x0(self, out, xt, t):
        """Map a raw network output to (eps_hat, x0_hat), unclamped.

        Under eps-prediction the output is eps and x0 follows from pred_x0.
        Under v-prediction both follow from v by rotation, which is the point:
        at high noise the v target still carries the signal direction, where
        eps is nearly all noise and says little about where structure belongs.
        """
        if self.parameterization == "eps":
            return out, self.pred_x0(xt, t, out)
        ab = self._ab(t, xt)
        x0 = ab.sqrt() * xt - (1 - ab).sqrt() * out
        eps = (1 - ab).sqrt() * xt + ab.sqrt() * out
        return eps, x0

    def snr(self, t):
        ab = self.alphas_bar.to(t.device)[t]
        return ab / (1 - ab).clamp(min=1e-20)

    def objective_weights(self, t, mode="none", gamma=5.0):
        """Per-sample weight on the main regression loss.

        ``min_snr`` (Hang et al., 2023) caps the effective SNR at gamma so the
        low-noise steps, where the task is nearly trivial, stop dominating the
        gradient. The eps and v forms differ by the +1 in the denominator.
        """
        if mode == "none":
            return torch.ones_like(self.alphas_bar.to(t.device)[t])
        if mode != "min_snr":
            raise ValueError(f"unknown loss weighting {mode!r}")
        if gamma <= 0:
            raise ValueError("min_snr_gamma must be positive")
        snr = self.snr(t)
        capped = snr.clamp(max=gamma)
        return capped / (snr if self.parameterization == "eps" else snr + 1)

    def clamp_x0(self, x0):
        """Apply the configured x0 guard rail, or nothing when disabled."""
        if self.x0_clamp is None:
            return x0
        return x0.clamp(*self.x0_clamp)

    def terminal_step(self, x, t, eps, x0):
        """Replace the network's estimate at zero-SNR steps when requested.

        ``model`` keeps the historical behaviour. ``mean`` uses the posterior
        mean at SNR ~ 0, which is the data mean (0 after normalisation), and
        the eps consistent with it; this stops the first DDIM step from
        injecting a saturated +/-clamp pattern into the trajectory.
        """
        if self.terminal_x0 == "model":
            return eps, x0
        ab = self._ab(t, x)
        zero_snr = ab < ZERO_SNR_ALPHA_BAR
        return (torch.where(zero_snr, x / (1 - ab).sqrt(), eps),
                torch.where(zero_snr, torch.zeros_like(x0), x0))

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
            eps, x0 = self.to_eps_x0(model(x, t, cond), x, t)
            eps, x0 = self.terminal_step(x, t, eps, self.clamp_x0(x0))
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
            eps, x0 = self.to_eps_x0(model(x, t, cond), x, t)
            _, x0 = self.terminal_step(x, t, eps, self.clamp_x0(x0))
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
