"""Temporal radar simulator: single-frame RD core (copied from RDDiffusion)
plus a TemporalRadarSimulator that generates kinematically consistent
multi-frame sequences of moving targets.
"""

import math

import torch

# ---------------------------------------------------------------------------
# Module-level matrix caches.  All of these are pure functions of fixed radar
# constants (N=64, K=64, B, fc, T0, c) so they are identical for every
# RadarDataset instance and every worker process.  They are computed once on
# first use and reused for the lifetime of the process (~200 KB total).
# ---------------------------------------------------------------------------
_RD_R: torch.Tensor | None = None   # range steering for RD map  (N, dR) complex
_RD_V: torch.Tensor | None = None   # Doppler steering for RD map (K, dV) complex
_PQ_DIFF: torch.Tensor | None = None          # (p-q) matrix (N, K) float
_CLUTTER_R_STEER: torch.Tensor | None = None  # range steering for clutter (N, dR) complex


def _get_rd_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    global _RD_R, _RD_V
    if _RD_R is None:
        _RD_R = generate_range_steering_matrix()
        _RD_V = generate_doppler_steering_matrix()
    return _RD_R, _RD_V


def _get_pq_diff(N: int, K: int) -> torch.Tensor:
    global _PQ_DIFF
    if _PQ_DIFF is None:
        p, q = torch.meshgrid(
            torch.arange(N, dtype=torch.float),
            torch.arange(K, dtype=torch.float),
            indexing="ij",
        )
        _PQ_DIFF = p - q
    return _PQ_DIFF


def _get_clutter_range_steer(N: int, R: torch.Tensor, B: float, c: float) -> torch.Tensor:
    global _CLUTTER_R_STEER
    if _CLUTTER_R_STEER is None:
        _CLUTTER_R_STEER = torch.exp(
            -1j * 2 * math.pi
            * torch.outer(torch.arange(N, dtype=torch.float), R)
            * (2 * B) / (c * N)
        )
    return _CLUTTER_R_STEER


# =====================================================================
#                       RD-domain helpers (unchanged)
# =====================================================================
def generate_range_steering_matrix(N=64, dR=64, B=50e6, c=3e8):
    rng_res = c / (2 * B)
    r_vals = torch.arange(dR) * rng_res
    n_vals = torch.arange(N)
    phase = -1j * 2 * math.pi * (2 * B) / (c * N)
    R = torch.exp(phase * torch.outer(n_vals, r_vals))
    return R


def generate_doppler_steering_matrix(K=64, dV=64, fc=9.39e9, T0=1e-3, c=3e8):
    vel_res = c / (2 * fc * K * T0)
    # linspace matches dataset.py exactly — same grid used for label V bins.
    v_vals = torch.arange(-dV // 2, dV // 2) * vel_res
    k_vals = torch.arange(K)
    phase = -1j * 2 * math.pi * (2 * fc * T0) / c
    V = torch.exp(phase * torch.outer(k_vals, v_vals))
    return V


def create_rd_map(IQ_map):
    if not torch.is_tensor(IQ_map):
        IQ_map = torch.from_numpy(IQ_map)
    if not torch.is_complex(IQ_map):
        IQ_map = IQ_map.to(torch.complex64)
    device = IQ_map.device
    R, V = _get_rd_matrices()
    RD_map = R.T.conj().to(device) @ IQ_map @ V.conj().to(device)
    return RD_map


# =====================================================================
#                    Temporal simulator (new for this task)
# =====================================================================
class TemporalRadarSimulator:
    """L-frame RD sequence generator with kinematically moving targets.

    Frame ell (0-indexed): r_l = r0 + v0*l*Tf + 0.5*a*(l*Tf)**2,
                           v_l = v0 + a*l*Tf.
    Sequences are rejection-sampled so every frame stays inside the RD grid.
    Clutter is added by Task 3 (AR(1) evolution); here C=0.
    """

    N = K = 64
    B_HZ = 50e6
    T0 = 1e-3
    FC = 9.39e9
    C_LIGHT = 3e8
    CNR_DB = 15.0

    def __init__(self, seq_len=16, frame_interval=0.5, max_targets=5,
                 rho_clutter=None, scnr=None, nu=None, clutter_mode="strip",
                 force_class=None):
        self.L = seq_len
        self.Tf = frame_interval
        self.max_targets = max_targets
        self.rho_clutter = rho_clutter
        self.scnr = scnr
        self.nu = nu
        self.clutter_mode = clutter_mode
        self.force_class = force_class

        self.r_min, self.r_max, self.dr = 0.0, 189.0, 3.0
        # Doppler grid MUST match generate_doppler_steering_matrix exactly:
        # vel_res = c/(2*fc*K*T0), bins arange(-K/2, K/2)*vel_res. Using the
        # rounded (-7.8, 0.249) grid biases traj labels by up to ~1 bin.
        self.dv = self.C_LIGHT / (2 * self.FC * self.K * self.T0)
        self.v_min = -(self.K // 2) * self.dv
        self.v_max = (self.K // 2 - 1) * self.dv
        self.R = torch.arange(self.r_min, self.r_max + self.dr, self.dr)
        self.V = torch.arange(-(self.K // 2), self.K // 2).float() * self.dv
        self.dR, self.dV = len(self.R), len(self.V)
        self.a_max = 0.5  # m/s^2 -> <=1 Doppler bin per frame at Tf=0.5

        self.sigma2 = self.N / (2 * 10 ** (self.CNR_DB / 10))
        self.cn_norm = torch.sqrt(torch.tensor(
            self.N * self.K * (self.N // 2 + self.sigma2), dtype=torch.float))

    # ---------------- target kinematics ----------------
    def _sample_kinematics(self, n):
        """Rejection-sample (r0, v0, a) per target s.t. all frames in-grid."""
        ell = torch.arange(self.L, dtype=torch.float) * self.Tf
        for _ in range(500):
            r0 = torch.empty(n).uniform_(self.r_min + 10, self.r_max - 10)
            v0 = torch.empty(n).uniform_(self.v_min + 0.5, self.v_max - 0.5)
            a = torch.empty(n).uniform_(-self.a_max, self.a_max)
            r = r0[:, None] + v0[:, None] * ell + 0.5 * a[:, None] * ell ** 2
            v = v0[:, None] + a[:, None] * ell
            ok = ((r >= self.r_min) & (r <= self.r_max)
                  & (v >= self.v_min) & (v <= self.v_max)).all()
            if ok:
                return r0, v0, a, r, v  # r, v: (n, L)
        raise RuntimeError("kinematics rejection sampling failed")

    def _to_bins(self, r, v):
        """Continuous bin coordinates for (n, L) range/velocity arrays."""
        rb = (r - self.r_min) / self.dr
        vb = (v - self.v_min) / self.dv
        return torch.stack([rb, vb], dim=-1)  # (n, L, 2)

    # ---------------- per-frame target signal ----------------
    def _target_iq(self, ranges, velocities, gains_dB):
        """Sum-of-targets IQ frame, amplitudes set by per-target SCNR in dB.
        ranges/velocities/gains_dB: (n,) for one frame."""
        n = len(ranges)
        w_r = (2 * torch.pi * 2 * self.B_HZ * ranges) / (self.C_LIGHT * self.N)
        rs = torch.exp(-1j * torch.outer(w_r, torch.arange(self.N, dtype=torch.float)))
        w_d = (2 * torch.pi * self.T0 * 2 * self.FC * velocities) / self.C_LIGHT
        ds = torch.exp(-1j * torch.outer(w_d, torch.arange(self.K, dtype=torch.float)))
        sig = rs.unsqueeze(-1) * ds.unsqueeze(1)                       # (n, N, K)
        phases = torch.empty(n, 1, 1).uniform_(0, 2 * torch.pi)
        sig = sig * torch.exp(1j * phases)
        s_norm = torch.linalg.norm(sig, dim=(1, 2)).real
        amp = (10 ** (gains_dB / 20)) * (self.cn_norm / s_norm)
        return (amp.view(-1, 1, 1) * sig).sum(dim=0)                   # (N, K)

    def _frame_targets(self, r_l, v_l, base_gain_dB, cls):
        """Assemble one frame's target IQ honoring class semantics.
        r_l, v_l, base_gain_dB, cls: (n,) tensors for frame l."""
        ranges, vels, gains = [], [], []
        for i in range(len(r_l)):
            g = base_gain_dB[i].clone()
            if cls[i] == 1:  # Swerling-1: per-frame exponential power
                g = g + 10 * torch.log10(-torch.log(torch.rand(1) + 1e-12)).squeeze()
            if cls[i] == 2:  # range-extended: 3 scatterers, -3 dB flanks
                for dr_m, dg in ((-self.dr, -3.0), (0.0, 0.0), (self.dr, -3.0)):
                    ranges.append(torch.clamp(r_l[i] + dr_m, self.r_min, self.r_max))
                    vels.append(v_l[i]); gains.append(g + dg)
            else:
                ranges.append(r_l[i]); vels.append(v_l[i]); gains.append(g)
        return self._target_iq(torch.stack(ranges), torch.stack(vels),
                               torch.stack(gains))

    # ---------------- clutter (AR(1) SIRP, Task 3) ----------------
    def _sample_texture(self, nu):
        """K-distribution texture: Gamma(nu, nu), E[s]=1, shape (dR,)."""
        nu_t = torch.tensor(float(nu))
        return torch.distributions.Gamma(nu_t, nu_t).sample((self.dR,)).view(self.dR)

    def _clutter_frames(self, rho, nu, sigma_f=0.05):
        """Strip clutter with AR(1) speckle evolution across frames.

        SIRP skeleton per RDDiffusion: per-frame speckle w = A @ z with
        A = V sqrt(E) from eigh of the Doppler covariance M; the Gaussian
        innovations z evolve as z_l = rho*z_{l-1} + sqrt(1-rho^2)*eps_l,
        so consecutive frames share correlated speckle. Texture s and the
        clutter Doppler velocity are fixed for the whole sequence.
        """
        clutter_vel = torch.empty(1).uniform_(self.v_min, self.v_max)
        fd = 2 * torch.pi * (2 * self.FC * clutter_vel) / self.C_LIGHT
        pq = _get_pq_diff(self.N, self.K)
        M = torch.exp(-2 * torch.pi ** 2 * sigma_f ** 2 * pq ** 2
                      - 1j * pq * fd * self.T0)
        e, Vm = torch.linalg.eigh(M)
        A = Vm @ torch.diag(torch.sqrt(torch.clamp(e.real, min=0.0))).to(Vm.dtype)
        steer = _get_clutter_range_steer(self.N, self.R, self.B_HZ, self.C_LIGHT)
        s = torch.clamp(self._sample_texture(nu), min=0.0)             # (dR,)

        rho_t = torch.tensor(float(rho))
        z = torch.randn(self.K, self.dR, dtype=torch.cfloat) / torch.sqrt(torch.tensor(2.0))
        frames = []
        for _ in range(self.L):
            w = A @ z                                                  # (K, dR)
            c_t = torch.sqrt(s).unsqueeze(0) * w
            frames.append(steer @ c_t.transpose(0, 1))                 # (N, K)
            eps = torch.randn(self.K, self.dR, dtype=torch.cfloat) / torch.sqrt(torch.tensor(2.0))
            z = rho_t * z + torch.sqrt(1 - rho_t ** 2) * eps
        return torch.stack(frames)

    # ---------------- sequence assembly ----------------
    def gen_sequence(self):
        n = int(torch.randint(1, self.max_targets + 1, (1,)).item())
        r0, v0, a, r, v = self._sample_kinematics(n)
        traj = self._to_bins(r, v)
        cls = (torch.randint(0, 3, (n,)) if self.force_class is None
               else torch.full((n,), int(self.force_class), dtype=torch.long))
        base_gain = (torch.empty(n).uniform_(-5, 10) if self.scnr is None
                     else torch.full((n,), float(self.scnr)))
        rho = (float(torch.rand(1).item()) if self.rho_clutter is None
               else float(self.rho_clutter))
        nu = (float(torch.empty(1).uniform_(0.1, 1.5).item()) if self.nu is None
              else float(self.nu))

        C = self._clutter_frames(rho, nu)
        frames, s_energy, cn_energy = [], 0.0, 0.0
        for l in range(self.L):
            S = self._frame_targets(r[:, l], v[:, l], base_gain, cls)
            W = (torch.randn(self.N, self.K, dtype=torch.cfloat)
                 / torch.sqrt(torch.tensor(2.0 * self.sigma2)))
            X = S + C[l] + W
            s_energy += S.abs().pow(2).sum().item()
            cn_energy += (C[l] + W).abs().pow(2).sum().item()
            rd = create_rd_map(X)
            frames.append(20 * torch.log10(rd.abs() + 1e-6))
        scnr_dB = 10 * torch.log10(torch.tensor(s_energy / (cn_energy + 1e-12)))
        return {
            "x": torch.stack(frames).float(),
            "traj": traj.float(),
            "v0": v0.float(), "acc": a.float(),
            "cls": cls.long(),
            "env": torch.tensor([self.CNR_DB, scnr_dB.item(), rho]).float(),
            "n_targets": n,
        }
