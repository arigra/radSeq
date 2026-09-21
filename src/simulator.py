"""Temporal radar simulator: single-frame RD core (copied from RDDiffusion)
plus a TemporalRadarSimulator that generates kinematically consistent
multi-frame sequences of moving targets.
"""

import dataclasses
import math

import torch

# ---------------------------------------------------------------------------
# Module-level matrix caches.  All of these are pure functions of fixed radar
# constants (N=64, K=64, B, fc, T0, c) so they are identical for every
# RadarDataset instance and every worker process.  They are computed once on
# first use and reused for the lifetime of the process (~200 KB total).
# ---------------------------------------------------------------------------
_STEER_CACHE: dict = {}   # geometry key -> (range steering, Doppler steering)
_PQ_CACHE: dict = {}      # (N, K) -> (p - q) matrix
_CLUTTER_CACHE: dict = {}  # geometry key -> clutter range steering


@dataclasses.dataclass(frozen=True)
class RadarGeometry:
    """Radar constants defining the range-Doppler grid.

    The defaults are the constants this simulator shipped with; the whole
    cached training set was drawn with them, so they must not drift.
    `RADIAL_GEOMETRY` matches the measured RADIal grid so a model pretrained
    on simulated data sees the same metres and metres-per-second per pixel as
    the real recordings -- without that, pretraining transfers nothing.
    """

    N: int = 64          # range bins
    K: int = 64          # Doppler bins
    B_HZ: float = 50e6   # sweep bandwidth -> range resolution
    T0: float = 1e-3     # chirp repetition interval -> Doppler resolution
    FC: float = 9.39e9   # carrier
    C_LIGHT: float = 3e8
    CNR_DB: float = 15.0

    @property
    def dr(self) -> float:
        """Metres per range bin."""
        return self.C_LIGHT / (2 * self.B_HZ)

    @property
    def dv(self) -> float:
        """Metres per second per Doppler bin."""
        return self.C_LIGHT / (2 * self.FC * self.K * self.T0)

    @property
    def r_max(self) -> float:
        return (self.N - 1) * self.dr


# Measured from the dataset (docs/notes/2026-09-21-radial-sim2real-design.md):
# 512 range bins at 0.2 m, 256 Doppler bins. T0 is set so the Doppler
# resolution is the published ~0.1 m/s; FC is the automotive 77 GHz band.
RADIAL_GEOMETRY = RadarGeometry(
    N=512, K=256, B_HZ=750e6, T0=7.61e-5, FC=77e9, CNR_DB=15.0)


def _geometry_key(g: "RadarGeometry") -> tuple:
    return (g.N, g.K, g.B_HZ, g.T0, g.FC, g.C_LIGHT)


def _get_rd_matrices(geometry: "RadarGeometry | None" = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Range and Doppler steering matrices, cached per geometry.

    Keyed by geometry: an unkeyed cache would hand a 64x64 simulator the
    512x256 matrices built for a RADIal-grid one earlier in the process.
    """
    g = geometry or RadarGeometry()
    key = _geometry_key(g)
    if key not in _STEER_CACHE:
        _STEER_CACHE[key] = (
            generate_range_steering_matrix(g.N, g.N, g.B_HZ, g.C_LIGHT),
            generate_doppler_steering_matrix(g.K, g.K, g.FC, g.T0, g.C_LIGHT),
        )
    return _STEER_CACHE[key]


def _get_pq_diff(N: int, K: int) -> torch.Tensor:
    if (N, K) not in _PQ_CACHE:
        p, q = torch.meshgrid(
            torch.arange(N, dtype=torch.float),
            torch.arange(K, dtype=torch.float),
            indexing="ij",
        )
        _PQ_CACHE[(N, K)] = p - q
    return _PQ_CACHE[(N, K)]


def _get_clutter_range_steer(N: int, R: torch.Tensor, B: float,
                             c: float) -> torch.Tensor:
    key = (N, len(R), B, c)
    if key not in _CLUTTER_CACHE:
        _CLUTTER_CACHE[key] = torch.exp(
            -1j * 2 * math.pi
            * torch.outer(torch.arange(N, dtype=torch.float), R)
            * (2 * B) / (c * N)
        )
    return _CLUTTER_CACHE[key]


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


def create_rd_map(IQ_map, geometry: "RadarGeometry | None" = None):
    if not torch.is_tensor(IQ_map):
        IQ_map = torch.from_numpy(IQ_map)
    if not torch.is_complex(IQ_map):
        IQ_map = IQ_map.to(torch.complex64)
    device = IQ_map.device
    R, V = _get_rd_matrices(geometry)
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
    `clutter` and `noise` switch the AR(1) clutter and the receiver noise off,
    which gives the easy regimes (e.g. one target on an empty map).
    """

    def __init__(self, seq_len=16, frame_interval=0.5, max_targets=5,
                 rho_clutter=None, scnr=None, nu=None, clutter=True, noise=True,
                 force_class=None, geometry=None, a_max=None,
                 sigma_f=0.05, n_looks=1, range_gain_db=None):
        self.geometry = geometry or RadarGeometry()
        g = self.geometry
        self.N, self.K = g.N, g.K
        self.B_HZ, self.T0, self.FC = g.B_HZ, g.T0, g.FC
        self.C_LIGHT, self.CNR_DB = g.C_LIGHT, g.CNR_DB
        self.L = seq_len
        self.Tf = frame_interval
        self.max_targets = max_targets
        self.rho_clutter = rho_clutter
        self.scnr = scnr
        self.nu = nu
        self.clutter = clutter
        self.noise = noise
        self.force_class = force_class
        self.sigma_f = sigma_f
        # Non-coherent integration. RADIal's maps sum power over receive
        # channels, which suppresses speckle; a single look cannot reproduce
        # that dB spread however the clutter is tuned.
        self.n_looks = int(n_looks)
        # Receiver range response (blind zone, peak, far roll-off), fitted from
        # real recordings rather than assumed. dB, one value per range bin.
        self.range_gain_db = (None if range_gain_db is None
                              else torch.as_tensor(range_gain_db, dtype=torch.float))
        if self.range_gain_db is not None and len(self.range_gain_db) != self.N:
            raise ValueError(f"range_gain_db must have {self.N} entries")

        self.r_min, self.r_max, self.dr = 0.0, g.r_max, g.dr
        # Doppler grid MUST match generate_doppler_steering_matrix exactly:
        # vel_res = c/(2*fc*K*T0), bins arange(-K/2, K/2)*vel_res. Using the
        # rounded (-7.8, 0.249) grid biases traj labels by up to ~1 bin.
        self.dv = g.dv
        self.v_min = -(self.K // 2) * self.dv
        self.v_max = (self.K // 2 - 1) * self.dv
        self.R = torch.arange(self.N, dtype=torch.float) * self.dr + self.r_min
        self.V = torch.arange(-(self.K // 2), self.K // 2).float() * self.dv
        self.dR, self.dV = len(self.R), len(self.V)
        # <=1 Doppler bin of velocity change per frame, i.e. dv / Tf. The
        # shipped 64x64 grid used the rounded 0.5 and the entire cached
        # training set was drawn from that random stream, so the literal is
        # kept for the default geometry; others derive it from their own dv.
        self.a_max = a_max if a_max is not None else (
            0.5 if geometry is None else self.dv / self.Tf)

        self.sigma2 = self.N / (2 * 10 ** (self.CNR_DB / 10))
        self.cn_norm = torch.sqrt(torch.tensor(
            self.N * self.K * (self.N // 2 + self.sigma2), dtype=torch.float))

    # ---------------- target kinematics ----------------
    def _sample_kinematics(self, n, r0=None, v0=None, a=None):
        """Rejection-sample (r0, v0, a) per target s.t. all frames in-grid.

        Any of r0/v0/a passed as an (n,) tensor is held fixed and only the
        rest are drawn. With none fixed the draws are exactly the original
        ones, so default generation still reproduces the locked cache."""
        ell = torch.arange(self.L, dtype=torch.float) * self.Tf
        for _ in range(500):
            r0_ = (r0 if r0 is not None
                   else torch.empty(n).uniform_(self.r_min + 10, self.r_max - 10))
            v0_ = (v0 if v0 is not None
                   else torch.empty(n).uniform_(self.v_min + 0.5, self.v_max - 0.5))
            a_ = (a if a is not None
                  else torch.empty(n).uniform_(-self.a_max, self.a_max))
            r = r0_[:, None] + v0_[:, None] * ell + 0.5 * a_[:, None] * ell ** 2
            v = v0_[:, None] + a_[:, None] * ell
            ok = ((r >= self.r_min) & (r <= self.r_max)
                  & (v >= self.v_min) & (v <= self.v_max)).all()
            if ok:
                return r0_, v0_, a_, r, v  # r, v: (n, L)
        if any(t is not None for t in (r0, v0, a)):
            raise ValueError("no in-grid trajectory exists for the fixed kinematics")
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
        # (K, K): M is the covariance across the K slow-time samples, so both
        # axes are Doppler. Passing (N, K) happened to be square only because
        # the shipped grid had N == K == 64; on any other geometry eigh fails.
        pq = _get_pq_diff(self.K, self.K)
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

    def _explicit_kinematics(self, r0, v0, a):
        """Evaluate the constant-acceleration trajectory for caller-supplied
        (r0, v0, a), for exact placement in tests/diagnostics. Raises if any
        frame would fall outside the RD grid (mirrors the rejection-sampling
        bound used for random kinematics)."""
        ell = torch.arange(self.L, dtype=torch.float) * self.Tf
        r = r0[:, None] + v0[:, None] * ell + 0.5 * a[:, None] * ell ** 2
        v = v0[:, None] + a[:, None] * ell
        ok = ((r >= self.r_min) & (r <= self.r_max)
              & (v >= self.v_min) & (v <= self.v_max)).all()
        if not ok:
            raise ValueError("explicit kinematics leave the RD grid")
        return r0, v0, a, r, v

    # ---------------- sequence assembly ----------------
    def gen_sequence(self, r0=None, v0=None, a=None, n_targets=None, cls=None,
                     gain_db=None):
        """One labelled L-frame sequence.

        Every argument is optional; anything left as None is drawn at random
        exactly as before. Per-target arguments take one value per target and
        must agree in length (they also fix the number of targets):
          r0, v0, a : initial range (m), radial velocity (m/s), acceleration (m/s^2)
          cls       : class id per target (0 steady, 1 Swerling-1, 2 extended)
          gain_db   : per-target SCNR gain in dB
        Fully specified kinematics that leave the grid raise ValueError;
        partially specified ones are completed by rejection sampling.
        """
        def as_vec(value, dtype):
            return (None if value is None
                    else torch.as_tensor(value, dtype=dtype).reshape(-1))

        r0, v0, a = (as_vec(t, torch.float) for t in (r0, v0, a))
        cls, gain_db = as_vec(cls, torch.long), as_vec(gain_db, torch.float)
        lengths = {len(t) for t in (r0, v0, a, cls, gain_db) if t is not None}
        if len(lengths) > 1:
            raise ValueError("per-target arguments must have one value per target")
        if n_targets is not None:
            n = int(n_targets)
            if lengths and lengths != {n}:
                raise ValueError(f"n_targets={n} disagrees with per-target "
                                 f"arguments of length {next(iter(lengths))}")
        elif lengths:
            n = next(iter(lengths))
        else:
            n = int(torch.randint(1, self.max_targets + 1, (1,)).item())
        if n < 1:
            raise ValueError("a sequence needs at least one target")
        if cls is not None and ((cls < 0) | (cls > 2)).any():
            raise ValueError("class ids must be 0 (steady), 1 (Swerling-1) or 2 (extended)")

        if r0 is not None and v0 is not None and a is not None:
            r0, v0, a, r, v = self._explicit_kinematics(r0, v0, a)
        else:
            r0, v0, a, r, v = self._sample_kinematics(n, r0=r0, v0=v0, a=a)
        traj = self._to_bins(r, v)
        if cls is None:
            cls = (torch.randint(0, 3, (n,)) if self.force_class is None
                   else torch.full((n,), int(self.force_class), dtype=torch.long))
        if gain_db is None:
            gain_db = (torch.empty(n).uniform_(-5, 10) if self.scnr is None
                       else torch.full((n,), float(self.scnr)))
        rho = (float(torch.rand(1).item()) if self.rho_clutter is None
               else float(self.rho_clutter))
        nu = (float(torch.empty(1).uniform_(0.1, 1.5).item()) if self.nu is None
              else float(self.nu))

        frames, s_energy, cn_energy = [], 0.0, 0.0
        # The target signal is common to every look (the array sees one scene);
        # clutter and noise are redrawn per look and the powers are averaged.
        looks = [(self._clutter_frames(rho, nu, self.sigma_f) if self.clutter
                  else torch.zeros(self.L, self.N, self.K, dtype=torch.cfloat))
                 for _ in range(self.n_looks)]
        for l in range(self.L):
            S = self._frame_targets(r[:, l], v[:, l], gain_db, cls)
            rds = []
            for C in looks:
                W = (torch.randn(self.N, self.K, dtype=torch.cfloat)
                     / torch.sqrt(torch.tensor(2.0 * self.sigma2)))
                if not self.noise:
                    W = torch.zeros_like(W)  # still drawn: keeps the stream aligned
                X = S + C[l] + W
                s_energy += S.abs().pow(2).sum().item()
                cn_energy += (C[l] + W).abs().pow(2).sum().item()
                rds.append(create_rd_map(X, self.geometry))
            if self.n_looks == 1 and self.range_gain_db is None:
                # Historical single-look path, kept as the literal expression
                # it shipped with: the cached training set and every trained
                # checkpoint depend on it, and 10*log10(|rd|^2 + 1e-12) is not
                # bit-identical to it.
                frames.append(20 * torch.log10(rds[0].abs() + 1e-6))
            else:
                power = sum(r.abs().pow(2) for r in rds) / self.n_looks
                if self.range_gain_db is not None:
                    power = power * (10 ** (self.range_gain_db / 10)).unsqueeze(1)
                frames.append(10 * torch.log10(power + 1e-12))
        scnr_dB = 10 * torch.log10(torch.tensor(s_energy / (cn_energy + 1e-12)))
        return {
            "x": torch.stack(frames).float(),
            "traj": traj.float(),
            "v0": v0.float(), "acc": a.float(),
            "cls": cls.long(),
            "env": torch.tensor([self.CNR_DB, scnr_dB.item(), rho]).float(),
            "n_targets": n,
        }


# =====================================================================
#                    One-call batch generation
# =====================================================================
CLASS_NAMES = {"steady": 0, "swerling1": 1, "extended": 2}


def generate_sequences(n=1, seq_len=16, frame_interval=0.5, n_targets=None,
                       target_class=None, snr_db=None, clutter=True, noise=True,
                       rho=None, nu=None, r0=None, v0=None, a=None, seed=None):
    """Generate a batch of labelled RD sequences in one call.

    Anything left as None is random, drawn per sequence exactly as the
    training data was. Per-target settings (target_class, snr_db, r0, v0, a)
    take either one value, applied to every target, or a list with one value
    per target; a list also fixes the number of targets.

      n             number of sequences
      seq_len       frames per sequence, frame_interval seconds apart
      n_targets     exact targets per sequence (1-5); None = random 1-5
      target_class  "steady" | "swerling1" | "extended" (or 0 / 1 / 2)
      snr_db        target gain in dB; None = U(-5, 10) per target
      clutter       AR(1) clutter on/off
      noise         receiver noise on/off
      rho, nu       clutter frame-to-frame correlation (0-1) and texture shape
      r0, v0, a     initial range (m), radial velocity (m/s), acceleration (m/s^2)
      seed          reseed torch first, for reproducible batches

    Returns a dict of stacked tensors, target fields zero-padded to 5:
      x (n, L, 64, 64) log-magnitude dB, traj (n, 5, L, 2) true
      (range, Doppler) bins, v0 / acc / cls (n, 5), env (n, 3) as
      [CNR dB, SCNR dB, rho], n_targets (n,).

    Example, one static target on an empty map:
      generate_sequences(n_targets=1, target_class="steady", snr_db=20,
                         clutter=False, noise=False, r0=90, v0=0, a=0)
    """
    from src.dataset import MAX_TARGETS, _pad   # deferred: dataset imports this module

    def class_id(value):
        if isinstance(value, str):
            if value not in CLASS_NAMES:
                raise ValueError(f"unknown target_class {value!r}; "
                                 f"use one of {sorted(CLASS_NAMES)}")
            return CLASS_NAMES[value]
        value = int(value)
        if value not in CLASS_NAMES.values():
            raise ValueError("class ids must be 0, 1 or 2")
        return value

    def normalise(value, cast):
        if value is None:
            return None
        if isinstance(value, (list, tuple)) or (torch.is_tensor(value) and value.dim() > 0):
            return [cast(v) for v in value]
        return cast(value)

    if n < 1:
        raise ValueError("n must be at least 1")
    if rho is not None and not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be between 0 and 1")
    if nu is not None and nu <= 0:
        raise ValueError("nu must be positive")
    per_target = {"cls": normalise(target_class, class_id),
                  "gain_db": normalise(snr_db, float),
                  "r0": normalise(r0, float), "v0": normalise(v0, float),
                  "a": normalise(a, float)}
    lengths = {len(v) for v in per_target.values() if isinstance(v, list)}
    if len(lengths) > 1:
        raise ValueError("per-target lists must all have one value per target")
    if n_targets is None and lengths:
        n_targets = next(iter(lengths))
    elif n_targets is not None and lengths and lengths != {n_targets}:
        raise ValueError(f"n_targets={n_targets} disagrees with per-target lists "
                         f"of length {next(iter(lengths))}")
    if n_targets is not None and not 1 <= n_targets <= MAX_TARGETS:
        raise ValueError(f"n_targets must be between 1 and {MAX_TARGETS}")

    if seed is not None:
        torch.manual_seed(seed)
    sim = TemporalRadarSimulator(seq_len=seq_len, frame_interval=frame_interval,
                                 max_targets=MAX_TARGETS, rho_clutter=rho, nu=nu,
                                 clutter=clutter, noise=noise)
    items = []
    for _ in range(n):
        m = (n_targets if n_targets is not None
             else int(torch.randint(1, MAX_TARGETS + 1, (1,)).item()))
        args = {k: (None if v is None else (v if isinstance(v, list) else [v] * m))
                for k, v in per_target.items()}
        items.append(_pad(sim.gen_sequence(n_targets=m, **args)))
    return {k: torch.stack([item[k] for item in items]) for k in items[0]}
