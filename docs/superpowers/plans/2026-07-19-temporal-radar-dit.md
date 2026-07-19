# Temporal Radar DiT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diffusion transformer that generates temporally coherent 16-frame sequences of 64×64 Range-Doppler maps with physics losses and motion/env/class conditioning, per the approved spec.

**Architecture:** Synthetic temporal radar simulator (kinematic targets + AR(1) clutter, adapted from RDDiffusion) feeds a temporal-only-attention DiT (overlapping patches p=8/s=4, adaLN-Zero) trained with DDPM ε-prediction plus smoothness/trajectory/Doppler losses; evaluation tracks extracted peaks for physics metrics.

**Tech Stack:** PyTorch (existing env), numpy, scipy (peak detection), matplotlib + imageio (viz), pytest, wandb (optional logging), PyYAML.

**Spec:** `docs/superpowers/specs/2026-07-18-temporal-radar-dit-design.md` — read it before starting any task.

## Global Constraints

- Project root: `/truenas/home/arigra/permuter/ariGranevich/radSeq/` (git repo already initialized). All paths below are relative to it.
- All code under `src/`, tests under `tests/`. Run pytest from project root: `python -m pytest tests/ -v`.
- Radar grid is fixed: N=K=64, dR=dV=64, B=50e6 Hz, T0=1e-3 s, fc=9.39e9 Hz, c=3e8 m/s, range grid 0..189 step 3 m, velocity grid `arange(-32,32)·dv` with `dv = c/(2·fc·K·T0) ≈ 0.2496 m/s` (must match `generate_doppler_steering_matrix` exactly — do NOT use the rounded −7.8/0.249 grid, it biases labels by ~1 bin).
- Sequence spec: L=16 frames, inter-frame interval T_f=0.5 s, accelerations bounded so targets move ~1–2 bins/frame.
- Model spec: p=8, s=4 → 15×15=225 patches/frame, d=256, B=8 blocks, H=8 heads, adaLN-Zero.
- Diffusion: DDPM T=1000, cosine ᾱ schedule, ε-prediction.
- Default loss weights: λ_smooth=0.1, λ_traj=0.01, λ_Doppler=0.01.
- Reference code to copy from (read-only, never modify): `/truenas/home/arigra/permuter/ariGranevich/RDDiffusion/radar_dataset.py`.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Seed all tests with `torch.manual_seed(0)` at the top of each test function that samples randomness.

---

### Task 1: Package scaffold + overlapping patchify/unpatchify

**Files:**
- Create: `src/__init__.py` (empty), `src/eval/__init__.py` (empty), `tests/__init__.py` (empty)
- Create: `configs/base.yaml`
- Create: `src/patching.py`
- Test: `tests/test_patching.py`

**Interfaces:**
- Produces: `patchify(x, p=8, s=4) -> Tensor` mapping `(B, L, 64, 64) -> (B, L, 225, 64)`, and `unpatchify(tokens, N=64, K=64, p=8, s=4) -> Tensor` mapping `(B, L, 225, 64) -> (B, L, 64, 64)`; `num_patches(N, K, p, s) -> (int, int)` per-axis patch counts.

- [ ] **Step 1: Create package skeleton and config**

Create empty `src/__init__.py`, `src/eval/__init__.py`, `tests/__init__.py`.

`configs/base.yaml`:

```yaml
data:
  n_train: 20000
  n_val: 2000
  seq_len: 16
  frame_interval: 0.5        # seconds between frames
  cache_dir: data/cache
  shard_size: 1000
  seed: 1234
model:
  patch: 8
  stride: 4
  dim: 256
  depth: 8
  heads: 8
diffusion:
  timesteps: 1000
train:
  batch_size: 16
  lr: 1.0e-4
  weight_decay: 0.0
  epochs: 100
  lambda_smooth: 0.1
  lambda_traj: 0.01
  lambda_doppler: 0.01
  phase: 1                   # 1: L_DiT+L_smooth, 2: +traj/doppler, 3: +conditioning
  ckpt_dir: checkpoints
  wandb: false
sample:
  ddim_steps: 50
```

- [ ] **Step 2: Write the failing test**

`tests/test_patching.py`:

```python
import torch
from src.patching import patchify, unpatchify, num_patches


def test_num_patches():
    assert num_patches(64, 64, 8, 4) == (15, 15)


def test_patchify_shape():
    x = torch.randn(2, 16, 64, 64)
    t = patchify(x, p=8, s=4)
    assert t.shape == (2, 16, 225, 64)


def test_roundtrip_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64, 64)
    t = patchify(x, p=8, s=4)
    y = unpatchify(t, N=64, K=64, p=8, s=4)
    assert torch.allclose(x, y, atol=1e-5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_patching.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'src.patching'`

- [ ] **Step 4: Write implementation**

`src/patching.py`:

```python
"""Overlapping patch extraction and reassembly for RD maps.

Patches of size p with stride s < p overlap; unpatchify averages
contributions in overlapped regions (F.fold divided by hit counts),
so patchify -> unpatchify is the identity on raw maps.
"""
import torch
import torch.nn.functional as F


def num_patches(N: int, K: int, p: int, s: int) -> tuple[int, int]:
    return ((N - p) // s + 1, (K - p) // s + 1)


def patchify(x: torch.Tensor, p: int = 8, s: int = 4) -> torch.Tensor:
    """(B, L, N, K) -> (B, L, P, p*p)"""
    B, L, N, K = x.shape
    u = F.unfold(x.reshape(B * L, 1, N, K), kernel_size=p, stride=s)  # (B*L, p*p, P)
    P = u.shape[-1]
    return u.transpose(1, 2).reshape(B, L, P, p * p)


def unpatchify(tokens: torch.Tensor, N: int = 64, K: int = 64,
               p: int = 8, s: int = 4) -> torch.Tensor:
    """(B, L, P, p*p) -> (B, L, N, K), averaging overlaps."""
    B, L, P, d = tokens.shape
    u = tokens.reshape(B * L, P, d).transpose(1, 2)          # (B*L, p*p, P)
    out = F.fold(u, (N, K), kernel_size=p, stride=s)
    cnt = F.fold(torch.ones_like(u), (N, K), kernel_size=p, stride=s)
    return (out / cnt).reshape(B, L, N, K)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_patching.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/ tests/ configs/
git commit -m "feat: package scaffold + overlapping patchify/unpatchify

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Single-frame radar core (copied from RDDiffusion) + kinematic temporal targets

**Files:**
- Create: `src/simulator.py`
- Test: `tests/test_simulator.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TemporalRadarSimulator` class with:
  - `__init__(self, seq_len=16, frame_interval=0.5, max_targets=5, rho_clutter=None, scnr=None, nu=None, clutter_mode="strip", force_class=None)` — `None` means "sample per sequence"; `force_class` pins every target's class id (used by tests/eval).
  - `gen_sequence(self) -> dict` with keys:
    - `"x"`: float32 `(L, 64, 64)` log-magnitude RD sequence, `20*log10(|RD|+1e-6)`
    - `"traj"`: float32 `(M, L, 2)` continuous (range_bin, doppler_bin) per target per frame
    - `"v0"`: float32 `(M,)` initial radial velocity (m/s); `"acc"`: float32 `(M,)` acceleration (m/s²)
    - `"cls"`: int64 `(M,)` class id (0 steady, 1 swerling1, 2 extended)
    - `"env"`: float32 `(3,)` = `[cnr_dB, scnr_dB, rho]`
    - `"n_targets"`: int
  - Module-level `create_rd_map(iq: Tensor) -> Tensor` (64,64 complex).
- Class semantics: 0 = steady point (fixed gain), 1 = Swerling-1 (per-frame power ~ Exponential), 2 = range-extended (3 scatterers at r−3, r, r+3 m with gains −3 dB on the flanks).

- [ ] **Step 1: Copy the single-frame radar core**

Create `src/simulator.py`. Copy **verbatim** from `/truenas/home/arigra/permuter/ariGranevich/RDDiffusion/radar_dataset.py` these pieces (they are pure functions of the fixed radar constants):

- module-level caches and getters: `_RD_R`, `_RD_V`, `_PQ_DIFF`, `_CLUTTER_R_STEER`, `_get_rd_matrices`, `_get_pq_diff`, `_get_clutter_range_steer` (lines ~60–115)
- `generate_range_steering_matrix`, `generate_doppler_steering_matrix` (lines ~509–525)
- `create_rd_map_differentiable` (lines ~528–536) — rename to `create_rd_map`

Then add the simulator class below them (Step 4).

- [ ] **Step 2: Write the failing tests**

`tests/test_simulator.py`:

```python
import torch
from src.simulator import TemporalRadarSimulator


def test_output_shapes_and_keys():
    torch.manual_seed(0)
    sim = TemporalRadarSimulator(seq_len=16)
    out = sim.gen_sequence()
    M = out["n_targets"]
    assert out["x"].shape == (16, 64, 64) and out["x"].dtype == torch.float32
    assert out["traj"].shape == (M, 16, 2)
    assert out["v0"].shape == (M,) and out["acc"].shape == (M,)
    assert out["cls"].shape == (M,) and out["env"].shape == (3,)
    assert 1 <= M <= 5


def test_trajectory_within_grid():
    torch.manual_seed(1)
    sim = TemporalRadarSimulator(seq_len=16)
    for _ in range(10):
        out = sim.gen_sequence()
        assert (out["traj"][..., 0] >= 0).all() and (out["traj"][..., 0] <= 63).all()
        assert (out["traj"][..., 1] >= 0).all() and (out["traj"][..., 1] <= 63).all()


def test_peak_follows_trajectory():
    """Single steady point target (class forced to 0), no clutter/noise floor
    dominance: the RD peak must land within 1.5 bins of the continuous
    analytic trajectory at every frame (integer peak vs off-grid ground
    truth can differ by up to ~1.5 bins at bin boundaries)."""
    torch.manual_seed(2)
    sim = TemporalRadarSimulator(seq_len=16, max_targets=1, scnr=20.0,
                                 force_class=0)
    out = sim.gen_sequence()
    assert out["n_targets"] == 1
    for l in range(16):
        frame = out["x"][l]
        idx = frame.flatten().argmax()
        r_pk, v_pk = (idx // 64).item(), (idx % 64).item()
        r_gt, v_gt = out["traj"][0, l]
        assert abs(r_pk - r_gt) <= 1.5, f"frame {l}: range {r_pk} vs {r_gt}"
        assert abs(v_pk - v_gt) <= 1.5, f"frame {l}: doppler {v_pk} vs {v_gt}"


def test_motion_magnitude():
    """Targets move, but no more than ~2 bins/frame in range."""
    torch.manual_seed(3)
    sim = TemporalRadarSimulator(seq_len=16)
    out = sim.gen_sequence()
    dr = (out["traj"][:, 1:, 0] - out["traj"][:, :-1, 0]).abs()
    assert dr.max() <= 2.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: FAIL with `ImportError` (no `TemporalRadarSimulator`)

- [ ] **Step 4: Implement TemporalRadarSimulator (targets + white noise; clutter comes in Task 3)**

Append to `src/simulator.py`:

```python
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

    # ---------------- clutter (replaced in Task 3) ----------------
    def _clutter_frames(self, rho, nu):
        return torch.zeros(self.L, self.N, self.K, dtype=torch.cfloat)

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: 4 passed. If `test_peak_follows_trajectory` fails at a frame boundary, check the bin convention: `traj` uses `(r - r_min)/dr`, the RD map's range axis is produced by `create_rd_map` on the same grid — axis 0 is range, axis 1 is Doppler.

- [ ] **Step 6: Commit**

```bash
git add src/simulator.py tests/test_simulator.py
git commit -m "feat: temporal radar simulator with kinematic targets

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: AR(1) correlated clutter

**Files:**
- Modify: `src/simulator.py` (replace `_clutter_frames` stub; add texture sampler)
- Test: `tests/test_clutter.py`

**Interfaces:**
- Consumes: `TemporalRadarSimulator` from Task 2, copied `_get_pq_diff` / `_get_clutter_range_steer` helpers.
- Produces: `_clutter_frames(rho, nu) -> Tensor (L, 64, 64) cfloat` — K-distributed strip clutter whose speckle innovations follow AR(1) with coefficient rho; texture fixed per sequence. Also `_sample_texture(nu) -> Tensor (dR,)`.

- [ ] **Step 1: Write the failing test**

`tests/test_clutter.py`:

```python
import torch
from src.simulator import TemporalRadarSimulator


def _clutter_corr(sim, rho, n_seq=8):
    """Mean lag-1 correlation of clutter IQ across frames."""
    torch.manual_seed(0)
    cors = []
    for _ in range(n_seq):
        C = sim._clutter_frames(rho=rho, nu=0.5)          # (L, 64, 64)
        a = C[:-1].flatten(1)
        b = C[1:].flatten(1)
        num = (a.conj() * b).sum(dim=1).real
        den = (a.abs().pow(2).sum(dim=1).sqrt()
               * b.abs().pow(2).sum(dim=1).sqrt())
        cors.append((num / den).mean())
    return torch.stack(cors).mean().item()


def test_clutter_shape_and_nonzero():
    torch.manual_seed(0)
    sim = TemporalRadarSimulator(seq_len=16)
    C = sim._clutter_frames(rho=0.5, nu=0.5)
    assert C.shape == (16, 64, 64) and C.dtype == torch.cfloat
    assert C.abs().sum() > 0


def test_ar1_correlation_tracks_rho():
    sim = TemporalRadarSimulator(seq_len=16)
    c_lo = _clutter_corr(sim, rho=0.1)
    c_hi = _clutter_corr(sim, rho=0.9)
    assert c_hi > c_lo + 0.3
    assert abs(c_hi - 0.9) < 0.15


def test_rho_zero_uncorrelated():
    sim = TemporalRadarSimulator(seq_len=16)
    assert abs(_clutter_corr(sim, rho=0.0)) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clutter.py -v`
Expected: `test_clutter_shape_and_nonzero` FAILS (stub returns zeros → `C.abs().sum() > 0` assertion fails).

- [ ] **Step 3: Implement AR(1) clutter**

In `src/simulator.py`, replace the `_clutter_frames` stub inside `TemporalRadarSimulator` with:

```python
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
```

- [ ] **Step 4: Run all simulator tests**

Run: `python -m pytest tests/test_clutter.py tests/test_simulator.py -v`
Expected: all pass. `test_peak_follows_trajectory` must still pass — it uses `scnr=20.0` so targets dominate clutter.

- [ ] **Step 5: Commit**

```bash
git add src/simulator.py tests/test_clutter.py
git commit -m "feat: AR(1) correlated clutter in temporal simulator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Dataset generation, caching, normalization

**Files:**
- Create: `src/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `TemporalRadarSimulator.gen_sequence()` (Task 2/3 dict schema).
- Produces:
  - `generate_cache(cache_dir, n_train, n_val, seq_len=16, seed=1234, shard_size=1000)` — writes `train_XXXX.pt` / `val_XXXX.pt` shards + `stats.pt` (`{"mean": float, "std": float}` over train `x`) + `manifest.yaml` (all params).
  - `RadarSequenceDataset(cache_dir, split)` — torch `Dataset`; `__getitem__` returns the simulator dict but with `"x"` normalized to `(x - mean)/std` and padded target fields: `"traj"` `(5, L, 2)`, `"v0"`/`"acc"` `(5,)`, `"cls"` `(5,)` (pad value 0, real count in `"n_targets"`), so default collate works.
  - `denormalize(x, stats) -> Tensor`.
- Shard format: each shard is a `torch.save`d list of per-sequence dicts.

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:

```python
import torch
from src.dataset import generate_cache, RadarSequenceDataset, denormalize


def test_cache_and_load(tmp_path):
    generate_cache(str(tmp_path), n_train=6, n_val=3, seq_len=16,
                   seed=7, shard_size=4)
    ds = RadarSequenceDataset(str(tmp_path), "train")
    assert len(ds) == 6
    item = ds[0]
    assert item["x"].shape == (16, 64, 64)
    assert item["traj"].shape == (5, 16, 2)
    assert item["v0"].shape == (5,) and item["cls"].shape == (5,)
    # normalized: roughly zero-mean unit-std over the split
    xs = torch.stack([ds[i]["x"] for i in range(len(ds))])
    assert xs.mean().abs() < 0.3 and (xs.std() - 1).abs() < 0.3


def test_val_uses_train_stats(tmp_path):
    generate_cache(str(tmp_path), n_train=6, n_val=3, seq_len=16,
                   seed=7, shard_size=4)
    tr = RadarSequenceDataset(str(tmp_path), "train")
    va = RadarSequenceDataset(str(tmp_path), "val")
    assert tr.stats == va.stats


def test_denormalize_roundtrip(tmp_path):
    generate_cache(str(tmp_path), n_train=2, n_val=1, seq_len=16,
                   seed=7, shard_size=4)
    ds = RadarSequenceDataset(str(tmp_path), "train")
    item = ds[0]
    raw = denormalize(item["x"], ds.stats)
    renorm = (raw - ds.stats["mean"]) / ds.stats["std"]
    assert torch.allclose(renorm, item["x"], atol=1e-5)


def test_collate_batches(tmp_path):
    generate_cache(str(tmp_path), n_train=4, n_val=1, seq_len=16,
                   seed=7, shard_size=4)
    ds = RadarSequenceDataset(str(tmp_path), "train")
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch["x"].shape == (2, 16, 64, 64)
    assert batch["traj"].shape == (2, 5, 16, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dataset'`

- [ ] **Step 3: Implement**

`src/dataset.py`:

```python
"""Pre-generate, cache, and load temporal radar sequences.

Shards: <split>_<idx:04d>.pt, each a list of per-sequence dicts from
TemporalRadarSimulator.gen_sequence() with target fields padded to
MAX_TARGETS. stats.pt holds train-split mean/std of the log-magnitude
maps; both splits are normalized with it.
"""
import math
from pathlib import Path

import torch
import yaml

from src.simulator import TemporalRadarSimulator

MAX_TARGETS = 5


def _pad(item):
    m = item["n_targets"]
    out = dict(item)
    out["traj"] = torch.zeros(MAX_TARGETS, item["traj"].shape[1], 2)
    out["traj"][:m] = item["traj"]
    for k in ("v0", "acc"):
        out[k] = torch.zeros(MAX_TARGETS)
        out[k][:m] = item[k]
    out["cls"] = torch.zeros(MAX_TARGETS, dtype=torch.long)
    out["cls"][:m] = item["cls"]
    out["n_targets"] = torch.tensor(m)
    return out


def generate_cache(cache_dir, n_train, n_val, seq_len=16, seed=1234,
                   shard_size=1000, frame_interval=0.5):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    sim = TemporalRadarSimulator(seq_len=seq_len, frame_interval=frame_interval)

    def write_split(split, n):
        n_shards = math.ceil(n / shard_size)
        for si in range(n_shards):
            count = min(shard_size, n - si * shard_size)
            shard = [_pad(sim.gen_sequence()) for _ in range(count)]
            torch.save(shard, cache / f"{split}_{si:04d}.pt")

    write_split("train", n_train)
    # stats from the train shards only
    total, total_sq, count = 0.0, 0.0, 0
    for f in sorted(cache.glob("train_*.pt")):
        for item in torch.load(f):
            x = item["x"]
            total += x.sum().item()
            total_sq += (x ** 2).sum().item()
            count += x.numel()
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean ** 2, 1e-12))
    torch.save({"mean": mean, "std": std}, cache / "stats.pt")

    write_split("val", n_val)
    with open(cache / "manifest.yaml", "w") as fh:
        yaml.safe_dump({"n_train": n_train, "n_val": n_val, "seq_len": seq_len,
                        "seed": seed, "shard_size": shard_size,
                        "frame_interval": frame_interval,
                        "mean": mean, "std": std}, fh)


def denormalize(x, stats):
    return x * stats["std"] + stats["mean"]


class RadarSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, split):
        cache = Path(cache_dir)
        self.stats = torch.load(cache / "stats.pt")
        self.items = []
        for f in sorted(cache.glob(f"{split}_*.pt")):
            self.items.extend(torch.load(f))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        item["x"] = (item["x"] - self.stats["mean"]) / self.stats["std"]
        return item
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: sequence dataset generation, caching, normalization

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: DDPM/DDIM diffusion machinery

**Files:**
- Create: `src/diffusion.py`
- Test: `tests/test_diffusion.py`

**Interfaces:**
- Consumes: nothing project-specific (model passed in as a callable).
- Produces: `GaussianDiffusion(timesteps=1000)` with:
  - `.alphas_bar`: `(T,)` tensor
  - `.q_sample(x0, t, eps) -> xt` (t: `(B,)` long)
  - `.pred_x0(xt, t, eps_hat) -> x0_hat`
  - `.loss_weight(t) -> (B,)` returning `1 - alphas_bar[t]` (the ω_t for L_smooth)
  - `.ddim_sample(model, shape, device, steps=50, cond=None) -> x0` where `model(xt, t, cond)` returns ε̂; `shape=(B, L, N, K)`
  - `.p_sample_loop(model, shape, device, cond=None) -> x0` (full ancestral DDPM)

- [ ] **Step 1: Write the failing test**

`tests/test_diffusion.py`:

```python
import torch
from src.diffusion import GaussianDiffusion


def test_schedule_monotone():
    d = GaussianDiffusion(timesteps=1000)
    ab = d.alphas_bar
    assert ab.shape == (1000,)
    assert (ab[1:] <= ab[:-1] + 1e-8).all()
    assert ab[0] > 0.99 and ab[-1] < 0.01


def test_q_sample_pred_x0_roundtrip():
    torch.manual_seed(0)
    d = GaussianDiffusion(timesteps=1000)
    x0 = torch.randn(2, 16, 64, 64)
    t = torch.tensor([100, 900])
    eps = torch.randn_like(x0)
    xt = d.q_sample(x0, t, eps)
    x0_hat = d.pred_x0(xt, t, eps)
    assert torch.allclose(x0, x0_hat, atol=1e-4)


def test_ddim_sample_shape_and_finite():
    torch.manual_seed(0)
    d = GaussianDiffusion(timesteps=1000)

    def dummy_model(xt, t, cond=None):
        return torch.zeros_like(xt)

    out = d.ddim_sample(dummy_model, (1, 16, 64, 64), torch.device("cpu"), steps=10)
    assert out.shape == (1, 16, 64, 64)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diffusion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/diffusion.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diffusion.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/diffusion.py tests/test_diffusion.py
git commit -m "feat: DDPM cosine schedule with DDIM sampler

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Temporal-only DiT backbone

**Files:**
- Create: `src/dit.py`
- Test: `tests/test_dit.py`

**Interfaces:**
- Consumes: `patchify` / `unpatchify` / `num_patches` from `src/patching.py`.
- Produces: `TemporalDiT(seq_len=16, N=64, K=64, patch=8, stride=4, dim=256, depth=8, heads=8)`; `forward(x, t, cond=None) -> Tensor` with `x: (B, L, N, K)`, `t: (B,) long`, `cond: (B, dim)` or `None`, returning ε̂ of shape `(B, L, N, K)`. Also module-level `timestep_embedding(t, dim) -> (B, dim)`.
- Attention is strictly temporal: tokens are reshaped to `(B·P, L, dim)` before MSA, so site i never attends to site j.

- [ ] **Step 1: Write the failing test**

`tests/test_dit.py`:

```python
import torch
from src.dit import TemporalDiT, timestep_embedding


def _small_model():
    return TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=4,
                       dim=32, depth=2, heads=4)


def test_timestep_embedding_shape():
    e = timestep_embedding(torch.tensor([0, 500]), 32)
    assert e.shape == (2, 32)


def test_forward_shape():
    torch.manual_seed(0)
    m = _small_model()
    x = torch.randn(2, 4, 16, 16)
    out = m(x, torch.tensor([10, 20]))
    assert out.shape == (2, 4, 16, 16)


def test_zero_init_output():
    """adaLN-Zero: an untrained model must output exactly zero."""
    m = _small_model()
    x = torch.randn(1, 4, 16, 16)
    out = m(x, torch.tensor([5]))
    assert out.abs().max() == 0.0


def test_spatial_independence():
    """Perturbing a distant spatial region must not change the output
    in a region whose overlapping patches don't cover it.
    With patch=8/stride=4 on 16x16, pixel (0,0) is only in patch (0,0)
    covering rows/cols 0-7; pixel (15,15) is only in patch (2,2)
    covering rows/cols 8-15. Disjoint -> output at (0,0) fixed."""
    torch.manual_seed(0)
    m = _small_model()
    for p in m.parameters():  # break zero-init so the test is non-trivial
        torch.nn.init.normal_(p, std=0.02)
    m.eval()
    x = torch.randn(1, 4, 16, 16)
    x2 = x.clone()
    x2[:, :, 12:, 12:] += 10.0
    with torch.no_grad():
        o1 = m(x, torch.tensor([100]))
        o2 = m(x2, torch.tensor([100]))
    assert torch.allclose(o1[:, :, :4, :4], o2[:, :, :4, :4], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dit.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/dit.py`:

```python
"""Temporal-only Diffusion Transformer for RD sequences.

Tokens: overlapping patches per frame. Attention runs over the L frames
of each spatial site independently ((B*P, L, d) reshape) — no spatial
mixing beyond patch overlap, per the proposal. adaLN-Zero conditioning
on diffusion timestep (+ optional condition vector added to it).
"""
import math

import torch
import torch.nn as nn

from src.patching import patchify, unpatchify, num_patches


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TemporalBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, z, c):
        # z: (B, L, P, d); c: (B, d)
        B, L, P, d = z.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c)[:, None, None].chunk(6, dim=-1)
        h = self.norm1(z) * (1 + sc1) + sh1
        h = h.permute(0, 2, 1, 3).reshape(B * P, L, d)
        a, _ = self.attn(h, h, h, need_weights=False)
        a = a.reshape(B, P, L, d).permute(0, 2, 1, 3)
        z = z + g1 * a
        h = self.norm2(z) * (1 + sc2) + sh2
        return z + g2 * self.mlp(h)


class TemporalDiT(nn.Module):
    def __init__(self, seq_len=16, N=64, K=64, patch=8, stride=4,
                 dim=256, depth=8, heads=8):
        super().__init__()
        self.N, self.K, self.p, self.s = N, K, patch, stride
        pr, pc = num_patches(N, K, patch, stride)
        P = pr * pc
        self.proj = nn.Linear(patch * patch, dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, P, dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, seq_len, 1, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                   nn.Linear(dim, dim))
        self.dim = dim
        self.blocks = nn.ModuleList(TemporalBlock(dim, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch)
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond=None):
        tokens = patchify(x, self.p, self.s)                  # (B, L, P, p*p)
        z = self.proj(tokens) + self.spatial_pos + self.temporal_pos
        c = self.t_mlp(timestep_embedding(t, self.dim))
        if cond is not None:
            c = c + cond
        for blk in self.blocks:
            z = blk(z, c)
        sh, sc = self.final_adaLN(c)[:, None, None].chunk(2, dim=-1)
        z = self.final_norm(z) * (1 + sc) + sh
        return unpatchify(self.out(z), self.N, self.K, self.p, self.s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dit.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/dit.py tests/test_dit.py
git commit -m "feat: temporal-only DiT with adaLN-Zero

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Phase-1 losses (L_DiT + L_smooth)

**Files:**
- Create: `src/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: `GaussianDiffusion.pred_x0` / `.loss_weight` (Task 5).
- Produces:
  - `diffusion_loss(eps, eps_hat) -> scalar` — mean squared error.
  - `smooth_loss(x0_hat, weight) -> scalar` — `x0_hat: (B, L, N, K)`, `weight: (B,)` = ω_t; mean over batch of `weight * mean_{l,pixels}((x0_hat[l] - x0_hat[l-1])**2)`.

- [ ] **Step 1: Write the failing test**

`tests/test_losses.py`:

```python
import torch
from src.losses import diffusion_loss, smooth_loss


def test_diffusion_loss_zero_when_equal():
    e = torch.randn(2, 16, 64, 64)
    assert diffusion_loss(e, e).item() == 0.0
    assert diffusion_loss(e, torch.zeros_like(e)).item() > 0


def test_smooth_loss_zero_for_static():
    x = torch.randn(2, 1, 64, 64).repeat(1, 16, 1, 1)   # identical frames
    w = torch.ones(2)
    assert smooth_loss(x, w).item() < 1e-10


def test_smooth_loss_scales_with_weight():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64, 64)
    l1 = smooth_loss(x, torch.ones(2))
    l2 = smooth_loss(x, 2 * torch.ones(2))
    assert torch.allclose(l2, 2 * l1)


def test_smooth_loss_grad_flows():
    x = torch.randn(1, 16, 64, 64, requires_grad=True)
    smooth_loss(x, torch.ones(1)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/losses.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_losses.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: diffusion and temporal smoothness losses

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Training loop + visualization + overfit smoke test

**Files:**
- Create: `src/train.py`, `src/viz.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `RadarSequenceDataset`, `GaussianDiffusion`, `TemporalDiT`, `diffusion_loss`, `smooth_loss`, config schema from `configs/base.yaml`.
- Produces:
  - `train.py`: `train(config: dict, device=None, max_steps=None) -> TemporalDiT` and CLI `python -m src.train --config configs/base.yaml`. Per step: sample `t ~ U{0..T-1}`, `eps`, compute `xt`, ε̂, `L = L_DiT + λ_smooth·L_smooth` (phase ≥ 2 adds traj/Doppler in Task 11). Saves `last.pt` checkpoint (`{"model": state_dict, "config": config}`) to `train.ckpt_dir` each epoch. wandb logging only when `train.wandb` is true.
  - `viz.py`: `sequence_grid(x, path)` — `(L, N, K)` tensor to a PNG grid of frames; `sequence_gif(x, path)` — animated GIF.

- [ ] **Step 1: Write the failing test**

`tests/test_train_smoke.py`:

```python
import torch
import yaml
from src.dataset import generate_cache
from src.train import train


def _tiny_config(tmp_path):
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2,
                       shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4)
    # lr=3e-4: tiny model overfitting a single batch converges too slowly at
    # the production 1e-4 within 150 steps; the config owns the lr, train()
    # has no debug-only hyperparameters
    cfg["train"].update(batch_size=2, epochs=1, lr=3.0e-4,
                        ckpt_dir=str(tmp_path / "ckpt"))
    return cfg


def test_single_batch_overfit(tmp_path):
    """Loss on a fixed batch and fixed t must drop substantially."""
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16,
                   seed=7, shard_size=4)
    losses = train(cfg, device=torch.device("cpu"), max_steps=150,
                   _record_losses=True)
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < 0.6 * early, f"no learning: {early:.4f} -> {late:.4f}"


def test_checkpoint_written(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    generate_cache(cfg["data"]["cache_dir"], 4, 2, seq_len=16,
                   seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=3)
    assert (tmp_path / "ckpt" / "last.pt").exists()


def test_viz_writes_files(tmp_path):
    from src.viz import sequence_grid, sequence_gif
    x = torch.randn(16, 64, 64)
    sequence_grid(x, str(tmp_path / "grid.png"))
    sequence_gif(x, str(tmp_path / "seq.gif"))
    assert (tmp_path / "grid.png").exists() and (tmp_path / "seq.gif").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement viz**

`src/viz.py`:

```python
"""Sequence visualization: frame grids and GIFs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sequence_grid(x, path, ncols=8):
    x = x.detach().cpu().numpy()
    L = x.shape[0]
    nrows = (L + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    vmin, vmax = x.min(), x.max()
    for i, ax in enumerate(np.atleast_1d(axes).flatten()):
        ax.axis("off")
        if i < L:
            ax.imshow(x[i], vmin=vmin, vmax=vmax, cmap="viridis",
                      origin="lower", aspect="auto")
            ax.set_title(f"t={i}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def sequence_gif(x, path, fps=4):
    import imageio.v2 as imageio
    x = x.detach().cpu().numpy()
    lo, hi = x.min(), x.max()
    frames = ((x - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    imageio.mimsave(path, list(frames), fps=fps)
```

- [ ] **Step 4: Implement train**

`src/train.py`:

```python
"""Training loop for the temporal radar DiT."""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import RadarSequenceDataset
from src.diffusion import GaussianDiffusion
from src.dit import TemporalDiT
from src.losses import diffusion_loss, smooth_loss


def build_model(cfg, device):
    m = cfg["model"]
    return TemporalDiT(seq_len=cfg["data"]["seq_len"], patch=m["patch"],
                       stride=m["stride"], dim=m["dim"], depth=m["depth"],
                       heads=m["heads"]).to(device)


def train(cfg, device=None, max_steps=None, _record_losses=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg["train"]
    ds = RadarSequenceDataset(cfg["data"]["cache_dir"], "train")
    loader = DataLoader(ds, batch_size=tr["batch_size"], shuffle=True,
                        num_workers=0, drop_last=True)
    model = build_model(cfg, device)
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    use_wandb = tr.get("wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="radSeq", config=cfg)

    ckpt_dir = Path(tr["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    losses, step = [], 0
    fixed_batch = next(iter(loader)) if _record_losses else None
    for epoch in range(tr["epochs"]):
        for batch in loader:
            if _record_losses:
                batch = fixed_batch  # overfit a single batch deterministically
            x0 = batch["x"].to(device)
            t = torch.randint(0, diff.T, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, eps)
            eps_hat = model(xt, t)
            loss = diffusion_loss(eps, eps_hat)
            # clamp matches the samplers' convention; unclamped pred_x0
            # explodes at high t (divide by sqrt(abar)->0) and destabilizes
            # the smoothness loss
            x0_hat = diff.pred_x0(xt, t, eps_hat).clamp(-4, 4)
            loss = loss + tr["lambda_smooth"] * smooth_loss(x0_hat, diff.loss_weight(t))
            if tr.get("phase", 1) >= 2:
                from src.losses import traj_loss_from_batch  # added in Task 11
                loss = loss + traj_loss_from_batch(x0_hat, batch, tr, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if use_wandb and step % 50 == 0:
                wandb.log({"loss": loss.item(), "step": step})
            step += 1
            if max_steps is not None and step >= max_steps:
                torch.save({"model": model.state_dict(), "config": cfg},
                           ckpt_dir / "last.pt")
                return losses if _record_losses else model
        torch.save({"model": model.state_dict(), "config": cfg},
                   ckpt_dir / "last.pt")
    return losses if _record_losses else model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    train(config)
```

Note: the `phase >= 2` import of `traj_loss_from_batch` is a forward reference implemented in Task 11; with the default config (`phase: 1`) it is never executed, so Phase-1 tests pass.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: 3 passed (overfit test takes ~2–4 min on CPU with the tiny model).

- [ ] **Step 6: Run full test suite and commit**

Run: `python -m pytest tests/ -v` — expected: all pass.

```bash
git add src/train.py src/viz.py tests/test_train_smoke.py
git commit -m "feat: training loop with smoothness loss + sequence viz

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Physics evaluation metrics

**Files:**
- Create: `src/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing project-specific (operates on raw `(L, 64, 64)` log-magnitude arrays).
- Produces:
  - `detect_peaks(frame, threshold_db=12.0, max_peaks=8) -> Tensor (n, 2)` — local maxima (3×3 neighborhood, scipy `maximum_filter`) at least `threshold_db` above the frame median, strongest first.
  - `link_tracks(peaks_per_frame, gate=3.0) -> list[list[tuple[int, tuple]]]` — greedy nearest-neighbor linking; each track is a list of `(frame_idx, (r, d))`; unmatched peaks start new tracks.
  - `velocity_consistency(tracks) -> float` — mean squared second difference of positions over tracks with ≥3 points (lower = more physical).
  - `doppler_drift(tracks) -> float` — mean absolute frame-to-frame change of the Doppler coordinate.
  - `persistence(tracks, seq_len) -> float` — fraction of tracks spanning ≥ 80% of frames.
  - `marginal_l1(x_gen, x_real, bins=64) -> float` — L1 distance between normalized intensity histograms.
  - `evaluate_sequences(x_gen, x_real, seq_len=16) -> dict` — runs all of the above over batches `(B, L, 64, 64)`, returns `{"velocity_consistency", "doppler_drift", "persistence", "marginal_l1", "mean_tracks_per_seq"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:

```python
import torch
from src.eval.metrics import (detect_peaks, link_tracks, velocity_consistency,
                              doppler_drift, persistence, marginal_l1,
                              evaluate_sequences)


def _synthetic_sequence(vel=(1.0, 0.5), start=(10.0, 20.0), L=16, noise=0.1):
    """One Gaussian blob moving at constant velocity on a noisy floor."""
    torch.manual_seed(0)
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for l in range(L):
        r = start[0] + vel[0] * l
        c = start[1] + vel[1] * l
        blob = 30 * torch.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 2.0)
        frames.append(blob + noise * torch.randn(64, 64))
    return torch.stack(frames)


def test_detect_peaks_finds_blob():
    x = _synthetic_sequence()
    pk = detect_peaks(x[0])
    assert len(pk) >= 1
    assert abs(pk[0][0] - 10.0) <= 1 and abs(pk[0][1] - 20.0) <= 1


def test_linking_and_consistency_constant_velocity():
    x = _synthetic_sequence()
    tracks = link_tracks([detect_peaks(f) for f in x])
    assert persistence(tracks, 16) > 0.99
    # integer-pixel peak detection of a blob moving 0.5 px/frame produces
    # alternating +/-1 rounding jitter in the second difference (~0.7);
    # still orders of magnitude below teleporting motion
    assert velocity_consistency(tracks) < 1.0


def test_consistency_penalizes_teleporting():
    x1 = _synthetic_sequence()
    torch.manual_seed(1)
    # teleporting blob: random position each frame
    frames = []
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    pos = torch.rand(16, 2) * 40 + 10
    for l in range(16):
        blob = 30 * torch.exp(-((rr - pos[l, 0])**2 + (cc - pos[l, 1])**2) / 2.0)
        frames.append(blob + 0.1 * torch.randn(64, 64))
    x2 = torch.stack(frames)
    t1 = link_tracks([detect_peaks(f) for f in x1])
    t2 = link_tracks([detect_peaks(f) for f in x2], gate=100.0)
    assert velocity_consistency(t2) > 10 * max(velocity_consistency(t1), 1e-6)


def test_marginal_l1_identical_is_zero():
    x = _synthetic_sequence()
    assert marginal_l1(x[None], x[None]) < 1e-9


def test_evaluate_sequences_keys():
    x = _synthetic_sequence()[None]
    out = evaluate_sequences(x, x)
    for k in ("velocity_consistency", "doppler_drift", "persistence",
              "marginal_l1", "mean_tracks_per_seq"):
        assert k in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/eval/metrics.py`:

```python
"""Physics metrics for generated RD sequences (spec §5).

Hard peak detection + greedy nearest-neighbor linking; metrics on the
resulting tracks quantify kinematic plausibility.
"""
import numpy as np
import torch
from scipy.ndimage import maximum_filter


def detect_peaks(frame, threshold_db=12.0, max_peaks=8):
    f = frame.detach().cpu().numpy()
    local_max = (maximum_filter(f, size=3) == f)
    thresh = np.median(f) + threshold_db
    mask = local_max & (f > thresh)
    rs, cs = np.nonzero(mask)
    if len(rs) == 0:
        return torch.zeros(0, 2)
    order = np.argsort(f[rs, cs])[::-1][:max_peaks]
    return torch.tensor(np.stack([rs[order], cs[order]], axis=1), dtype=torch.float)


def link_tracks(peaks_per_frame, gate=3.0):
    tracks = []          # each: list of (frame_idx, (r, d))
    for l, peaks in enumerate(peaks_per_frame):
        unused = list(range(len(peaks)))
        for tr in tracks:
            if tr[-1][0] != l - 1 or not unused:
                continue
            last = torch.tensor(tr[-1][1])
            d = torch.tensor([torch.dist(last, peaks[j]) for j in unused])
            j_best = int(d.argmin())
            if d[j_best] <= gate:
                tr.append((l, tuple(peaks[unused[j_best]].tolist())))
                unused.pop(j_best)
        for j in unused:
            tracks.append([(l, tuple(peaks[j].tolist()))])
    return tracks


def _positions(track):
    return torch.tensor([p for _, p in track])


def velocity_consistency(tracks):
    vals = []
    for tr in tracks:
        if len(tr) < 3:
            continue
        pos = _positions(tr)
        acc = pos[2:] - 2 * pos[1:-1] + pos[:-2]
        vals.append(acc.pow(2).sum(dim=1).mean())
    return float(torch.stack(vals).mean()) if vals else float("nan")


def doppler_drift(tracks):
    vals = []
    for tr in tracks:
        if len(tr) < 2:
            continue
        dop = _positions(tr)[:, 1]
        vals.append((dop[1:] - dop[:-1]).abs().mean())
    return float(torch.stack(vals).mean()) if vals else float("nan")


def persistence(tracks, seq_len):
    if not tracks:
        return 0.0
    spans = torch.tensor([len(tr) for tr in tracks], dtype=torch.float)
    return float((spans >= 0.8 * seq_len).float().mean())


def marginal_l1(x_gen, x_real, bins=64):
    lo = min(x_gen.min().item(), x_real.min().item())
    hi = max(x_gen.max().item(), x_real.max().item())
    hg = torch.histc(x_gen.flatten(), bins=bins, min=lo, max=hi)
    hr = torch.histc(x_real.flatten(), bins=bins, min=lo, max=hi)
    return float((hg / hg.sum() - hr / hr.sum()).abs().sum())


def evaluate_sequences(x_gen, x_real, seq_len=16):
    vc, dd, ps, nt = [], [], [], []
    for seq in x_gen:
        tracks = link_tracks([detect_peaks(f) for f in seq])
        vc.append(velocity_consistency(tracks))
        dd.append(doppler_drift(tracks))
        ps.append(persistence(tracks, seq_len))
        nt.append(len(tracks))
    def _nanmean(v):
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")
    return {
        "velocity_consistency": _nanmean(vc),
        "doppler_drift": _nanmean(dd),
        "persistence": _nanmean(ps),
        "marginal_l1": marginal_l1(x_gen, x_real),
        "mean_tracks_per_seq": sum(nt) / len(nt),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: physics evaluation metrics (tracks, drift, marginals)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9b: CFAR detector-response metric (spec §5 "detector reuse")

**Files:**
- Create: `src/eval/cfar.py`
- Modify: `src/eval/metrics.py` (append one function)
- Test: `tests/test_cfar.py`

**Interfaces:**
- Consumes: log-magnitude sequences `(B, L, 64, 64)`.
- Produces:
  - `src/eval/cfar.py`: `ca_cfar_2d(signal, num_train, num_guard, Pfa) -> ndarray` — copied **verbatim** from `/truenas/home/arigra/permuter/ariGranevich/RDDiffusion/cfar_rd.py` lines 7–37 (drop the `from dataset import *` import; keep only `import numpy as np`).
  - In `src/eval/metrics.py`: `cfar_detection_stats(x_seqs, Pfa=1e-3, num_train=4, num_guard=2) -> float` — mean CA-CFAR detections per frame, computed on linear power `10**(x/10)`. Comparing this number between generated and real sequences is the detector-response check: a generator that fools the detector produces a similar detection count.

- [ ] **Step 1: Write the failing test**

`tests/test_cfar.py`:

```python
import torch
from src.eval.metrics import cfar_detection_stats


def _seq_with_targets(n_targets, L=4):
    torch.manual_seed(0)
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for _ in range(L):
        f = torch.randn(64, 64).abs() * 0.5
        for k in range(n_targets):
            r, c = 10 + 12 * k, 15 + 10 * k
            f = f + 25 * torch.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 1.5)
        frames.append(20 * torch.log10(f + 1e-6))
    return torch.stack(frames)


def test_more_targets_more_detections():
    x0 = _seq_with_targets(0)[None]
    x3 = _seq_with_targets(3)[None]
    d0 = cfar_detection_stats(x0)
    d3 = cfar_detection_stats(x3)
    assert d3 > d0 + 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cfar.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

Create `src/eval/cfar.py` with the verbatim copy of `ca_cfar_2d` as described in Interfaces. Then append to `src/eval/metrics.py`:

```python
def cfar_detection_stats(x_seqs, Pfa=1e-3, num_train=4, num_guard=2):
    """Mean CA-CFAR detections per frame over (B, L, N, K) log-mag input."""
    from src.eval.cfar import ca_cfar_2d
    counts = []
    for seq in x_seqs:
        power = (10 ** (seq / 10)).detach().cpu().numpy()
        for frame in power:
            counts.append(ca_cfar_2d(frame, num_train, num_guard, Pfa).sum())
    return float(sum(counts) / len(counts))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cfar.py -v`
Expected: 1 passed (CA-CFAR is a python double loop; the test uses L=4 to stay fast).

- [ ] **Step 5: Commit**

```bash
git add src/eval/cfar.py src/eval/metrics.py tests/test_cfar.py
git commit -m "feat: CA-CFAR detection-count metric (RDDiffusion reuse)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Note for Task 10 step 7: also report `cfar_detection_stats` for generated vs real val sequences alongside the other metrics.

---

### Task 10: Sampling CLI + Phase-1 training run and exit check

**Files:**
- Create: `src/sample.py`
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: checkpoint format from Task 8 (`{"model": state_dict, "config": cfg}`), `GaussianDiffusion.ddim_sample`, `denormalize`, viz + metrics.
- Produces: `sample.py` with `generate(ckpt_path, n_seq, device, steps=50, cond=None) -> Tensor (n, L, 64, 64)` (denormalized log-magnitude) and CLI `python -m src.sample --ckpt checkpoints/last.pt --n 8 --out samples/` writing `seq_i.png`, `seq_i.gif`, and `metrics.yaml` (metrics vs the val split).

- [ ] **Step 1: Write the failing test**

`tests/test_sample.py`:

```python
import torch
import yaml
from src.dataset import generate_cache
from src.train import train
from src.sample import generate


def test_generate_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2, shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4)
    cfg["train"].update(batch_size=2, epochs=1, ckpt_dir=str(tmp_path / "ckpt"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=3)
    x = generate(str(tmp_path / "ckpt" / "last.pt"), n_seq=1,
                 device=torch.device("cpu"), steps=5)
    assert x.shape == (1, 16, 64, 64)
    assert torch.isfinite(x).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sample.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/sample.py`:

```python
"""Sample sequences from a checkpoint; write viz + metrics."""
import argparse
from pathlib import Path

import torch
import yaml

from src.dataset import RadarSequenceDataset, denormalize
from src.diffusion import GaussianDiffusion
from src.train import build_model


def generate(ckpt_path, n_seq, device, steps=50, cond=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    L = cfg["data"]["seq_len"]
    x = diff.ddim_sample(model, (n_seq, L, 64, 64), device, steps=steps, cond=cond)
    stats = RadarSequenceDataset(cfg["data"]["cache_dir"], "val").stats
    return denormalize(x.cpu(), stats)


if __name__ == "__main__":
    from src.eval.metrics import evaluate_sequences
    from src.viz import sequence_gif, sequence_grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = generate(args.ckpt, args.n, device, steps=args.steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, seq in enumerate(x):
        sequence_grid(seq, out / f"seq_{i}.png")
        sequence_gif(seq, out / f"seq_{i}.gif")

    cfg = torch.load(args.ckpt, map_location="cpu")["config"]
    val = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
    x_real = torch.stack([denormalize(val[i]["x"], val.stats)
                          for i in range(min(len(val), args.n * 4))])
    metrics = evaluate_sequences(x, x_real, seq_len=cfg["data"]["seq_len"])
    with open(out / "metrics.yaml", "w") as fh:
        yaml.safe_dump(metrics, fh)
    print(yaml.safe_dump(metrics))
```

- [ ] **Step 4: Run tests, then commit code**

Run: `python -m pytest tests/test_sample.py -v` — expected: 1 passed.

```bash
git add src/sample.py tests/test_sample.py
git commit -m "feat: DDIM sampling CLI with metrics + viz output

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Generate the real dataset**

Run (background; ~1–2 h CPU):

```bash
python -c "
import yaml
from src.dataset import generate_cache
cfg = yaml.safe_load(open('configs/base.yaml'))
d = cfg['data']
generate_cache(d['cache_dir'], d['n_train'], d['n_val'], seq_len=d['seq_len'],
               seed=d['seed'], shard_size=d['shard_size'],
               frame_interval=d['frame_interval'])
print('done')
"
```

Expected: `data/cache/` contains 20 train shards, 2 val shards, `stats.pt`, `manifest.yaml`. Add `data/` and `checkpoints/` and `samples/` and `wandb/` to `.gitignore`; commit the `.gitignore`.

- [ ] **Step 6: Phase-1 training run**

Run on GPU (background, several hours): `python -m src.train --config configs/base.yaml`
Then: `python -m src.sample --ckpt checkpoints/last.pt --n 16 --out samples/phase1`

- [ ] **Step 7: Phase-1 exit check**

Compute the same metrics on real val data as a reference:

```bash
python -c "
import torch
from src.dataset import RadarSequenceDataset, denormalize
from src.eval.metrics import evaluate_sequences
ds = RadarSequenceDataset('data/cache', 'val')
xs = torch.stack([denormalize(ds[i]['x'], ds.stats) for i in range(64)])
print(evaluate_sequences(xs[:32], xs[32:]))
"
```

Exit criteria (spec §8 Phase 1): overfit test passed (Task 8); generated GIFs show targets persisting and moving smoothly across frames; `samples/phase1/metrics.yaml` has `persistence` within 0.2 of the real-data reference and finite `velocity_consistency`. Record both metric dicts in the commit message or a `samples/phase1/README.md`. **Stop and review with the user before starting Phase 2.**

---

### Task 11: Phase 2 — soft-argmax trajectory + Doppler losses

**Files:**
- Modify: `src/losses.py` (append), `configs/base.yaml` (set `phase: 2`)
- Test: `tests/test_traj_losses.py`

**Interfaces:**
- Consumes: batch dict from `RadarSequenceDataset` (padded `"traj"` `(B, 5, L, 2)`, `"n_targets"` `(B,)`), `x0_hat` `(B, L, N, K)`.
- Produces (append to `src/losses.py`):
  - `soft_positions(x0_hat, traj, half=2, temp=1.0) -> Tensor (B, M, L, 2)` — differentiable target positions: 5×5 window centered on the **ground-truth** integer bin per target/frame; soft-argmax of window intensities gives predicted continuous position (association solved by construction; spec §3 deviation).
  - `doppler_centroids(x0_hat, traj, half=2) -> Tensor (B, M, L)` — intensity-weighted mean Doppler coordinate within the window.
  - `traj_consistency_loss(pos, mask) -> scalar` — mean squared second difference over frames (ℓ = 3..L), masked to real targets.
  - `doppler_consistency_loss(cent, mask) -> scalar` — mean squared first difference (ℓ = 2..L), masked.
  - `traj_loss_from_batch(x0_hat, batch, tr_cfg, device) -> scalar` — the combination `λ_traj·L_traj + λ_Doppler·L_Doppler` used by `train.py` (forward-referenced in Task 8).

- [ ] **Step 1: Write the failing test**

`tests/test_traj_losses.py`:

```python
import torch
from src.losses import (soft_positions, doppler_centroids,
                        traj_consistency_loss, doppler_consistency_loss,
                        traj_loss_from_batch)


def _blob_sequence(traj_bins):
    """(M, L, 2) integer trajectory -> (1, L, 64, 64) map with blobs."""
    M, L, _ = traj_bins.shape
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for l in range(L):
        f = torch.zeros(64, 64)
        for m in range(M):
            r, c = traj_bins[m, l]
            f = f + 10 * torch.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 1.5)
        frames.append(f)
    return torch.stack(frames)[None]


def test_soft_positions_recover_blobs():
    ell = torch.arange(8, dtype=torch.float)
    traj = torch.stack([10 + ell, 20 + 0.5 * ell], dim=-1)[None]     # (1, 8, 2)
    x = _blob_sequence(traj)
    pos = soft_positions(x, traj[None], half=2, temp=0.5)            # (1, 1, 8, 2)
    assert (pos[0, 0] - traj[0]).abs().max() < 0.5


def test_traj_loss_zero_for_constant_velocity():
    ell = torch.arange(8, dtype=torch.float)
    traj = torch.stack([10 + ell, 20 + 0.5 * ell], dim=-1)[None][None]
    x = _blob_sequence(traj[0])
    pos = soft_positions(x, traj, half=2, temp=0.5)
    mask = torch.ones(1, 1)
    assert traj_consistency_loss(pos, mask).item() < 0.05


def test_gradients_flow():
    ell = torch.arange(8, dtype=torch.float)
    traj = torch.stack([10 + ell, 20 + 0.5 * ell], dim=-1)[None][None]
    x = _blob_sequence(traj[0]).requires_grad_(True)
    pos = soft_positions(x, traj, half=2, temp=0.5)
    cent = doppler_centroids(x, traj, half=2)
    mask = torch.ones(1, 1)
    (traj_consistency_loss(pos, mask)
     + doppler_consistency_loss(cent, mask)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_traj_loss_from_batch_masks_padding():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64, 64)
    batch = {
        "traj": torch.rand(2, 5, 8, 2) * 50 + 5,
        "n_targets": torch.tensor([1, 2]),
    }
    tr = {"lambda_traj": 0.01, "lambda_doppler": 0.01}
    loss = traj_loss_from_batch(x, batch, tr, torch.device("cpu"))
    assert torch.isfinite(loss) and loss.item() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_traj_losses.py -v`
Expected: FAIL with `ImportError` (missing functions)

- [ ] **Step 3: Implement**

Append to `src/losses.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_traj_losses.py tests/test_losses.py tests/test_train_smoke.py -v`
Expected: all pass (smoke test still on phase 1).

- [ ] **Step 5: Enable phase 2 and commit**

Edit `configs/base.yaml`: `phase: 2`.

```bash
git add src/losses.py tests/test_traj_losses.py configs/base.yaml
git commit -m "feat: soft-argmax trajectory and Doppler consistency losses

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Phase-2 training run and exit check**

Retrain: `python -m src.train --config configs/base.yaml`, then
`python -m src.sample --ckpt checkpoints/last.pt --n 16 --out samples/phase2`.

Exit criteria (spec §8 Phase 2): `velocity_consistency` and `doppler_drift` in `samples/phase2/metrics.yaml` improve vs `samples/phase1/metrics.yaml`, with `marginal_l1` not degrading by more than 20%. If not, sweep λ_traj/λ_Doppler in {0.001, 0.01, 0.1} before concluding. **Stop and review with the user before starting Phase 3.**

---

### Task 12: Phase 3 — conditioning module + DiT/train integration

**Files:**
- Create: `src/conditioning.py`
- Modify: `src/train.py` (build conditioning from batch when `phase >= 3`), `configs/base.yaml` (set `phase: 3`, add `train.cond_dropout: 0.1`)
- Test: `tests/test_conditioning.py`

**Interfaces:**
- Consumes: batch fields `"v0"`, `"acc"`, `"cls"` (padded `(B, 5)`), `"env"` `(B, 3)`, `"n_targets"` `(B,)`; `TemporalDiT.forward(..., cond=(B, dim))`.
- Produces: `ConditionEncoder(dim=256, n_classes=3, max_targets=5)` — `forward(batch, device, dropout_p=0.0) -> Tensor (B, dim)`:
  - motion: per-target `[v0, acc]` through `phi_motion` MLP (2→dim→dim), masked mean-pool over real targets
  - env: 3→dim MLP
  - class: `nn.Embedding(n_classes, dim)`, masked mean-pool
  - fusion: concat (3·dim) → MLP → dim
  - CFG dropout: with probability `dropout_p` per sample, replace the output row with a learned `null_cond` vector. `null(batch_size, device) -> (B, dim)` returns the null vector for unconditional sampling.

- [ ] **Step 1: Write the failing test**

`tests/test_conditioning.py`:

```python
import torch
from src.conditioning import ConditionEncoder


def _batch(B=4):
    return {
        "v0": torch.randn(B, 5), "acc": torch.randn(B, 5) * 0.3,
        "cls": torch.randint(0, 3, (B, 5)),
        "env": torch.randn(B, 3),
        "n_targets": torch.tensor([1, 2, 5, 3][:B]),
    }


def test_output_shape():
    enc = ConditionEncoder(dim=64)
    c = enc(_batch(), torch.device("cpu"))
    assert c.shape == (4, 64)


def test_padding_invariance():
    """Changing padded (unused) target slots must not change the output."""
    torch.manual_seed(0)
    enc = ConditionEncoder(dim=64)
    b1 = _batch()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b1.items()}
    b2["v0"][0, 1:] = 99.0     # sample 0 has n_targets=1; slots 1..4 are pad
    b2["acc"][0, 1:] = 99.0
    c1 = enc(b1, torch.device("cpu"))
    c2 = enc(b2, torch.device("cpu"))
    assert torch.allclose(c1[0], c2[0], atol=1e-6)


def test_cfg_dropout_yields_null():
    torch.manual_seed(0)
    enc = ConditionEncoder(dim=64)
    c = enc(_batch(), torch.device("cpu"), dropout_p=1.0)
    null = enc.null(4, torch.device("cpu"))
    assert torch.allclose(c, null)


def test_feeds_dit():
    from src.dit import TemporalDiT
    m = TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=4,
                    dim=64, depth=2, heads=4)
    enc = ConditionEncoder(dim=64)
    x = torch.randn(4, 4, 16, 16)
    out = m(x, torch.tensor([1, 2, 3, 4]), cond=enc(_batch(), torch.device("cpu")))
    assert out.shape == (4, 4, 16, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_conditioning.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/conditioning.py`:

```python
"""Motion / environment / class conditioning encoder (spec §4)."""
import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, dim=256, n_classes=3, max_targets=5):
        super().__init__()
        self.phi_motion = nn.Sequential(nn.Linear(2, dim), nn.SiLU(),
                                        nn.Linear(dim, dim))
        self.phi_env = nn.Sequential(nn.Linear(3, dim), nn.SiLU(),
                                     nn.Linear(dim, dim))
        self.cls_emb = nn.Embedding(n_classes, dim)
        self.fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(),
                                    nn.Linear(dim, dim))
        self.null_cond = nn.Parameter(torch.zeros(dim))

    def null(self, batch_size, device):
        return self.null_cond.to(device).expand(batch_size, -1)

    def forward(self, batch, device, dropout_p=0.0):
        v0 = batch["v0"].to(device)
        acc = batch["acc"].to(device)
        cls = batch["cls"].to(device)
        env = batch["env"].to(device)
        n = batch["n_targets"].to(device)
        B, Mmax = v0.shape
        mask = (torch.arange(Mmax, device=device)[None] < n[:, None]).float()
        mdenom = mask.sum(dim=1, keepdim=True).clamp(min=1)

        motion = self.phi_motion(torch.stack([v0, acc], dim=-1))     # (B, M, d)
        motion = (motion * mask[..., None]).sum(dim=1) / mdenom
        cemb = (self.cls_emb(cls) * mask[..., None]).sum(dim=1) / mdenom
        fused = self.fusion(torch.cat([motion, self.phi_env(env), cemb], dim=-1))

        if dropout_p > 0:
            drop = (torch.rand(B, device=device) < dropout_p)[:, None]
            fused = torch.where(drop, self.null(B, device), fused)
        return fused
```

- [ ] **Step 4: Integrate into train.py**

In `src/train.py`, inside `train(...)` after `model = build_model(cfg, device)` add:

```python
    encoder = None
    if tr.get("phase", 1) >= 3:
        from src.conditioning import ConditionEncoder
        encoder = ConditionEncoder(dim=cfg["model"]["dim"]).to(device)
        opt_params = list(model.parameters()) + list(encoder.parameters())
    else:
        opt_params = list(model.parameters())
```

Change the optimizer line to use `opt_params`. In the step, replace `eps_hat = model(xt, t)` with:

```python
            cond = (encoder(batch, device, dropout_p=tr.get("cond_dropout", 0.1))
                    if encoder is not None else None)
            eps_hat = model(xt, t, cond)
```

And save the encoder in the checkpoint: `{"model": ..., "encoder": encoder.state_dict() if encoder else None, "config": cfg}` (both save sites).

- [ ] **Step 5: Run tests**

First pin the smoke test to phase 1 so the base.yaml flip in Step 6 cannot break it: in `tests/test_train_smoke.py`, add `cfg["train"]["phase"] = 1` inside `_tiny_config` (right after the other `cfg["train"].update(...)` line).

Run: `python -m pytest tests/test_conditioning.py tests/test_train_smoke.py -v`
Expected: all pass.

- [ ] **Step 6: Enable phase 3 and commit**

Edit `configs/base.yaml`: set `phase: 3`, add `cond_dropout: 0.1` under `train:` (the smoke test was already pinned to phase 1 in Step 5).

Run: `python -m pytest tests/ -v` — expected: all pass.

```bash
git add src/conditioning.py src/train.py tests/ configs/base.yaml
git commit -m "feat: motion/env/class conditioning with CFG dropout

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: CFG sampling + conditioning-adherence evaluation

**Files:**
- Modify: `src/sample.py` (CFG + conditioned generation), `src/eval/metrics.py` (append adherence metric)
- Test: `tests/test_adherence.py`

**Interfaces:**
- Consumes: checkpoint with `"encoder"` state (Task 12), `ConditionEncoder`, tracks from `link_tracks`.
- Produces:
  - In `src/sample.py`: `generate_conditioned(ckpt_path, batch, device, steps=50, guidance=2.0) -> Tensor` — classifier-free guidance: `eps = eps_null + g·(eps_cond − eps_null)` via a wrapper model calling the DiT twice; `batch` is a conditioning dict (same schema as dataset items, batched).
  - In `src/eval/metrics.py`: `velocity_adherence(x_gen, v0_cmd, frame_interval=0.5, dv=0.2496006389776358, v_min=-7.987220447284345) -> float` (defaults = the steering-matrix Doppler grid, `dv = c/(2·fc·K·T0)`, `v_min = −32·dv`) — for each sequence, take the longest track, convert its mean Doppler coordinate to m/s via `v = v_min + dv * doppler_bin`, compare to commanded `v0`; returns the Pearson correlation between commanded and measured velocity across the batch.

- [ ] **Step 1: Write the failing test**

`tests/test_adherence.py`:

```python
import torch
from src.eval.metrics import velocity_adherence


def _blob_seq(dop_bin, L=16):
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for l in range(L):
        r = 10 + 0.5 * l
        frames.append(30 * torch.exp(-((rr - r) ** 2 + (cc - dop_bin) ** 2) / 2.0)
                      + 0.1 * torch.randn(64, 64))
    return torch.stack(frames)


def test_velocity_adherence_perfect():
    torch.manual_seed(0)
    v_cmd = torch.tensor([-5.0, -2.0, 1.0, 4.0, 7.0])
    dop_bins = ((v_cmd - (-7.987220447284345)) / 0.2496006389776358).round()
    x = torch.stack([_blob_seq(b) for b in dop_bins])
    corr = velocity_adherence(x, v_cmd)
    assert corr > 0.95


def test_velocity_adherence_random_is_low():
    torch.manual_seed(1)
    v_cmd = torch.tensor([-5.0, -2.0, 1.0, 4.0, 7.0])
    dop_bins = torch.randint(5, 59, (5,)).float()
    x = torch.stack([_blob_seq(b) for b in dop_bins])
    corr = velocity_adherence(x, v_cmd)
    assert abs(corr) < 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_adherence.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement adherence metric**

Append to `src/eval/metrics.py`:

```python
def velocity_adherence(x_gen, v0_cmd, frame_interval=0.5,
                       dv=0.2496006389776358, v_min=-7.987220447284345):
    """Pearson correlation between commanded initial velocity and the
    velocity implied by the longest track's mean Doppler coordinate."""
    measured = []
    for seq in x_gen:
        tracks = link_tracks([detect_peaks(f) for f in seq])
        if not tracks:
            measured.append(float("nan"))
            continue
        tr = max(tracks, key=len)
        dop = _positions(tr)[:, 1].mean()
        measured.append(v_min + dv * float(dop))
    m = torch.tensor(measured)
    ok = torch.isfinite(m)
    if ok.sum() < 3:
        return float("nan")
    a, b = m[ok], v0_cmd[ok].float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
```

- [ ] **Step 4: Implement CFG generation**

Append to `src/sample.py`:

```python
def generate_conditioned(ckpt_path, batch, device, steps=50, guidance=2.0):
    from src.conditioning import ConditionEncoder

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    encoder = ConditionEncoder(dim=cfg["model"]["dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    model.eval(); encoder.eval()

    B = batch["v0"].shape[0]
    with torch.no_grad():
        cond = encoder(batch, device)
        null = encoder.null(B, device)

    def guided(xt, t, _cond=None):
        e_c = model(xt, t, cond)
        e_n = model(xt, t, null)
        return e_n + guidance * (e_c - e_n)

    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    L = cfg["data"]["seq_len"]
    x = diff.ddim_sample(guided, (B, L, 64, 64), device, steps=steps)
    stats = RadarSequenceDataset(cfg["data"]["cache_dir"], "val").stats
    return denormalize(x.cpu(), stats)
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/ -v` — expected: all pass.

```bash
git add src/sample.py src/eval/metrics.py tests/test_adherence.py
git commit -m "feat: CFG conditioned sampling + velocity adherence metric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Phase-3 training run and exit check**

Retrain with `phase: 3`: `python -m src.train --config configs/base.yaml`.
Then build a conditioning sweep and measure adherence:

```bash
python -c "
import torch
from src.sample import generate_conditioned
from src.eval.metrics import velocity_adherence
v = torch.linspace(-6, 6, 16)
batch = {'v0': torch.zeros(16, 5), 'acc': torch.zeros(16, 5),
         'cls': torch.zeros(16, 5, dtype=torch.long),
         'env': torch.tensor([15.0, 5.0, 0.5]).repeat(16, 1),
         'n_targets': torch.ones(16, dtype=torch.long)}
batch['v0'][:, 0] = v
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = generate_conditioned('checkpoints/last.pt', batch, device)
print('velocity adherence corr:', velocity_adherence(x, v))
"
```

Exit criteria (spec §8 Phase 3): velocity-adherence correlation > 0.8 across the sweep; visually, high-ρ vs low-ρ conditioning produces visibly more/less stable clutter in GIFs. Record results in `samples/phase3/README.md`. **Project complete — review with the user.**

---

## Execution notes

- Tasks 1–9 are pure TDD on CPU and need no GPU. Tasks 10, 11 (step 6), 13 (step 6) involve real training runs — run them on the GPU node and expect hours, not minutes.
- Task order is strict: each task's Interfaces block consumes earlier tasks' products.
- Phase gates (end of Tasks 10, 11, 13) require user review before continuing.
