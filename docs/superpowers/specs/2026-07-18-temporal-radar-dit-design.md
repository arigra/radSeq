# Temporal Radar Sequence Generation via Diffusion Transformer — Design

**Date:** 2026-07-18
**Source proposal:** `prop.md` (§ Temporal Radar Sequence Generation via Diffusion Transformer)
**Status:** Approved

## Goal

Implement the research in `prop.md`: a Diffusion Transformer (DiT) that generates
temporally coherent sequences of Range-Doppler (RD) maps with physically plausible
target motion, controllable via kinematic, environmental, and class conditioning.
Extends the single-frame RDDiffusion work to sequences.

## Locked decisions

| Decision | Choice |
|---|---|
| Training data | Synthetic — temporal extension of the RDDiffusion simulator |
| Sequence spec | L=16 frames of 64×64 RD maps; targets move ~1–2 bins/frame |
| Scope | Staged: backbone → physics losses → conditioning |
| Backbone | Faithful to proposal: temporal-only attention, overlapping patches |
| Data pipeline | Pre-generate + cache (20k train / 2k val), locked normalization stats |
| Evaluation | Physics metrics + RDDiffusion CFAR/ROC reuse + visual grids/GIFs |
| Code home | New self-contained package in `radSeq/`; simulator copied+adapted from RDDiffusion (no import dependency) |

Approved deviations from the proposal's letter:
1. **adaLN-Zero** (γ, β, plus zero-initialized gate α per Peebles et al.) instead of plain adaLN.
2. **GT-window soft-argmax** for the differentiable trajectory/Doppler losses instead of
   thresholded local-maxima detection (which is non-differentiable and has an unsolved
   association problem). Hard peak detection is still used at evaluation time.

## 1. Temporal simulator

`src/simulator.py` — `TemporalRadarDataset`, adapted from RDDiffusion's
`radar_dataset.py` (copied, not imported).

- **Targets:** M ∈ {1..5} per sequence. Initial range/velocity sampled as in RDDiffusion;
  add per-target acceleration `a`. Kinematic evolution per frame ℓ:
  `r_ℓ = r₀ + v·ℓT_f + ½aℓ²T_f²`; Doppler bin tracks instantaneous radial velocity
  `v_ℓ = v₀ + aℓT_f`. Inter-frame interval `T_f` chosen so typical targets traverse
  1–2 range/Doppler bins per frame across L=16 frames.
- **Clutter:** AR(1) evolution on the correlated-speckle skeleton with coefficient
  ρ ∈ [0,1] (ρ=0 independent frames, ρ→1 frozen clutter). Texture per existing
  K/Weibull/Pareto machinery.
- **Classes** (used in Phase 3): {steady point target, Swerling-1 fluctuating,
  range-extended 3-bin target}.
- **Output per sequence:** log-magnitude RD sequence (16×64×64) + ground truth:
  per-target per-frame (range-bin, Doppler-bin) trajectory, (v₀, a), class id,
  and scene params (σ_clutter, SCR, ρ).

**Data pipeline** (`src/dataset.py`): pre-generate 20k train / 2k val sequences into
`.pt` shards; normalization stats computed on the train split and locked
(RDDiffusion normalize-and-cache pattern); fixed seed; a manifest file records all
simulator parameters.

## 2. Backbone — temporal-only DiT

`src/patching.py`, `src/dit.py`.

- **Patchify:** overlapping patches, p=8, s=4 on 64×64 → 15×15 = 225 patches/frame,
  patch dim 64; linear projection to d=256.
- **Embeddings:** learnable 2D spatial pos-emb (225 sites), learnable temporal pos-emb
  (16 frames), sinusoidal diffusion-timestep embedding — summed per the proposal.
- **Blocks:** B=8 DiT blocks, H=8 heads, adaLN-Zero conditioned on timestep (+ **c** in
  Phase 3). Attention reshapes tokens to (batch·P, L, d) and attends **only over
  time** — each spatial site is an independent length-16 sequence. Spatial
  independence is exact by construction; cost per block is P·L²·d.
- **Unpatchify:** `F.fold` with overlap-count normalization (overlapped regions
  averaged). Patchify→unpatchify must be the identity on raw maps (unit-tested).
- ~6–7 M parameters.

## 3. Diffusion and losses

`src/diffusion.py`, `src/losses.py`.

- **Process:** DDPM, T=1000, cosine ᾱ schedule, ε-prediction. DDIM sampler for fast
  evaluation.
- **L_DiT:** Frobenius MSE between ε and ε_θ over the full sequence.
- **L_smooth:** ‖x̂₀^(ℓ) − x̂₀^(ℓ−1)‖² summed over ℓ, weighted ω_t = (1−ᾱ_t), with
  x̂₀ recovered from ε_θ via the standard DDPM inversion.
- **L_traj (Phase 2):** for each GT target and frame, a 5×5 window centered on the GT
  trajectory position; predicted position = spatial soft-argmax of x̂₀ intensity in
  the window; loss = second-difference (constant-velocity) penalty on predicted
  positions over ℓ = 3..L.
- **L_Doppler (Phase 2):** Doppler centroid = intensity-weighted mean Doppler bin in
  the same window; loss = first-difference penalty over ℓ = 2..L.
- **Total:** L_DiT + λ_smooth·L_smooth + λ_traj·L_traj + λ_Doppler·L_Doppler.
  Initial values λ_smooth = 0.1, λ_traj = λ_Doppler = 0.01 (config; tuned in Phase 2).

## 4. Conditioning (Phase 3)

`src/conditioning.py`.

- **Motion:** φ_motion(v₀, a) per target, mean-pooled over targets (handles variable M).
- **Environment:** [σ_clutter, SCR, ρ] → small MLP.
- **Class:** learnable class embeddings, pooled with the motion embeddings.
- **Fusion:** MLP_fusion over the concatenation → **c**, injected through adaLN-Zero
  alongside the timestep embedding.
- **CFG:** 10% conditioning dropout during training; classifier-free guidance at
  sampling time.

## 5. Evaluation

`src/eval/metrics.py`, `src/viz.py`.

- **Trajectory physics:** hard peak detection per generated frame; greedy
  nearest-neighbor linking across frames → velocity-consistency error
  (second-difference of tracks), Doppler-centroid drift, track persistence rate
  (fraction of tracks surviving all 16 frames).
- **Marginals:** per-frame intensity histograms and mean/std vs held-out simulator data.
- **Detector reuse:** RDDiffusion CFAR/ROC tooling applied per-frame, generated vs real.
- **Conditioning adherence (Phase 3):** commanded v₀ vs measured track slope;
  commanded SCR vs measured SCR.
- **Visual:** sequence grids and GIFs logged to wandb during training.

## 6. Code layout

```
radSeq/
  prop.md
  configs/base.yaml
  src/
    simulator.py      # TemporalRadarDataset (adapted from RDDiffusion)
    dataset.py        # generation, caching, normalization, loaders
    patching.py       # overlapping patchify / fold-unpatchify
    dit.py            # temporal-only DiT blocks + model
    diffusion.py      # DDPM/DDIM schedules and samplers
    losses.py         # L_DiT, L_smooth, L_traj, L_Doppler
    conditioning.py   # motion/env/class embeddings + fusion (Phase 3)
    train.py          # training loop, wandb logging, checkpoints
    sample.py         # sequence sampling CLI
    viz.py            # grids / GIFs
    eval/metrics.py   # physics metrics, CFAR reuse
  tests/
  docs/superpowers/specs/
```

## 7. Testing

Test-driven per phase. Key unit tests:
- Patchify→unpatchify round-trip identity.
- Simulator GT correctness: RD peak within 1 bin of the analytic trajectory
  position at every frame (leakage-free config).
- Spatial-independence: perturbing site j's input tokens leaves site i's output
  unchanged.
- Soft-argmax gradient flow (nonzero grads through L_traj/L_Doppler).
- AR(1) clutter correlation: empirical inter-frame correlation matches ρ.
- Single-batch overfit smoke test (loss → ~0).

## 8. Phases and exit criteria

1. **Backbone** — simulator, cache, DiT, L_DiT + L_smooth, unconditional training.
   *Exit:* single-batch overfit converges; samples show persistent moving targets;
   full metric suite runs end to end.
2. **Physics losses** — add L_traj + L_Doppler.
   *Exit:* velocity-consistency and Doppler-drift improve over Phase 1 without
   degrading marginal statistics.
3. **Conditioning** — motion/env/class + CFG.
   *Exit:* strong commanded-vs-measured velocity correlation; SCR adherence across
   a sweep.

## Out of scope

Real-data training (RADDet/CARRADA), latent/VAE compression, factorized
spatial-temporal attention (possible later ablation), downstream tracker evaluation.
