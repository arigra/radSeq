# radSeq — complete technical briefing

*Self-contained as of 2026-09-06. Written to be handed to another reader (human
or model) with no other context. Every number here comes from a committed JSON
result file in `samples/`, listed at the end.*

---

## 1. What the project is

radSeq generates synthetic **16-frame, 64x64 Range-Doppler (RD) map sequences**
with a diffusion model, to produce controllable labelled radar training data
when real labelled data is scarce.

The intended pipeline: a physics simulator supplies volume and free ground
truth; a generative model trained on it learns the *distribution* rather than
the recipe; a small measured collection later moves that model onto a real
sensor's statistics. Simulation buys sample efficiency, measured data buys
fidelity.

The research proposal's central architectural claim is that RD maps have **no
long-range spatial correlation** — a target in one part of the map is
physically independent of targets elsewhere — so full spatial self-attention is
unnecessary and attention should run **over time only**.

## 2. Data

- Simulator: 1-5 targets with constant-acceleration kinematics
  `r(t) = r0 + v0*t + 0.5*a*t^2`; three target classes (steady point,
  Swerling-1 with per-frame exponential power, range-extended with 3 scatterers
  at -3 dB flanks); AR(1) clutter correlated across frames
  (`z_l = rho*z_{l-1} + sqrt(1-rho^2)*eps_l`).
- Output per frame: `20*log10(|RD| + 1e-6)`, i.e. log-magnitude in dB, 64x64.
- Sequence length L = 16, frame interval 0.5 s.
- Ground truth per sequence: continuous (range bin, Doppler bin) per target per
  frame, plus v0, acceleration, class, and environment `[CNR, SCNR, rho]`.
- Cache: **20,000 train / 2,000 validation**, ~5.5 GB, in `data/cache`.
  Normalisation on fixed training statistics: mean 41.479 dB, std 11.065 dB.
  This split is **locked** — every result below is defined on it.

Real data reference statistics (normalised), used throughout as the target:

| quantity | value |
|---|---:|
| std | 1.0043 |
| mean | -0.0072 |
| p99 | 2.4855 |
| p99.9 | 3.8744 |
| range | -7.30 to +5.71 |
| marginal L1, real vs real floor | 0.0647 |
| track persistence | 0.1437 |
| target tracks / sequence | 3.09 |
| velocity inconsistency | 1.358 |
| Doppler drift | 0.5075 |

## 3. Model

Diffusion Transformer (DiT) denoiser, 9.83 M parameters:

- Each 64x64 frame is cut into 8x8 patches at **stride 4**, so patches overlap
  by half: 15 x 15 = **225 patch sites per frame**. Each patch is flattened and
  linearly projected to d = 256. A sequence becomes a (16, 225, 256) tensor.
  The overlap is deliberate: a target crossing a patch boundary stays visible
  inside at least one patch at all times.
- **Attention over time only.** The token tensor is reshaped
  `(B, L, P, d) -> (B*P, L, d)` and self-attention runs over the length-L axis.
  Each of the 225 spatial sites therefore has its own independent temporal
  transformer and never exchanges information with any other site at any depth.
  This is `src/dit.py` line 40.
- 8 blocks, 8 heads, adaLN-Zero conditioning on the timestep.
- Reconstruction: patch predictions are reassembled with `F.fold` and divided
  by hit counts, i.e. **overlapping predictions are averaged**. Every pixel is
  predicted by four patches.

## 4. Diffusion and losses

- DDPM, T = 1000, cosine schedule, **epsilon-prediction**.
- Sampling: DDIM, 50 steps by default.
- `pred_x0(xt, t, eps) = (xt - sqrt(1-abar)*eps) / sqrt(abar)`. Note
  `sqrt(abar) = 4.9e-5` at t = 999, so epsilon error is amplified ~20,000x
  there. A **clamp of x0 to +/-4 sigma** is applied in the sampler and in the
  training loss; it is load-bearing for stability (see §8.2).
- Loss: `L_dit + lambda_smooth * L_smooth (+ physics terms in phase 2+)`, with
  `L_dit` the epsilon MSE and

  `L_smooth = E[(1 - abar_t) * sum_l || x0hat^(l+1) - x0hat^(l) ||^2]`,
  `lambda_smooth = 0.1`.

  The `(1 - abar_t)` factor makes this penalty strongest at **high noise**.
- Training: AdamW, lr 1e-4, batch 16, seed 2026, one RTX 3090.

**Phases** (from the locked design spec):

1. Backbone: `L_dit + L_smooth`, unconditional. **Trained.**
2. Physics: add differentiable soft-argmax trajectory and Doppler losses.
   Implemented, never trained.
3. Conditioning: motion / class / environment with classifier-free guidance.
   Implemented, never trained. Conditioning is a single global vector added to
   the timestep embedding and carries **no spatial layout**, so it cannot by
   itself tell a site to be a target.

## 5. Evaluation

- **marginal L1**: L1 distance between 64-bin normalised histograms of all
  pixel values, generated vs real. Range [0, 2]. Real-vs-real floor 0.0647.
- **Track metrics**: peaks detected per frame at `median + 12 dB`, keeping the
  **5 strongest**, greedily linked into tracks, keeping tracks spanning at least
  `L/2 = 8` frames. This detector was validated against ground truth at
  **93.5% precision / 90.7% recall**.
  - `n_target_tracks_per_seq` — real reference 3.09
  - `persistence` (higher better) — real 0.1437
  - `velocity_consistency` (lower better) — real 1.358
  - `doppler_drift` (lower better) — real 0.5075
- **Grid seam ratio**: mean |neighbour difference| on an 8-pixel periodic grid
  divided by the same off-grid. 1.0 = seamless; real data 0.999.

**Important history**: the original detector kept 8 peaks with no track-length
filter, giving 45.4 tracks/sequence against ~3.2 true targets — **19.9%
precision**. Every kinematic number reported before 2026-08-07 was averaged
over tracks that were ~80% clutter and is **not comparable** to anything here.
Correcting it inverted the project's headline result: the model went from
appearing better than real data on motion metrics to worse on all of them.

## 6. The central finding

The trained Phase-1 model **restores but does not synthesise**.

- Given a *real* sequence noised to a known t and asked to recover it, the
  network returns x0hat with **std 0.986** against real 1.003 — essentially
  perfect, bright tail included.
- Given only noise and asked to generate, the same weights produce
  **std 0.655**, with the bright tail where targets live missing
  (p99 1.32 vs 2.49 real).
- This is flat from epoch 10 to epoch 70, so it is not undertraining, and it is
  unchanged at 50/100/250 DDIM steps or full ancestral sampling, so it is not a
  sampling budget artifact.

Generated frames contain roughly the same *number* of bright blobs as real
(47.9 vs 46.2 per frame) but each is **3.6x smaller** (4.8 px vs 17.4 px). The
model scatters isolated specks instead of forming extended structures.

## 7. Where the failure lives along the noise schedule

Fixed held-out sequences, fixed paired noise, denoised at 11 timesteps, with
every error split into ground-truth **target windows** (half-width 2 bins) and
**clutter** outside a 4-bin guard:

| t | target/clutter eps-MSE | x0hat std in target windows | target-frame recall |
|---:|---:|---:|---:|
| 50 | 0.70 | 1.222 | 0.953 |
| 100 | 0.64 | 1.216 | 0.947 |
| 200 | 0.63 | 1.203 | 0.945 |
| 400 | 0.78 | 1.148 | 0.926 |
| 600 | 1.28 | 1.018 | 0.854 |
| 800 | 2.74 | 0.755 | 0.567 |
| 950 | 4.83 | 0.483 | 0.092 |

Real target-window std is 1.225; real target-frame recall 0.950.

Reading: up to t = 400 target regions are *easier* to denoise than clutter,
which is expected — a bright compact target is more predictable than a
fluctuating background. From t = 600 the ratio crosses 1 and runs away.

**The decisive number**: one-step x0hat from near-pure noise has std **0.604**,
and full 50-step DDIM synthesis produces **0.655**. They are the same number.
The deficit is *set in the first few reverse steps*, at exactly the noise levels
where the reverse process decides **where targets go** — not accumulated over
sampling.

An independent check confirmed the diffusion machinery itself is correct:
`GaussianDiffusion` was exercised against analytically specified denoisers on
known 1-D distributions (schedule monotone; q_sample marginals match closed
form; q_sample/pred_x0 round trip exact to 8e-12; DDIM with an oracle denoiser
recovers a Gaussian to 2.6% and a two-point distribution to 0.15%). A sampler or
objective bug is ruled out.

## 8. Every experiment run, and what it showed

All arms below use a **matched 12,500-step budget** (10 epochs at batch 16),
50-step DDIM at sampling seed 1, n = 32 generated sequences, unless stated.
Baseline is `checkpoints/phase1_wandb/epoch_0010.pt`.

### 8.1 Architecture (pre-registered control) — REFUTED the leading hypothesis

The hypothesis was that temporal-only attention causes the collapse: with no
cross-spatial information flow, no site can agree with any other that *this*
location is a target, so under a squared-error objective every site hedges
toward the conditional mean. `FactorizedBlock` adds a spatial-attention
sublayer; everything else is shared code.

Decision rule committed **before** results existed: generated std above **0.85**
in either arm confirms the diagnosis.

| arm | attention | depth | params | std | marg. L1 |
|---|---|---:|---:|---:|---:|
| baseline | temporal only | 8 | 9.83 M | 0.655 | 0.380 |
| control d5 | factorized | 5 | **8.58 M** | 0.704 | 0.909 |
| control d6 | factorized | 6 | 10.22 M | 0.700 | 0.456 |

Neither cleared 0.85, including the depth-5 arm which has **fewer** parameters
than the baseline (so "the new model is just bigger" was excluded by design).
The claim "the intensity gap is architectural" is **withdrawn**.

A conventional **convolutional U-Net** denoiser (5.81 M params, same data,
schedule, loss and budget) reaches std 0.674 — beside the transformer. A
completely different inductive bias fails the same way, so the failure is not
specific to patch transformers either.

### 8.2 The +/-4 sigma x0 clamp — NOT a fix; it is load-bearing

The clamp alters **4.5% of ground-truth target pixels** and drops target-frame
recall from 0.951 to **0.774** on data the model reconstructed near-perfectly,
so it does truncate real signal. But:

| clamp | std | mean | marg. L1 |
|---|---:|---:|---:|
| +/-4 (current) | 0.655 | -0.266 | 0.380 |
| +/-6 | 0.694 | -0.400 | 0.384 |
| +/-8 | 0.768 | -0.508 | 0.382 |
| **off** | **1134.3** | **-220.0** | 2.00 |

Removing it makes DDIM diverge, for the reason in §4 (division by
sqrt(abar) = 4.9e-5). Widening it raises std but leaves marginal L1 flat at 0.38
and drifts the mean downward, i.e. it adds spread to the *lower* tail, not
target contrast. **Keep the clamp.**

### 8.3 Patch reconstruction — a real, replicated effect

Every pixel is predicted by four overlapping patches and the four are averaged.
Three reassembly rules, same weights, same initial noise:

| decode | std | marg. L1 | persistence | tgt/seq | seam ratio |
|---|---:|---:|---:|---:|---:|
| real | 1.004 | 0.065 | 0.144 | 3.09 | 1.00 |
| mean (default) | 0.655 | 0.380 | 0.026 | 1.59 | 1.12 |
| tile (non-overlapping subset) | 0.830 | 0.239 | 0.021 | 1.44 | 1.77 |
| hann (raised-cosine weighted) | 0.612 | 0.552 | 0.050 | 2.50 | 0.91 |

Retraining at stride 8 so the model owns non-overlapping patches gives std
0.744, marg. L1 0.354, but persistence 0.008 and 0.69 tracks/seq — the overlap
is doing real work for target continuity.

**Replication across four independent sampling seeds** (same checkpoint, only
the decode rule differing):

| seed | mean tgt/seq | hann tgt/seq | delta |
|---:|---:|---:|---:|
| 1 | 1.59 | 2.50 | +0.91 |
| 2 | 2.03 | 2.47 | +0.44 |
| 3 | 1.81 | 2.22 | +0.41 |
| 4 | 1.56 | 2.19 | +0.62 |

mean decode **1.75 +/- 0.22**, hann **2.34 +/- 0.16**, real reference 3.09.
Every seed favours hann and the gap exceeds either arm's spread. Marginal L1 is
consistently worse for hann (0.55 vs 0.39).

**This is the most robust result in the project.** It is a single isolated
variable with a replicated effect on the metric that matters, and it concerns a
reconstruction detail that patch-based diffusion papers do not report.

### 8.4 The temporal smoothness loss — hypothesis inverted

Since the failure is localised at high noise and `L_smooth` is weighted by
`(1 - abar_t)` (maximal at high noise), the hypothesis was that this penalty
suppresses target placement. Four arms:

| arm | lambda | schedule | std | marg. L1 | tgt/seq |
|---|---:|---|---:|---:|---:|
| real | - | - | 1.004 | 0.065 | 3.09 |
| baseline | 0.1 | (1 - abar) | 0.655 | 0.380 | 1.59 |
| off | 0.0 | - | 1.285 | 0.533 | 0.34 |
| inverted | 0.1 | abar | 1.281 | **0.339** | 0.72 |
| uniform | 0.1 | 1 | 0.631 | 0.423 | 1.62 |
| magnitude cut | 0.01 | (1 - abar) | 0.826 | 0.951 | 0.59 |

**The opposite of the hypothesis holds.** Every arm that reduces the high-noise
penalty (off, inverted, magnitude cut) raises variance *and destroys targets*.
`uniform` — which keeps full weight at high noise and only adds penalty at low
noise — reproduces the baseline within noise. The high-noise smoothness penalty
is what makes targets temporally coherent at all; without it the model emits
bright, high-variance speckle.

Note the `inverted` arm achieves the **best marginal L1 of any arm measured**
(0.339) while producing 0.72 target tracks per sequence against 3.09 real. A
distribution-matching metric alone would have selected it.

Incidental but sharp: removing `L_smooth` changes the *denoising* objective by
**0.0015** (0.2853 -> 0.2838) while nearly doubling generated std. An auxiliary
term can be invisible in the training loss and decisive in the sampling
distribution.

### 8.5 EMA — no effect, but it produced the noise floor

An EMA arm was trained with a config identical to the baseline except for EMA
tracking, which does not alter the raw weights' trajectory. Its **raw** weights
are therefore an accidental exact replicate of the baseline run:

- baseline: std 0.655
- replicate: std 0.713

**Run-to-run variability in generated std is ~0.06 at a fixed seed**, from GPU
nondeterminism alone. EMA weights give 0.717, i.e. within noise of raw. A second
noise floor comes from §8.3: `n_target_tracks_per_seq` varies **1.56-2.03**
(sd 0.22) across sampling seed for a fixed checkpoint.

Consequences: both factorized control arms' leads (+0.049, +0.045) and EMA's
(+0.058) are inside noise. Any single-seed difference below ~0.1 in std or ~0.2
in tracks is not a result.

## 9. Claims that were made and then withdrawn

Recorded because they are part of the evidence:

1. **"The intensity gap is architectural"** (2026-08-07). Withdrawn after the
   pre-registered control failed and a conv U-Net failed identically.
2. **"Removing the x0 clamp is a free fix"** (2026-09-06). Withdrawn: removal
   makes sampling diverge.
3. **"Distributional fidelity and detection utility are anti-correlated across
   arms"** (2026-09-06). Withdrawn twice. Built initially from four hand-picked
   rows. Recomputed over all arms, dropping diverged runs and collapsing
   clamp-width variants of the same model+decode (these are one condition
   sampled repeatedly, not independent arms), leaving 13 independent
   conditions:

   | relationship | Spearman rho | permutation p |
   |---|---:|---:|
   | marginal L1 vs target tracks | -0.148 | 0.63 |
   | marginal L1 vs persistence | -0.330 | 0.27 |
   | \|std - 1.004\| vs target tracks | +0.544 | 0.06 |
   | \|std - 1.004\| vs persistence | +0.341 | 0.26 |

   Nothing reaches significance and the L1 relationship changed sign between
   analyses. **No correlation claim is supported.** The §8.3 decode result
   stands on its own as a single-variable comparison and needs no correlation.

4. Phase 1's written exit criterion ("persistence within 0.2 of reference")
   is **unfalsifiable** — the reference is 0.1437, so a 0.2 tolerance admits
   anything from 0 to 0.34, including a model producing no persistent tracks.
   It should never be cited as evidence.

## 10. Current state, in one paragraph

Phase 1 trains cleanly and restores near-perfectly but synthesises at roughly
two thirds of the real intensity spread with the target tail missing. The
failure is confined to t >= 600 and is set in the first reverse steps. It is not
the diffusion machinery, not the attention topology, not patch-transformers in
general, not EMA, and not the clamp. Roughly half the intensity deficit is
produced by the decode path; the auxiliary smoothness loss is the largest single
lever on variance but trades directly against target coherence. The best
configuration found is the baseline weights with raised-cosine decode: 2.34
target tracks/sequence against 3.09 real, at a worse intensity histogram. No
configuration is good at both.

## 11. The next experiment (planned, not yet run)

Every intervention so far has been *around* the denoiser — the clamp, the
decode rule, the auxiliary loss, the architecture. The **objective itself is
untouched**. At t >= 600 epsilon-prediction carries almost no gradient about
where structure belongs, which is exactly the regime where the failure lives.
The standard remedies target that directly:

- **v-parameterization** (predict `v = sqrt(abar)*eps - sqrt(1-abar)*x0`
  instead of epsilon)
- **min-SNR loss weighting** (stop low-t terms dominating the objective)

Both are config-level changes at the same budget, so they remain honest
single-variable arms. Planned with **three seeds each**, given the noise floors.

Decision rule to be fixed before running: an arm counts as a fix only if
generated std exceeds ~0.85 **while keeping target tracks/seq at or above the
baseline's 1.75**, since every previous variance gain came at the cost of
targets.

If neither works — after architecture, decode, clamp and objective have all been
eliminated — the honest conclusion is that unconditional synthesis is not
reachable in this setup, and the project should reframe around **restoration**
(denoising, super-resolution, conditional infill on measured data), where the
same model demonstrably works at 0.986.

## 12. What a publishable contribution would still need

- Working synthesis, or an explicit reframing to restoration.
- Phase 3 conditioning trained — it is the actual selling point and has never
  been run.
- Validation on measured data (RADIal is available locally); everything so far
  is simulator-on-simulator.
- A downstream experiment: train a detector on generated data and show it helps.
  Nothing to date demonstrates the generated data is *useful*, only that it is
  or isn't distributionally close.
- An external baseline; all comparisons so far are against the project's own
  arms.
- Three seeds on every claim.

## 13. Where things are in the repository

Code that matters (about 400 lines total):

| file | lines | what |
|---|---:|---|
| `src/diffusion.py` | 112 | schedule, q_sample, pred_x0, DDIM, clamp, loss weighting |
| `src/dit.py` | 133 | the model; line 40 is the temporal-only reshape |
| `src/patching.py` | 81 | patchify / unpatchify; mean, tile and hann decode |
| `src/losses.py` | 75 | epsilon loss, smoothness loss, phase-2 physics losses |
| `src/eval/metrics.py` | 156 | detector, track linking, all reported metrics |

Harness (read as needed): `train.py` 324, `simulator.py` 279, `sample.py` 136,
`unet.py` 102, `dataset.py` 86, `viz.py` 89, `ema.py` 24, `conditioning.py` 39.

Result files, every number above:

- `samples/diag_diffusion_sanity.json` — model-independent pipeline checks
- `samples/diag_denoising_by_timestep.json` — the timestep/region diagnostic
- `samples/control_comparison.json` — the pre-registered attention control
- `samples/ablation_single_variable.json` — no-smooth, EMA, conv U-Net
- `samples/overlap_reduction_comparison.json`, `ablation_no_overlap.json` — decode
- `samples/fix_decode_and_clamp.json`, `fix_clamp_width.json` — clamp and hann
- `samples/fix_smoothness_schedule.json` — the smoothness arms
- `samples/decode_seed{2,3,4}.json` — the decode seed replication
- `samples/diag_marginal_l1.json`, `diag_tail_origin.json`,
  `diag_track_validity.json`, `phase1_rescore.json` — the original diagnosis

Documents:

- `docs/superpowers/specs/2026-07-18-temporal-radar-dit-design.md` — the locked
  design and the phase exit criteria (165 lines; the actual contract)
- `docs/notes/radseq-project-status.md` — authoritative status
- `docs/notes/2026-08-07-phase1-marginal-l1-diagnosis.md` — first diagnosis
- `docs/notes/2026-09-05-post-control-preregistered-plan.md` — the plan the
  ablations answer
- `notebooks/01_research_story.ipynb` — narrative version with figures
- `prop.md` — the research proposal

Environment: one local RTX 3090 (24 GB). A 12,500-step arm takes ~55 minutes.
85+ tests pass (`python -m pytest`).
