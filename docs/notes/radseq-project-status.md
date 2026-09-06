---
title: radSeq — Project Status
tags:
  - radar
  - diffusion
  - research
---

# radSeq

**Status as of 2026-09-06 (revised 23:00).** This file supersedes every earlier status summary.
Metric values recorded before 2026-08-07 used a broken detector and are not
comparable to anything here.

## What it does

radSeq generates synthetic **16-frame, 64x64 Range-Doppler sequences**, to create
controllable radar training data with realistic target motion and clutter when
labeled real data is scarce.

`Simulator -> cached sequences -> diffusion training -> DDIM sampling -> physics/statistical evaluation`

- **Data:** 20,000 train / 2,000 val sequences in `data/cache`, normalized on
  fixed training statistics. This split is locked; every result below is defined
  on it.
- **Model:** DiT over overlapping 8x8 patches (stride 4, 225 sites/frame),
  attention **over time only** at each spatial site.
- **Phases:** 1 backbone (trained), 2 physics losses (implemented, untrained),
  3 conditioning (implemented, untrained).

## Bottom line (revised)

**Target motion is solved by training budget.** Scored on the corrected detector
across three sampling seeds, the 87,500-step checkpoint produces 3.15 target
tracks per sequence against 3.09 real, with velocity consistency 1.46 against
1.36. At the 12,500-step ablation budget it produced 1.81. The intensity
distribution is unchanged across the same 7x (std 0.652 -> 0.655, marginal L1
0.388 -> 0.398).

So there is **one** open problem, the intensity distribution, and it has survived
every structural change tried: 7x budget, spatial attention, a convolutional
denoiser, EMA, every clamp width, every smoothness weighting and magnitude,
v-prediction and min-SNR. The remaining suspects are the data representation
itself — the dB log scaling and the fixed normalisation — not the model.

Two corrections this forces:

- **Every 12,500-step comparison in this project is budget-confounded on the
  target metrics**, including the architecture control and the decode
  comparison. Their distribution verdicts stand; their target verdicts do not.
  The decode comparison in particular *reverses* at full budget: the plain mean
  decode is calibrated (3.15 vs 3.09 real) while hann over-produces (3.99).
- **`best.pt` is the wrong checkpoint.** Validation-selected at step 78,750, it
  scores worse on target metrics than `epoch_0070` at step 87,500. The
  validation objective is not aligned with target quality; train to a fixed
  budget and keep the last checkpoint.

## Bottom line (as written earlier on 2026-09-06)

Phase 1 trains cleanly and **restores** near-perfectly, but **synthesizes** with
roughly two thirds of the real intensity spread, and the bright tail where
targets live is missing. Nine arms at a matched 12,500-step budget now locate
where that deficit comes from, and it is **not the attention topology**:

- Removing the **temporal smoothness loss** moves generated std 0.655 -> **1.285**,
  the largest single effect measured. That loss is weighted by $(1-\bar\alpha_t)$,
  so it is strongest at exactly the high-noise steps where the failure lives.
- Decoding **without overlap averaging** moves std 0.655 -> **0.830** with no
  change to the weights at all.
- Adding **spatial attention** moves it 0.655 -> 0.704, and a **convolutional
  denoiser** lands at 0.674. Both are inside or barely outside measurement noise.

The variance deficit is dominated by what is applied *around* the denoiser — the
high-noise smoothness penalty and the patch-averaging decode — rather than by the
denoiser's architecture.

## Measurement noise floor (new, and it changes how to read everything)

`abl_ema` is an accidental exact replicate of the baseline: identical model,
data, seed 2026, lambda_smooth and budget, differing only by EMA tracking, which
does not alter the raw weights' trajectory. Its raw weights give **0.713** where
the baseline gives **0.655**.

**Run-to-run variability in generated std is therefore about 0.06**, from GPU
nondeterminism alone at a fixed seed. Consequences:

- The factorized arms' +0.049 and +0.045 leads are **within noise**. The control's
  negative result is stronger than it looked.
- The EMA arm's +0.058 is **within noise**. It shows nothing.
- The no-smooth (+0.63) and tile-decode (+0.175) effects are far outside it.
- The pre-registered "material lead" of 0.10 absolute std is only about 1.7x the
  noise floor. Any future single-seed claim near that size is not a result.

## All arms, matched 12,500 steps, 50-step DDIM at seed 1, n = 32

| Arm | Params | std | mean | p99 | marg. L1 | persist (higher better) | tgt/seq |
|---|---:|---:|---:|---:|---:|---:|---:|
| **real data** | — | **1.004** | -0.007 | **2.486** | **0.065** | **0.144** | **3.09** |
| baseline temporal d8 | 9.83 M | 0.655 | -0.266 | 1.324 | 0.380 | 0.026 | 1.59 |
| baseline, tile decode | 9.83 M | **0.830** | -0.281 | 1.940 | **0.239** | 0.021 | 1.44 |
| retrained stride 8 | 9.78 M | 0.744 | -0.038 | 1.756 | 0.354 | 0.008 | 0.69 |
| factorized d5 | 8.58 M | 0.704 | +0.777 | 2.489 | 0.909 | 0.002 | 1.09 |
| factorized d6 | 10.22 M | 0.700 | -0.542 | 1.128 | 0.456 | 0.016 | 1.72 |
| **no smoothness loss** | 9.83 M | **1.285** | +0.763 | 3.729 | 0.533 | 0.000 | 0.34 |
| EMA arm, raw weights | 9.83 M | 0.713 | +0.064 | 1.716 | 0.463 | 0.014 | 1.53 |
| EMA arm, EMA weights | 9.83 M | 0.717 | -0.006 | 1.647 | 0.415 | 0.022 | 1.69 |
| conv U-Net (2D) | 5.81 M | 0.674 | -0.008 | 1.639 | 0.474 | 0.001 | 0.47 |

Read the columns together. No arm both restores the spread *and* keeps targets:
`no_smooth` overshoots the variance and produces essentially no persistent
targets (0.34 target tracks/seq against 3.09 real); `tile` decode buys the
largest honest gain but leaves visible 8-pixel seams (ratio 1.77 against 0.999
for real data).

## Where the failure lives

From the region-resolved timestep diagnostic (`samples/diag_denoising_by_timestep.json`):

| t | target/clutter eps-MSE | x0-hat std in target windows | target-frame recall |
|---:|---:|---:|---:|
| 50 | 0.70 | 1.222 | 0.953 |
| 400 | 0.78 | 1.148 | 0.926 |
| 600 | 1.28 | 1.018 | 0.854 |
| 800 | 2.74 | 0.755 | 0.567 |
| 950 | 4.83 | 0.483 | 0.092 |

Real target-window std is 1.225 and real target-frame recall 0.950. Up to
t = 400 target regions are *easier* than clutter; above t = 600 they become
several times harder and target placement collapses.

One-step x0-hat from near-pure noise has std **0.604**; full 50-step DDIM
synthesis gives **0.655**. The deficit is set in the first reverse steps, not
accumulated — and that is precisely where $(1-\bar\alpha_t)$ makes the smoothness
penalty strongest.

Separately, the sampler's +/-4 sigma x0-hat clamp alters **4.5% of ground-truth
target pixels** and drops target-frame recall from 0.951 to **0.774** on data the
model reconstructed near-perfectly. A guard rail is deleting a fifth of the
signal.

## Established

1. Diffusion and DDIM machinery are correct, verified against analytic denoisers
   on known distributions independently of any trained model.
2. Restoration works (x0-hat std 0.986 against real 1.003); synthesis does not
   (0.655).
3. The earlier "motion is learned" result was a measurement artifact — tracks
   were 19.9% real targets. Corrected to 93.5% precision, the backbone is worse
   than real data on every target-level kinematic metric.
4. Roughly half the variance deficit is produced by the decode path.
5. The temporal smoothness loss is the largest single lever on variance.
6. The remaining failure is confined to t >= 600.
7. Run-to-run noise on the primary metric is about 0.06 std.

## Refuted or unsupported

- **Missing spatial attention is not the established cause.** Both factorized
  arms failed their pre-registered 0.85 threshold, by margins inside measurement
  noise, including a depth-5 arm with fewer parameters than the baseline. The
  "the gap is architectural" claim from 2026-08-07 is withdrawn.
- **Not specific to patch transformers either.** A convolutional denoiser with a
  completely different inductive bias lands at 0.674, near the baseline. Under
  the pre-registered rules this points away from architecture and toward the
  shared objective, schedule and decode.
- **EMA does nothing here** (within noise).
- Phase 1's persistence exit criterion remains unfalsifiable — a 0.2 tolerance
  around a reference of 0.1437 admits any value in 0 to 0.34. Do not cite it.

## Next, in order

1. **Redesign the smoothness term, do not just delete it.** Removing it restores
   variance and destroys targets. Try reducing lambda, dropping the
   $(1-\bar\alpha_t)$ weighting so it stops dominating high noise, or applying it
   to detected tracks rather than whole frames.
2. **Fix the decode.** Weighted or learned patch merging instead of a plain mean.
3. **Remove or widen the +/-4 sigma clamp.** No retraining needed.
4. **Then, the high-noise objective:** v-parameterization or min-SNR weighting,
   aimed at the t >= 600 regime specifically.
5. Any claim from here needs at least 3 seeds, given the 0.06 noise floor.

Phases 2 and 3 stay paused. Conditioning as implemented is a single global vector
added to the timestep embedding; it carries no spatial layout and cannot fix
target placement.

## Key locations

- `docs/superpowers/specs/2026-07-18-temporal-radar-dit-design.md` — the locked design and phase exit criteria
- `docs/notes/2026-09-05-post-control-preregistered-plan.md` — the plan these results answer
- `docs/notes/2026-08-07-phase1-marginal-l1-diagnosis.md` — the earlier diagnosis and the control design
- `notebooks/01_research_story.ipynb` — the narrative version, with figures
- `samples/ablation_single_variable.json`, `control_comparison.json`, `overlap_reduction_comparison.json`, `ablation_no_overlap.json` — every number above
- `samples/diag_denoising_by_timestep.json`, `diag_diffusion_sanity.json` — the diagnostics
- `checkpoints/phase1_wandb/epoch_0010.pt` — the matched-budget baseline; `best.pt` — the full 78-epoch run
