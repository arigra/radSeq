# E0 with a standard video-DiT setup (2026-09-14)

Written before either run finished. Goal (Ari): make the DiT generate the easy
E0 regime (1 steady target, fixed 20 dB, no clutter, no noise) before anything
harder.

## Why the earlier E0 arm collapsed (mean -1.71, std 0.18)

Three things were stacked against it, all fixable:

1. **No spatial attention.** E0 maps are noise-free, so the whole 64x64 map is
   the target's sidelobe pattern (-39 to 107 dB). With temporal-only attention
   each of the 225 patch sites sees only itself over time; from pure noise the
   sites cannot agree where the single target is, so each regresses to the mean.
2. **Smoothness loss** held ~45% of the E0 objective (addendum in
   `2026-09-12-easy-regime-preregistration.md`), pushing toward flat output.
3. **Broken first sampling step** (`2026-09-14-terminal-step-diagnosis.md`).

## Runs

| config | attention | lambda_smooth | params | other |
|---|---|---:|---:|---|
| `experiments/dit_64/configs/e0_fact_s0.yaml` | factorized (temporal + spatial) | 0 | 13.5 M | bf16 |
| `experiments/dit_64/configs/e0_temp_s0.yaml` | temporal only | 0 | 9.8 M | bf16 |

Both: E0 cache, seed 2026, batch 16, lr 1e-4, 12,500 steps, eps-prediction.
Sampling with `terminal_x0=mean`, 50 DDIM steps, n=32, seeds 1-3.
bf16 is new: fp32 factorized needs 14.7 GB and ~14.5 GB is free (another
user's process holds 9 GB).

## Scoring (`experiments/dit_64/score_e0.py`)

Real E0 reference (16 val sequences): std 0.95, peak 105 dB, peak prominence
22 dB over the next peak >4 bins away, 100% of sequences with peak range and
Doppler within 1 bin RMS of a straight line, marginal L1 real-vs-real 0.10.
Old collapsed checkpoint with the sampler fix: std 0.25, peak 34 dB,
prominence 1.2 dB, 6% on a line.

## Decision rule

**Works on E0** if, averaged over seeds: std within 0.1 of real, marginal L1
<= 0.2, peak prominence >= 15 dB, and >= 90% of sequences on a line.

- Factorized works, temporal does not: spatial attention was the missing piece.
- Both work: smoothness loss and/or sampler were the cause.
- Neither works: continue debugging on E0 (budget, parameterization) before
  touching harder data.

## Result at 12,500 steps (`experiments/dit_64/results/e0_final_scores.json`)

Final validation dit loss: factorized 0.0033, temporal 0.0122.
3 seeds x 32 sequences, 50 DDIM steps, terminal_x0=mean. Real: std 1.04, peak
105 dB, prominence 22 dB, 100% on a line, marginal L1 floor 0.064.

| arm | std | peak dB | prominence dB | on line | marginal L1 |
|---|---:|---:|---:|---:|---:|
| factorized, mean decode | 0.56 | 63 | 3.0 | 1% | 0.42 |
| factorized, tile decode | 0.83 | 93 | 2.3 | 3% | 0.31 |
| temporal, mean decode | 0.53 | 68 | 3.6 | 0% | 0.49 |
| temporal, tile decode | 1.27 | 96 | 1.9 | 0% | 0.30 |

**Neither works.** Overlap-mean decoding dulls peaks (tiling restores peak
height), but prominence stays ~2 dB: several near-equal peaks, no single
target. Spatial attention cut validation loss 3.7x without making sampling
coherent.

Next hypothesis: in a noise-free 16-frame sequence the full-map cross is
visible at very low SNR, so the target position is decided in the first 2-3
DDIM steps, where alpha_bar jumps 0 -> 1e-3 -> 4e-3. Coarse steps there blend
several positions. Test: more DDIM steps and ancestral sampling on the same
checkpoint (`experiments/dit_64/results/e0_fact_step_sweep.json`).

## Step count / sampler (`experiments/dit_64/results/e0_fact_step_sweep.json`, factorized, n=16, seed 1)

| decode | DDIM 50 | DDIM 250 | DDIM 1000 | ancestral 1000 |
|---|---|---|---|---|
| mean: std / prominence dB / on line | 0.55 / 3.8 / 0% | 1.52 / 2.7 / 0% | 1.01 / 2.8 / 6% | 2.41 / 1.4 / 0% |
| tile: std / prominence dB / on line | 0.83 / 2.5 / 0% | 1.30 / 1.5 / 0% | 0.56 / 1.4 / 0% | 1.96 / 0.0 / 0% |

**Rejected:** step size at high noise is not the cause.

## Next suspect: the +/-4 x0 clamp

With E0 normalisation (35.15 dB, 15.65 dB) the clamp caps every pixel at
97.7 dB, while the real target peak is ~105 dB (z ~ 4.5). Generated frames sit
against that ceiling at several places at once (temporal tile, frame 0 top five:
97, 97, 97, 96, 96 dB at different locations), so no single peak can dominate.
The clamp existed to contain the terminal-step blow-up, which
`terminal_x0=mean` now handles. Test: sample the same checkpoints with clamp
+/-8 and off (`experiments/dit_64/results/e0_clamp_test.json`).

## Clamp result (`experiments/dit_64/results/e0_clamp_test.json`, n=16, seed 1, DDIM 50)

Real frame peaks: median z 4.48 (above the +/-4 clamp); 0.17% of pixels |z|>4.

| arm | clamp 4: peak dB / prominence / on line | clamp off: peak dB / prominence / on line |
|---|---|---|
| factorized, mean | 66 / 3.8 / 0% | 66 / 3.8 / 0% |
| factorized, tile | 93 / 2.5 / 0% | 102 / 6.2 / 0% |
| temporal, mean | 71 / 4.2 / 0% | 71 / 4.2 / 0% |
| temporal, tile | 97 / 1.5 / 0% | 105 / 5.1 / 0% |

(clamp 8 identical to off.) **Partial:** the clamp capped peak height, and
removing it restores real peak height under tile decode, but prominence stays
5-6 dB (real 22) and no sequence has a target on a line. The core failure
remains: the model does not commit to one target moving coherently.
Mean decode is unaffected by the clamp because averaging already keeps peaks
below it.

## Next diagnostic (not run)

Locate the failing noise range: noise real E0 sequences to t in
{900, 700, 500, 300}, run DDIM from there, and score. The largest t that still
yields a single target on a line marks where synthesis breaks.

## Standard recipe result (`experiments/dit_64/results/e0_standard_scores.json`)

Checkpoint `checkpoints/e0_standard_bs32/last.pt`, step 62500, 32 sequences x seeds 1,2,3. Recipe: `experiments/dit_64/configs/e0_standard.yaml` (v-pred, schedule shift 4, no clamp, non-overlapping 8x8 patches, factorized attention, dim 384 x 12, EMA 0.9999, no smoothness).

| arm | mean | std | peak dB | prominence dB | on line | marginal L1 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| real | -0.01 | 1.04 | 105 | 22.0 | 100% | 0.064 (floor) | |
| raw_ddim50 | 0.07 | 0.90 | 105 | 22.6 | 93% | 0.111 | fails |
| raw_ddim250 | 0.05 | 0.94 | 105 | 22.7 | 93% | 0.084 | WORKS |
| ema_ddim50 | 0.04 | 0.94 | 106 | 23.1 | 95% | 0.079 | WORKS |
| ema_ddim250 | 0.02 | 1.00 | 106 | 23.2 | 95% | 0.057 | WORKS |

**Verdict (pre-set rule): the standard recipe WORKS on E0; earlier failures were setup issues.**

## Memorisation / diversity (`experiments/dit_64/results/e0_diversity.json`, `experiments/dit_64/e0_diversity.py`)

256 EMA samples (DDIM 50) vs 256 held-out val sequences, both compared with the
20k training set.

| | generated | held-out val |
|---|---:|---:|
| peak-path nearest-neighbour distance to train (median) | 0.341 | 0.338 |
| pixel nearest-neighbour distance to train (median) | 156.2 | 151.5 |
| spread of start range / range rate | 14.5 / 0.62 | 15.5 / 0.65 |
| spread of start Doppler / Doppler rate | 15.7 / 0.73 | 16.0 / 0.52 |

Generated sequences are as far from the training set as unseen real data
(pixel ratio 1.03), 254 of 256 peak paths are distinct, and the spread of
target positions and motions matches. **Not memorising.** Doppler-rate spread
is ~40% wider than real, a minor mismatch.
