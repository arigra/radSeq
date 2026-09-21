# DiT synthesis failure: terminal DDIM step (2026-09-14)

Written before running the sampler-only test below.

## Confound in earlier controls

Every architecture/objective control (U-Net, factorized attention, v-pred,
min-SNR) kept `lambda_smooth=0.1`, so they did not test architecture
independently of the smoothness loss.

## Evidence (`experiments/dit_64/diag_ddim_trace.py`, `experiments/dit_64/results/diag_ddim_trace.json`)

Cosine schedule: alpha_bar(999) = 2.4e-9, so the first DDIM step computes
x0 = (x - sqrt(1-ab) eps_hat) / sqrt(ab) with a 20,291x error gain.

| | epoch70 (lambda 0.1) | no_smooth (lambda 0) |
|---|---:|---:|
| step 0 raw x0 std / fraction clamped | 600 / 100% | 388 / 99% |
| step 0 clamped x0 corr. with final sample | 0.15 | 0.46 |
| x0 std at t=978, generated path | 0.35 | 1.13 |
| x0 std at t=978, noised real data | 0.48 | 0.68 |
| final std (real 0.986) | 0.667 | 1.286 |

Step 0 injects +/-4 white sign noise; scaled by sqrt(ab(978)) = 0.033 that is
+/-0.13, ~16x the energy of the true signal at that level. From step 1 the
generated path is off-manifold: variance too low with smoothing, too high
without. Both known failures start at the same step.

## Hypothesis

The zero-SNR first step, not the network, is the dominant cause. At SNR ~ 0
the posterior-mean x0 is the data mean (0 after normalisation).

## Test (no retraining)

Same checkpoints, seed, noise and 50-step grid; only change: x0 := 0 and
eps := x / sqrt(1-ab) at steps with alpha_bar < 1e-6 (only t=999).

## Decision rule

- **Confirmed** if epoch70 final std >= 0.85 with |mean| <= 0.2, and no_smooth
  final std moves toward 1 (within 0.85-1.15).
- **Partial** if either moves by >= 0.1 toward real but misses the bands.
- **Rejected** if both change by < 0.1 (run-to-run noise is ~0.06).

## Result (`experiments/dit_64/results/diag_ddim_trace_meanx0.json`)

| | before mean / std | after mean / std |
|---|---:|---:|
| epoch70 (lambda 0.1) | -0.33 / 0.667 | -0.11 / 0.717 |
| no_smooth (lambda 0) | +0.84 / 1.286 | +0.03 / 0.787 |

**Partial.** The terminal step caused the no_smooth overshoot and both mean
shifts. It does not cause under-dispersion: after the fix both models sit at
0.72-0.79, and generated-path x0 std trails the noised-real path at every t
(e.g. t=570: 0.60 vs 0.86 for epoch70). A second cause remains.

## Step count (terminal fix on; `samples/diag_ddim_trace_meanx0_steps*.json`)

| DDIM steps | epoch70 mean / std | no_smooth mean / std |
|---:|---:|---:|
| 50 | -0.11 / 0.717 | +0.03 / 0.787 |
| 200 | -0.21 / 0.739 | +0.45 / 1.020 |
| 1000 | +0.39 / 0.789 | +0.18 / 0.870 |

Single seed, n=16. Discretisation is not the main cause for epoch70: it stays
under-dispersed for every step count, and `experiments/dit_64/results/sampler_stochasticity.json`
already showed ancestral 1000-step sampling at 0.67-0.69. The smoothness-trained
model is biased regardless of sampler. The earlier finding that lambda=0
destroys targets (0.34 tracks/seq) was measured with the broken terminal step
and must be re-scored.

## Full metrics, sampler fix in `src/diffusion.py` (`terminal_x0="mean"`)

`samples/terminal_fix_seed{1,2,3}.json` (seed 3 agrees within 0.03 std; table
shows seeds 1/2), 32 sequences, 50 DDIM steps. Real:
std 1.004, marginal L1 real-vs-real 0.065, 3.09 target tracks/seq, persistence 0.144.

| arm (seed 1 / seed 2) | mean | std | marginal L1 | target tracks/seq | persistence |
|---|---:|---:|---:|---:|---:|
| epoch70 | -0.35 / -0.35 | 0.660 / 0.654 | 0.393 / 0.396 | 3.06 / 3.19 | 0.092 / 0.106 |
| epoch70 + fix | -0.14 / -0.13 | 0.714 / 0.701 | 0.334 / 0.358 | 2.62 / 2.69 | 0.052 / 0.060 |
| no_smooth | +0.76 / +0.80 | 1.285 / 1.280 | 0.533 / 0.562 | 0.34 / 0.31 | 0.000 / 0.000 |
| no_smooth + fix | +0.01 / +0.02 | 0.785 / 0.774 | 0.347 / 0.364 | 0.78 / 0.75 | 0.008 / 0.006 |

The fix repairs the mean and marginal L1 in both models. It does not restore
dispersion, and without smoothness the model still has no persistent targets
(about 50 tracks/seq, persistence ~0). So the lambda=0 target collapse is real,
not a sampler artefact.

## Next hypothesis (untested)

Training has the same pathology: `smooth_loss` is applied to clamped x0_hat
at every t, weighted by 1 - alpha_bar, which is largest where x0_hat is an
amplified eps error. It therefore flattens the network's high-noise
predictions over time. Every smoothness-schedule arm was scored through the
broken sampler, so they should be re-scored with the fix before any retraining.
