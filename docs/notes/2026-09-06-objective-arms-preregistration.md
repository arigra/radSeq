# Objective arms — pre-registration

**Fixed 2026-09-06, before any of these runs started.** No result existed when
this file was committed.

## Why these arms

Everything tried so far sits *around* the denoiser: attention topology, patch
reconstruction, the x0 clamp, the auxiliary smoothness loss. The objective
itself has never changed. The failure is localised at t >= 600, where
epsilon-prediction carries almost no gradient about where structure belongs —
which is exactly the decision the model fails to make. These are the two
standard remedies aimed at that regime.

- **vpred**: predict `v = sqrt(abar)*eps - sqrt(1-abar)*x0` instead of epsilon.
- **minsnr**: keep epsilon-prediction, weight the loss by
  `min(SNR_t, gamma)/SNR_t` with gamma = 5, so low-noise steps stop dominating.

Each changes exactly one factor against the Phase-1 baseline. Everything else —
data split, model, lambda_smooth = 0.1, the (1 - abar) smoothness schedule, the
+/-4 sigma clamp, 12,500 steps, batch 16, lr 1e-4 — is unchanged.

## Protocol

- Seeds 2026, 2027, 2028 per arm; six runs total.
- Sampling: 50-step DDIM, n = 32, sampling seed 1, mean patch decode.
- Reported per arm: the mean over the three training seeds, and the spread.
- Baseline for comparison: `checkpoints/phase1_wandb/epoch_0010.pt` at the same
  budget — generated std 0.655, target tracks/seq 1.75 (seed-averaged over
  sampling seeds 1-4), marginal L1 0.380.

## Measured noise floors, established before this experiment

- generated std: **~0.06** at fixed seed (from an accidental exact replicate).
- target tracks/seq: **~0.22** across sampling seed at fixed checkpoint.

Differences smaller than these are not results.

## Decision rule

An arm counts as **a fix** only if, averaged over its three seeds:

1. generated std > **0.85**, and
2. target tracks/seq >= **1.75**.

Both conditions are required. Every previous arm that raised variance did so by
destroying targets (no_smooth: std 1.285 with 0.34 tracks; inverted smoothness:
std 1.281 with 0.72 tracks), so variance alone is explicitly not sufficient.

Two known weaknesses of these thresholds, recorded now rather than argued later:

- 0.85 is inherited from the original attention control and was never justified
  against anything; real data is 1.004.
- 1.75 is lenient. The hann decode already reaches 2.34 without touching the
  objective, so an arm that clears 1.75 but not 2.34 is worse than a change we
  already have.

## Outcomes

- **Either arm clears both bars**: the objective was the problem. Adopt it,
  re-run the decode comparison on top of it, and resume the phase plan.
- **An arm clears std but not tracks**: same failure mode as every previous
  variance gain. Not a fix; report as such.
- **Neither clears**: architecture, decode, clamp, auxiliary loss and objective
  have all been eliminated. Treat unconditional synthesis as unreachable in this
  setup and reframe the project around restoration, where the same weights reach
  x0hat std 0.986 against real 1.003.
