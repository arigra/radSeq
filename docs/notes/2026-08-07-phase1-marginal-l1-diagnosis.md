---
title: "Phase 1 diagnosis — marginal_l1 gap and metric validity"
date: 2026-08-07
tags:
  - radar
  - diffusion
  - diagnosis
---

# Phase 1 diagnosis: why `marginal_l1` is 0.44 against a 0.05 floor

Reproduce with `python -m scripts.diag_marginal_l1`, `diag_tail_origin`,
`diag_track_validity`. Raw reports land in `samples/diag_*.json`.

## Summary

Two independent problems, one of them invalidating the evidence for the other's
"good" counterpart:

1. **The intensity gap is real and architectural.** The generated bright tail is
   collapsed because the DiT has *no spatial information flow whatsoever*. This
   is a property of the proposal's design, not a training or sampling bug.
2. **The physics metrics are ~80% clutter false alarms.** They do not support the
   claim that Phase 1 learned target motion, and they cannot gate Phase 2 as the
   plan currently specifies.

## 1. The intensity gap

### Ruled out

| Hypothesis | Evidence | Verdict |
|---|---|---|
| DDIM `x0.clamp(-4, 4)` truncates the range | 2.6e-5 of generated mass sits at the boundary; only 0.09% of real data exceeds ±4σ | rejected |
| `marginal_l1`'s shared min/max range inflated by outliers | shared-range L1 0.4532 vs percentile-robust L1 0.4538 | rejected |
| The 0.065 floor was measured under a different protocol | remeasured at the actual protocol (16 gen vs 64 real, 8 disjoint draws): **0.052 mean**, range 0.011–0.191 | rejected — floor is sound |
| Too few DDIM steps | std 0.655 / 0.650 / 0.669 at 50 / 100 / 250 steps; ancestral DDPM 0.721 | rejected |
| Undertrained | std flat across epochs 10→70: 0.659, 0.654, 0.662, 0.667. `marginal_l1` flat 0.38–0.41 | rejected — converged by epoch 10 |

### What is actually happening

Generated data has **0.63× the real standard deviation**, with the bright tail gone:

| percentile (σ) | real | generated |
|---|---:|---:|
| 1 | −2.05 | −2.00 |
| 50 | −0.15 | −0.39 |
| 75 | +0.69 | −0.02 |
| 99 | +2.50 | +1.24 |
| 99.9 | +3.92 | +1.84 |
| max | +5.69 | +2.74 |

The noise floor is reproduced well; bright content — where targets live — is not.

### Root cause

The model denoises *real* noised data essentially perfectly:

| t | 10 | 100 | 400 | 800 |
|---|---:|---:|---:|---:|
| `pred_x0` std (real = 1.003) | 0.986 | 0.975 | 0.907 | 0.777 |
| `pred_x0` p99.9 (real = 3.92) | 3.80 | 3.79 | 3.76 | 3.12 |

So the network can represent and restore the full dynamic range **when the spatial
structure is already present in its input**. It cannot create that structure from
noise.

`src/dit.py:40-42` reshapes tokens to `(B*P, L, d)` before attention. Each of the
225 spatial patch sites therefore runs an **independent** temporal transformer.
There is no information flow between spatial locations at any depth — the only
coupling is passive patch overlap (p=8, s=4) and a static `spatial_pos` embedding.

Generating unconditionally, each site must decide its own brightness from its own
white-noise input, which carries no information about where targets belong. The
MSE-optimal choice per site is close to the unconditional mean, so no site commits
to being bright. The distribution collapses toward the mode, which drops both the
mean and the variance.

The predicted signature is present in the data:

| | bright pixel fraction | blobs/frame | mean blob size (px) |
|---|---:|---:|---:|
| real | 0.182 | 46.2 | 17.4 |
| generated | 0.054 | 47.9 | 4.8 |

Same *number* of bright blobs, 3.6× smaller. The model scatters small specks
instead of forming the coherent extended structures real RD maps contain.

This is a property of the architecture the proposal specifies, and it is
converged — more compute will not move it.

## 2. The physics metrics measure clutter, not targets

Over 32 real val sequences with ground-truth trajectories (tolerance 2 bins):

| quantity | value |
|---|---:|
| true targets per sequence | 3.41 |
| **tracks reported per sequence** | **45.38** |
| detected peaks matching a target | 46.3% |
| **tracks starting on a target** | **19.9%** |

`detect_peaks` thresholds at `median + 12 dB`, a *relative* threshold, so clutter
fluctuations are detected freely. `evaluate_sequences` then averages
`velocity_consistency`, `doppler_drift`, and `persistence` over all tracks, so
roughly four fifths of every reported number describes clutter.

**This inverts the Phase 1 read.** Generated sequences scored *better* than the
real reference on `velocity_consistency` (2.67 vs 3.04) and `doppler_drift`
(0.437 vs 0.706). That is not better target motion — it is the same
over-smoothing that causes the intensity gap: blurrier, smaller clutter blobs
drift less between frames, which these metrics reward.

The status note's claim that the backbone "learned temporally coherent target
motion" is not supported by this evidence.

## Consequences for the plan

- **Phase 2's exit criteria are currently unmeasurable.** The plan gates on
  `velocity_consistency` and `doppler_drift` improving; at 20% track precision
  those numbers are dominated by clutter. Fix evaluation before running Phase 2.
- **Phase 3 conditioning will not fix the spatial problem.** `ConditionEncoder`
  produces one global `(B, dim)` vector added to the timestep embedding — it
  carries no spatial layout, so it cannot tell any site to be a target.
- The proposal's temporal-only attention is *sufficient for denoising and
  temporal coherence* and *insufficient for unconditional spatial synthesis*.
  That is a real, defensible finding either way it gets resolved.

## Open question (needs a control run)

The diagnosis — no spatial information flow — is established by code inspection
and matches every measurement. The *fix* is still a hypothesis: adding factorized
spatial attention should restore the tail, but that has not been tested. A short
run of a spatial+temporal variant on a data subset, compared on generated std and
`marginal_l1`, would settle it.
