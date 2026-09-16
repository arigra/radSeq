# Standard DiT recipe on the full data (2026-09-14)

Written before the run. The same recipe passed E0
(`2026-09-14-e0-dit-standard-setup.md`: EMA + DDIM 250 gave std 1.00 vs 1.04,
prominence 23 vs 22 dB, 95% of sequences on a line). Now the full, locked
data (`data/cache`: 1-5 targets, all classes, clutter, noise).

## Setup

`configs/e3_standard.yaml`: identical to `configs/e0_standard.yaml` except the
data. v-prediction, schedule shift 4, no x0 clamp, non-overlapping 8x8
patches, factorized attention, dim 384 x 12, EMA 0.9999, bf16, no smoothness
loss, batch 32, 100 epochs (62,500 steps, ~4.7 h). Run unattended by
`scripts/run_e3_standard.sh`; scored by `scripts/score_hard_standard.py`
(raw and EMA weights, DDIM 50 and 250, 3 seeds x 32 sequences).

The intermediate regimes (E1 noise only, E2 noise + clutter) are skipped;
they are the fallback to locate a failure if this run does not pass.

## Decision rule (Ari, 2026-09-14)

Against held-out real data, using `src/eval/metrics.evaluate_sequences` as in
every earlier arm (reference: std 1.004, marginal L1 real-vs-real 0.065,
3.09 target tracks/seq, persistence 0.144):

| check | passes if |
|---|---|
| std (contrast) | within 0.1 of real |
| marginal L1 (brightness mix) | <= 0.15 |
| target tracks/seq (right number of lasting targets) | within 15% of real |
| persistence (how long detections last) | within 25% of real |

An arm passes only if all four pass.

## Result (`samples/e3_standard_scores.json`)

Checkpoint `checkpoints/e3_standard_bs32/last.pt`, step 62625, 32 sequences x seeds 1,2,3.

| arm | std | marginal L1 | target tracks/seq | persistence | verdict |
|---|---:|---:|---:|---:|---|
| real | 1.004 | 0.065 (floor) | 3.09 | 0.144 | |
| raw_ddim50 | 0.968 ok | 0.060 ok | 3.03 ok | 0.088 FAIL | fails |
| raw_ddim250 | 0.995 ok | 0.034 ok | 2.76 ok | 0.079 FAIL | fails |
| ema_ddim50 | 0.970 ok | 0.047 ok | 2.88 ok | 0.096 FAIL | fails |
| ema_ddim250 | 0.998 ok | 0.025 ok | 2.68 ok | 0.087 FAIL | fails |

**Verdict (pre-set rule): the DiT does NOT yet pass on the full data; see which checks fail.**

## Reading (added after the result)

Three of four checks pass on every arm; only persistence fails (0.08-0.10 vs
0.144, pass needs >= 0.108). Persistence is long tracks (>= 13 of 16 frames)
divided by all tracks, so it splits into two numbers from the same JSON
(persistence x mean_tracks_per_seq):

| | long tracks/seq | all tracks/seq | velocity consistency (lower = smoother) |
|---|---:|---:|---:|
| real | 3.40 | 23.7 | 1.36 |
| raw DDIM 50 | 2.30 | 26.2 | 2.01 |
| EMA DDIM 50 | 2.70 | 28.1 | 1.79 |
| EMA DDIM 250 | 2.66 | 30.6 | 1.96 |

Both halves contribute: about 0.7-1.1 fewer detections survive 13+ frames,
and 3-7 more short-lived detections appear per sequence. Motion is also
jerkier than real. Frame-level appearance is right (std 0.97-1.00, marginal L1
0.025-0.06, at or below the 0.065 real-vs-real floor); what is off is
frame-to-frame continuity of weaker returns.

## Attempt 1: noise floor, track anatomy, sampler steps, noise-level localisation (2026-09-15)

**Noise floor** (`samples/diag_persistence.json`): persistence of eight disjoint
32-sequence subsets is 0.147 +/- 0.020 for real data and 0.101 +/- 0.012 for
EMA DDIM 50 samples. The gap (0.046) is real, about 2.3 real SDs.

**Anatomy** (same file, 256 sequences each):

| track length | real /seq (target-matched) | generated /seq |
|---|---:|---:|
| 1-2 frames | 19.0 (3.1) | 21.0 |
| 3-7 | 3.1 (0.8) | 3.0 |
| 8-12 | 0.57 (0.36) | 0.70 |
| 13-16 | 2.48 (2.41) | 2.28 |

Long tracks are almost all true targets. Generated sequences break a few
targets (long -> medium) and add ~2 single-frame blips. Brightness and Doppler
per bucket match.

**Sampler steps** (`samples/e3_step_sweep.json`, 3 seeds x 32):

| EMA | std | marginal L1 | target tracks | persistence | all four |
|---|---:|---:|---:|---:|---|
| DDIM 10 | 0.850 | 0.199 | 3.59 | 0.157 | fails (std, L1) |
| DDIM 20 | 0.921 | 0.104 | 3.34 | 0.135 | passes |
| DDIM 30 | 0.948 | 0.071 | 3.07 | 0.113 | passes (persistence marginal) |
| DDIM 50 | 0.970 | 0.047 | 2.88 | 0.096 | fails (persistence) |
| DDIM 250 | 0.998 | 0.025 | 2.68 | 0.087 | fails (persistence) |

Fewer steps trade frame-level fidelity for fewer blips. Raw weights pass
persistence only at DDIM 10.

**Localisation** (`samples/diag_persistence_by_t.json`): noising real
sequences to t and denoising back keeps persistence at the real level up to
t=500 (0.145-0.149), then 0.130 at t=700 and 0.107-0.112 at t>=900. The
continuity loss happens in the high-noise stage, where the global temporal
layout is decided, not in low-noise refinement.

Caveat: DDIM 20/30 were chosen after seeing seeds 1-3, so they are confirmed
on fresh seeds 4-9 before being called a fix.

## Attempt 2: confirmation on fresh seeds 4-9 (`samples/e3_confirm_seeds4to9.json`)

6 seeds x 32 sequences, same real reference as the rule.

| arm | std | marginal L1 | target tracks | persistence (per-seed range) | all four |
|---|---:|---:|---:|---:|---|
| EMA DDIM 20 | 0.920 | 0.105 | 3.49 | 0.135 (0.120-0.152) | **passes** |
| EMA DDIM 30 | 0.947 | 0.074 | 3.27 | 0.121 (0.105-0.143) | **passes** |
| raw DDIM 20 | 0.918 | 0.121 | 3.28 | 0.109 (0.102-0.122) | passes (persistence at the limit) |
| raw DDIM 30 | 0.944 | 0.084 | 3.15 | 0.104 (0.093-0.121) | fails |

**Verdict (pre-set rule, held-out seeds): the DiT passes on the full data with
EMA weights and 20-30 DDIM steps.** EMA DDIM 30 is the balanced setting (every
check inside its band with margin; target count 3.27 vs 3.09). EMA DDIM 20 has
more persistence margin but its target count (3.49) is near the upper limit
(3.56) and contrast is lower (0.92).

This is a sampler setting, not a change to the model. The model itself still
loses continuity in the high-noise stage (t >= 700). A training-side fix for
that stage would widen the margins, but the rule is met without it.
