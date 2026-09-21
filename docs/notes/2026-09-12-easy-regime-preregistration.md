# The easy-regime arm — pre-registration

**Fixed 2026-09-12, before the dataset was generated or anything trained.**

## Question

The intensity deficit is the project's one open problem: generated std 0.655 against real
1.004, marginal L1 0.398 against a 0.065 real-vs-real floor. It has survived 7x training
budget, spatial attention, a convolutional denoiser, EMA, every clamp width, every smoothness
weighting and magnitude, v-prediction and min-SNR
(`experiments/dit_64/results/budget_progression.json`, `archive/ablations_aug/results/ablation_single_variable.json`,
`archive/ablations_aug/results/control_comparison.json`, `experiments/dit_64/results/fix_smoothness_schedule.json`,
`experiments/dit_64/results/fix_clamp_width.json`, `experiments/dit_64/results/objective_arms_partial.json`,
`experiments/dit_64/results/objective_minsnr.json`).

Every one of those arms varied the **model or the objective**. None varied the **data**. So the
question here is:

> Is the deficit a function of data complexity, or is it intrinsic to the MSE-optimal denoiser
> and the smoothness/decode stack around it?

The motivating observation is the restoration/synthesis split: handed a noisy *real* sequence
the network returns std 0.986, essentially perfect; handed only noise it returns 0.655
(`experiments/dit_64/results/diag_tail_origin.json`). Something about producing the distribution from scratch
fails in a way that reproducing a given sample does not. If the training distribution were
simple enough, that failure might not appear.

## The regime ladder

Measured on 24 sequences per regime before fixing this note. E3 reproduces the locked cache
statistics (41.479 dB / 11.065 dB in `data/cache/manifest.yaml`) to three decimal places,
which is the check that the regime harness is faithful.

| regime | targets | gain | clutter | noise | dB mean | dB std | norm p99 | norm p99.9 | excess kurtosis |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **E0** | 1 steady | 20 dB fixed | off | off | 34.95 | 15.73 | 2.53 | 3.54 | 1.97 |
| E1 | 1 steady | 20 dB fixed | off | on | 39.28 | 11.24 | 3.16 | 4.57 | 2.38 |
| E2 | 1 steady | 20 dB fixed | on | on | 43.66 | 11.49 | 2.71 | 4.05 | 0.82 |
| E3 | 1-5, all classes | U(-5, 10) dB | on | on | 41.47 | 11.08 | 2.51 | 3.94 | 0.39 |

No regime has a pathological dB floor, so E0 is a usable training target. Note E0 is *more*
heavy-tailed than E3 (excess kurtosis 1.97 vs 0.39) — simplifying the physics does not
simplify the intensity distribution, which is worth keeping in mind when reading the result.

## The arm

**E0 only.** One arm, single variable: the data.

| | value |
|---|---|
| data | E0, 20,000 train / 2,000 val, own train-split normalisation |
| E0 definition | 1 target, class 0 (steady point), fixed 20 dB gain, no clutter, no white noise |
| model | identical — dim 256, depth 8, heads 8, 9.83 M params, temporal attention |
| budget | 12,500 steps (10 epochs at 20k / batch 16) |
| optimiser | AdamW, lr 1e-4, weight decay 0, grad clip 1.0 |
| loss | `L_DiT + 0.1 * L_smooth`, smoothness weight `(1 - abar)` |
| sampling | DDIM 50 steps, mean decode, x0 clamp +/-4, no moment calibration |
| seed | 2026 (training), 3 sampling seeds for scoring |
| scoring | n = 32, corrected detector (`max_peaks=5`, `min_track_len=8`) |

The 12,500-step budget is known to be confounded for *target* metrics
(`experiments/dit_64/results/budget_progression.json`: tracks/seq went 1.59 → 3.06 between 12,500 and 87,500 steps) but
**budget-independent for distribution metrics**, which are what this question is about
(std 0.652 → 0.655 and marginal L1 0.388 → 0.398 across the same 7x). Target metrics from this
arm are therefore reported but not used in the decision.

E0's own real-vs-real marginal L1 floor must be measured on E0 data. It is **not** the 0.065 of
E3 and may not be comparable to it.

## Decision rule

Read on generated std against E0's own real reference (≈1.00 after its own train-split
normalisation), at n = 32 across three sampling seeds. The measured run-to-run noise floor on
generated std is 0.06 (`abl_ema` accidental replicate).

- **std within 0.06 of its real reference** → the deficit is **data-complexity-driven**. The
  dB-scaling and fixed-normalisation suspects are wrong, and the next arms are E1 and E2 to
  isolate which simplification was responsible.
- **std below 0.75** (still ≈0.65x real) → the deficit is **intrinsic** to the denoiser and the
  stack around it. Data complexity is eliminated as a cause, and the representation suspects
  stay open.
- **std in 0.75-0.94** → **inconclusive**. Report it as inconclusive. Do not narrate it as
  partial support for either side.

Marginal L1 against the E0 floor and target tracks/seq are reported alongside std in every
case. E0 contains exactly one target per sequence, so ≈1.0 track/sequence is the sanity check
that the detector is working on this regime at all; a value far from 1.0 invalidates the arm's
target numbers but not its distribution numbers.

## Declared confound

E0 differs from E3 in **four ways at once**: one target instead of 1-5, class fixed to steady,
target gain fixed at 20 dB instead of U(-5, 10) dB, and both clutter and white noise removed.

- A **null** result (std still ≈0.65) is clean and fully informative: no amount of
  simplification along any of those four axes helped.
- A **positive** result does **not** isolate which simplification caused it. That is precisely
  why E1 and E2 are named here as the follow-ups rather than run now — E1 adds noise back, E2
  adds clutter back, and the pair separates the axes.

This arm is not a fishing expedition for a better number. It is a test of one hypothesis, with
the outcome that most advances the project being the null.

## Scope

One arm. No E1/E2 without approval. Result lands in `archive/codex_research/results/easy_regime_arm.json` in the same
shape as the other arm files, and is reported in
`notebooks/02_radseq_self_contained.ipynb` §17 against this rule, read verbatim.

---

## Addendum, added during the run (decision rule unchanged)

**Discovered while the arm was training, before any result was scored.** Recorded here
rather than folded into the text above, so that what was pre-registered stays visible as
pre-registered.

Comparing the two runs' held-out losses at matched epochs:

| epoch 5 | dit | smooth | 0.1 x smooth | total | smoothness share |
|---|---:|---:|---:|---:|---:|
| E0 arm | 0.0234 | 0.1875 | 0.0187 | 0.0421 | **44.5%** |
| locked data | 0.2920 | 0.0734 | 0.0073 | 0.2993 | **2.4%** |

Two things follow.

**Good for the arm:** E0's denoising loss is about 12x lower, so the model fits E0's
conditional means far better than it fits the locked data's. Whatever the arm shows cannot be
attributed to the model failing to learn the easy regime.

**A confound that was not foreseen:** because `L_DiT` collapsed while `L_smooth` did not, the
same `lambda_smooth = 0.1` gives the smoothness prior roughly **18x more influence** on the
E0 objective than on the locked one. `experiments/dit_64/results/fix_smoothness_schedule.json` identifies that
penalty as the largest single lever on generated std in the project. So "data is the only
variable" holds for the *configuration* but not for the *optimisation*: the pressure that
most suppresses dispersion increased in the arm.

How this changes the reading of each outcome:

- **INTRINSIC (std stays low):** weaker than pre-registered. Data complexity is not cleanly
  eliminated, because a competing std-suppressing effect strengthened simultaneously.
- **DATA-COMPLEXITY-DRIVEN (std reaches its reference):** stronger than pre-registered. The
  deficit closed *despite* more smoothness pressure.
- **INCONCLUSIVE:** unchanged.

**The decision rule is not modified.** The verdict is read from the bands fixed above. The
remedy is an additional arm, named here and not run: repeat E0 with `lambda_smooth` scaled so
the auxiliary term holds the same share of the objective (approximately 0.005), or with
`lambda_smooth = 0`. That isolates the data change from the weighting change.

This is the second time in this project that an auxiliary loss has had a large effect nobody
intended, and the first time a *data* change has silently reweighted an objective. Worth
checking the loss-component shares whenever a data regime changes.
