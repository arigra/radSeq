# Physics-loss research pilot (2026-09-13)

The question was whether radar-specific loss equations could support a Q1
journal contribution. This is a novelty screen and a pilot, not a publication
claim. We found no measured RD sequences in this repository; every quantitative
result below uses the same synthetic simulator family.

## 1. Acceleration-aware range–Doppler constraint plus target evidence

**Prior-art check before implementation.** Doppler-consistency losses already
appear in [SDDiff](https://www.ijcai.org/proceedings/2025/0979.pdf), and
geometry-aware conditioning with temporal regularization appears in
[synthetic FMCW radar-map diffusion](https://arxiv.org/abs/2601.06228).
Acceleration-aware range–Doppler signal processing is also established
([Neuberger et al.](https://arxiv.org/abs/2601.09317)). We did not find this
exact combined training loss for unconditional temporal RD generation in the
searched literature, but that is not evidence of patent-level or journal-level
novelty.

**Implemented.** `src/research_losses.py` estimates soft target centroids and
penalizes

`[(r[l+1]-r[l])/Tf - (v_D[l]+v_D[l+1])/2]^2`,

with range and Doppler converted into metres and m/s. A second term requires
the predicted target/background contrast to reach the contrast visible in the
paired reference frame. The second term matters: on hard synthetic scenes a
flat map scored 6.06 on the kinematic term, about the same as real maps (6.08),
because the GT-centred measurement windows themselves move. It scored 8.98 on
target evidence, compared with 0 for the real map. Reversed and range-shifted
sequences scored much worse on both terms.

**Training ablation.** Both arms resumed the same E0 pixel-DiT checkpoint
at step 12,500 for another 1,250 steps with `lambda_smooth=0.005`. The control
had no new terms; the candidate added `lambda_range_doppler=0.0005` and
`lambda_target_support=0.003`, active only for diffusion t <= 700. Scores use
32 generated sequences for each of seeds 1, 2 and 3, 50 DDIM steps, and the
same 64 held-out E0 reference sequences. Full raw results are in
`archive/codex_research/results/research_kinematic_ablation.json`.

| Metric (mean across seeds) | Control | Candidate | Real reference |
|---|---:|---:|---:|
| Normalized map std | 0.323 | 0.324 | 1.019 |
| Normalized mean | -1.377 | -1.210 | approximately 0 |
| Marginal L1 | 1.601 | 1.531 | real-vs-real 0.044 |
| Inter-frame-difference std | 0.389 | 0.390 | 1.066 |

**Decision.** The new loss detects deliberately damaged sequences and slightly
improves generated mean/L1, but it does not restore target-rich synthesis. This
is a negative result for this *fine-tuning pilot*, not a proof that the loss
could never work with another architecture or a full run from scratch. The
original high-noise, globally uncoordinated synthesis failure remains.

## 2. Detector-calibrated clutter objective

**Prior-art check before implementation.** Spatially and temporally correlated
K-clutter reconstruction is established
([spatiotemporal K-clutter reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC7219325/)); CFAR is a
standard radar detection method. We did not find an exact soft-CFAR
operating-curve objective for temporal radar generation in the targeted search,
but the ingredients are established and novelty depends on outperforming
strong detector-aware and gain-invariant baselines.

**Implemented.** `soft_cfar_map` in `src/research_losses.py` converts dB
amplitude to linear power before computing a differentiable CA-CFAR response.
The candidate loss matches occupancy and spatiotemporal detection moments at
three Pfa settings. `archive/codex_research/research_cfar_calibration.py` asks it to recover
known clutter correlation rho and texture shape nu on a 5x5 grid, from 16
fresh clutter sequences per setting. The fair baseline uses gain-invariant
skew, kurtosis and lag-one correlation, not raw mean alone.

| Test | Soft CFAR correct | Gain-invariant moments correct |
|---|---:|---:|
| Three parameter settings x three seeds | 8/9 | 9/9 |
| Same test with unknown receiver-gain offsets | 8/9 | 9/9 |

The raw mean/std baseline also scored 9/9 without gain shift, but failed under
gain shift. Raw results are in `archive/codex_research/results/research_cfar_calibration.json` and
`archive/codex_research/results/research_cfar_gain_shift.json`.

**Decision.** Detector calibration did not beat a competent simple baseline.
Do not claim this as a new method on the present evidence.

## What would change the assessment

A Q1-grade method claim needs an independent, preferably measured radar dataset,
a strong baseline set, repeated seeds, and improvement on detection/tracking or
another downstream task. Matching the simulator that created the training data
is insufficient. A plausible next experiment is to learn a *residual* radar
measurement model from measured sequences while retaining an explicit
acceleration-aware target renderer, then test whether it improves real-data
tracking over both the simulator and pixel/latent diffusion baselines. That
experiment needs measured sequences and has not been implemented or claimed
as novel here.
