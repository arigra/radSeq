# Q1 ideas: implementation screen (2026-09-13)

These are synthetic-only experiments. The repository has no measured radar
sequences, so none establishes real-data fidelity or a publishable method.

## Rare detector errors

Implemented `archive/codex_research/research_tail_risk.py`, which derives a local-power CFAR
score from independent target-free clutter sequences. It calibrates thresholds
using a Gaussian score model, direct empirical quantiles, and a generalized
Pareto peaks-over-threshold (POT) fit. Training uses 6 sequences (150,528 cells)
and evaluation uses 12 independent sequences (301,056 cells) per clutter
setting. Results are in `archive/codex_research/results/research_tail_risk.json`.

At nominal Pfa 1e-3, the Gaussian model produced observed Pfa 0.0118–0.0191;
empirical quantiles produced 0.00059–0.00150; POT produced 0.00053–0.00144.
At nominal 1e-4, empirical quantiles produced 0.000033–0.000146 and POT
produced 0–0.000126. POT did not consistently improve on empirical quantiles.
The extreme-tail problem is real, but a simple EVT correction is not enough
to claim a new method. The 1e-4 estimates are noisy because test counts are
only about 0–44 exceedances, and neighboring cells are correlated.

## Exact target count and kinematics

Added `target_count` to `src/hard_latent.generate`. It accepts one count for
all sequences or one count per sequence, each in [1, 5], retaining the
previous empirical count sampling when omitted. A 5-sequence smoke/stress
check requested counts 1, 2, 3, 4, 5 and obtained exactly those counts.
Maximum range/velocity kinematic residual was 1.5e-5 in float32 coordinates;
all output maps were finite. This proves the latent-label and renderer
invariants, not that all targets are detectable under clutter or that the
architecture itself is novel. The property comes from the explicit renderer.

## Real-data residual model

Not trained. There are no measured radar sequences in the repository, and a
synthetic residual learned from the same simulator would test only recovery
of an injected artifact. The central claim—better real-data counterfactual
fidelity and transfer across environments—cannot be assessed here. A paired
or unpaired set of target-free and target-containing measured sequences is
needed before implementing a meaningful version of this experiment.

## Decision

None of the three is demonstrated as Q1-ready. The most promising question
remains rare *track-level* errors under distribution shift, but a publication
requires a model that beats empirical-tail and gain-invariant baselines on
independent measured data, with calibrated Pfa/Pd and tracking outcomes.
