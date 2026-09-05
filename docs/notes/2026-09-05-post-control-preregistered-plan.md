# Post-control diagnosis: preregistered experiment plan

**Date fixed:** 2026-09-05, before the experiments below were run.

## Starting evidence

The factorized-attention control in `samples/control_comparison.json` did not
meet its preregistered success threshold. At the matched 12,500-step budget,
normalized generated standard deviation was 0.6554 for the temporal baseline,
0.7043 for factorized depth 5, and 0.6999 for factorized depth 6, against 1.0043
for held-out data. Neither factorized arm exceeded 0.85. Consequently, absence
of spatial attention is not an established root cause and the diagnosis is
reopened.

The following tests are ordered from measurement and implementation checks to
new training. Phase 2 and Phase 3 remain paused.

## Shared protocol

- Dataset: the existing locked `data/cache` train/validation split.
- Training seed: 2026.
- Controlled training budget: 12,500 optimizer steps (10 epochs at batch 16),
  unless a run is explicitly labelled a smoke test.
- Sampling: 50-step DDIM, seed 1, with the same initial noise across comparable
  arms.
- Primary distribution measurements: normalized mean, standard deviation,
  p50/p75/p99/p99.9, and marginal histogram L1.
- Target measurements: corrected detector settings plus error inside padded
  ground-truth target regions versus background.
- Existing reference arm: `checkpoints/phase1_wandb/epoch_0010.pt`.

An arm is a convincing distributional recovery only if generated standard
deviation exceeds 0.85 without replacing the variance deficit with a large
mean shift or worse marginal L1. A change of at least 0.10 absolute standard
deviation versus the 0.655 baseline is considered a material lead, not proof.
Negative results from a newly introduced architecture are interpreted weakly
unless its optimization is independently shown to have converged.

## Experiment 1: timestep and region diagnosis

For fixed held-out examples and fixed noise, measure at selected diffusion
timesteps:

- epsilon MSE and reconstructed-x0 MSE/correlation;
- reconstructed mean, standard deviation, and upper-tail quantiles;
- epsilon and x0 error inside ground-truth target neighborhoods;
- the same errors on the complementary clutter/background mask.

This tests whether the global loss hides a target-region failure and identifies
where along the noise schedule the failure begins. Low correlation with the
particular clean example at near-terminal noise is expected and is not, by
itself, evidence of a broken diffusion model.

## Experiment 2: independent pipeline sanity check

Exercise `GaussianDiffusion` using analytically specified denoisers and simple
known distributions. The check must distinguish exact algebraic round trips
from finite-step DDIM discretization and record its expected and observed
moments. Failure blocks interpretation of all model ablations.

## Experiment 3: single-variable trained ablations

1. **No smoothness:** temporal depth-8 model with `lambda_smooth: 0`; tests
   whether the auxiliary temporal penalty suppresses target motion or contrast.
2. **No patch overlap:** temporal depth-8 model with patch 8, stride 8; tests
   whether averaging independently predicted overlapping patches attenuates
   variance. Parameter-count and throughput differences will be reported.
3. **EMA:** reproduce the temporal baseline training while maintaining an
   exponential moving average of weights; compare raw and EMA weights from the
   exact same optimization trajectory.
4. **Convolutional baseline:** a timestep-conditioned spatial convolutional
   denoiser trained with the same data, diffusion schedule, loss, and budget.
   A positive result isolates the failure to the patch-transformer family; a
   negative result does not prove the shared pipeline is at fault.

Each arm changes only the named factor where feasible. If resource limits force
a reduced budget, it will be labelled exploratory and compared with a baseline
run at that same reduced budget.

## Decision after the experiments

- Pipeline sanity failure: fix diffusion/sampling before further research runs.
- Target-only error concentration: investigate target-aware sampling or loss
  weighting before architecture expansion.
- No-smooth success: redesign or remove the physics regularizer.
- No-overlap success: redesign patch reconstruction/spatial coupling.
- EMA-only success: adopt EMA and rerun the architectural comparison.
- Convolutional success with transformer failures: focus the next study on the
  denoiser/tokenization architecture.
- No decisive arm: run replicated seeds and a longer conventional baseline,
  then revisit parameterization/loss weighting rather than Phase 2 or Phase 3.
