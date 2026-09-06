# Phases 2 and 3 — pre-registration

**Fixed 2026-09-07, before either run started.**

## Reference

Phase 1 at full budget, `checkpoints/phase1_wandb/epoch_0070.pt` (87,500 steps),
three sampling seeds, mean decode, no moment calibration:

| metric | real | Phase 1 @ 87.5k |
|---|---:|---:|
| velocity inconsistency (lower better) | 1.358 | 1.485 |
| Doppler drift (lower better) | 0.508 | **0.332** |
| persistence (higher better) | 0.144 | 0.092 |
| target tracks / seq | 3.094 | 3.062 |
| marginal L1 (lower better) | 0.065 | 0.393 |

Noise floors measured earlier and applying here: **0.06** on generated std,
**0.22** on target tracks/seq. For velocity inconsistency the seed spread on the
reference was 0.295, so **0.30** is the smallest difference worth reading.

## Budget

Both phases train **70 epochs / 87,500 steps**, matching the reference. The
12,500-step budget used for all earlier ablations is now known to be confounded
for target metrics — Phase 1 went from 1.59 to 3.06 tracks/seq between 12,500 and
87,500 steps — so a short-budget phase comparison would measure nothing.

Phase 3 is trained from scratch, as the design spec requires, because it
introduces the conditioning encoder (+0.40 M parameters).

## Phase 2 — physics losses

Adds the soft-argmax trajectory and Doppler consistency losses at
lambda_traj = lambda_doppler = 0.01, everything else identical to Phase 1.

**Spec exit criterion:** velocity-consistency and Doppler-drift improve over
Phase 1 without degrading marginal statistics.

**Made concrete here:** velocity inconsistency below **1.19** (a 0.30 improvement
on 1.485, i.e. larger than the seed spread), Doppler drift not worse than 0.332,
and marginal L1 not worse than 0.393.

**Note the criterion is now weaker than when it was written.** Phase 1 at full
budget already reaches velocity 1.485 against 1.358 real, and its Doppler drift
of 0.332 is already *better* than real. There is little room left for the physics
losses to claim. A null result here is the likely outcome and would mean the
auxiliary losses are unnecessary, not that they are broken.

**Additional caution, from experience in this project:** the one auxiliary loss
already in use (temporal smoothness) turned out to be the largest single lever on
the intensity distribution, with effects nobody intended. Phase 2's two new
auxiliary terms will be checked against the *distribution* metrics as well, not
only the kinematic ones they target.

## Phase 3 — conditioning

Motion / class / environment conditioning with classifier-free guidance,
cond_dropout 0.1.

**Spec exit criterion:** strong commanded-vs-measured velocity correlation; SCR
adherence across a sweep.

**Made concrete here**, using `velocity_adherence` (Pearson correlation between
commanded v0 and the velocity measured from generated tracks):

- **r >= 0.5** counts as the conditioning working at all.
- **r >= 0.7** counts as strong.
- The unconditional Phase-1 model must be measured on the same protocol as a
  null. If it scores materially above 0, the metric is broken and no Phase-3
  number may be reported until that is resolved.
- Guidance scale will be swept (1.0, 2.0, 4.0). Guidance is known to trade
  diversity for adherence, so any adherence gain will be reported alongside its
  effect on target tracks/seq and marginal L1, not on its own.

Kinematic and distribution metrics are reported for Phase 3 too. Conditioning is
a single global vector added to the timestep embedding and carries no spatial
layout, so it is not expected to change target placement; if it does, that needs
explaining rather than celebrating.

## Interaction with moment calibration

The in-loop moment calibration is a sampling-time procedure and applies to any
checkpoint. Both phases will be evaluated **without** it for comparison against
the reference above, and the better phase additionally **with** it, to check the
calibration and the conditioning do not interfere.

## Order and stopping

Phase 3 runs first, then Phase 2, ~6.2 hours each. Phase 3 is first because it is
the proposal's actual proposition — controllable radar data — and because Phase
2's rationale is largely obsolete now that motion converges with budget alone.
Neither depends on the other: Phase 3 trains from scratch regardless.

When both have run and been evaluated, this queue ends. No further arms without
approval.
