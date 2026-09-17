# Trajectory-conditioned DiT (2026-09-17)

Written before the run. Research plan step 1: the DiT generates the targets it
is asked for, so synthetic sequences come with exact labels.
Design: `docs/superpowers/specs/2026-09-17-trajectory-conditioning-design.md`.

## Setup

`configs/cond_traj.yaml`: the passing full-data recipe (`e3_long`) plus
`model.cond_channels: 4` (3 class blob channels + presence), warm-started from
`checkpoints/e3_long_bs32/last.pt` (EMA weights; new input weights zero, so
training starts from exactly the unconditional model), condition dropout 0.1,
40 epochs (25,000 steps).

## Decision rule (Ari: trajectories as the control, pass bar relative to the simulator)

On requests taken from held-out validation labels, EMA weights, DDIM 30:

- hit rate (requested target-frames with a detected peak within 2 bins)
  >= simulator hit rate on the same labels - 0.05;
- unrequested lasting tracks per sequence (tracks of >= 8 frames not near a
  requested target on most frames) <= the simulator's;
- Ari's four checks pass (std within 0.1, marginal L1 <= 0.15, target
  tracks/seq within 15%, persistence within 25% of real);
- null check: the unconditional e3_long model on the same requests and seeds
  hits at least 0.20 less, else the test is too easy and no result is reported.

Guidance w in {1, 2, 3} is chosen on val 0-95 / seeds 1-3; the verdict uses
val 96-287 / seeds 4-9 (192 sequences).

Known before the run: the simulator hits 0.96 of its own labelled
target-frames, so the bar is about 0.91. It has ~0.4 unrequested lasting
tracks/seq (8-sequence estimate), so with 192 sequences that comparison has a
sampling error of roughly +/-0.05; a miss by less than that should be read as
a tie, not a failure of the method.

Not measured: whether generated targets look like their requested class.

## If the session closes

Rerun `setsid nohup bash scripts/run_cond_traj.sh > logs/cond_traj_pipeline.console.log 2>&1 < /dev/null &`
in a new session, or `sbatch scripts/cond_traj.sbatch` from `ece-hpc`. Both
resume from `checkpoints/cond_traj_bs32/last.pt`.

## Result (`samples/cond_traj_scores.json`)

Checkpoint `checkpoints/cond_traj_bs32/last.pt`. Guidance w = 1.0, chosen on val 0-95 / seeds 1-3 (w=1.0: hit 0.964, passes True, w=2.0: hit 0.979, passes False, w=3.0: hit 0.977, passes False). Verdict on val 96-287, seeds 4-9, 192 sequences.

| check | simulator / real | conditional DiT | |
|---|---:|---:|---|
| hit rate | 0.976 | 0.973 | ok |
| unrequested lasting tracks / seq | 0.33 | 0.29 | ok |
| std | 1.004 | 0.955 | ok |
| marginal L1 | 0.065 | 0.086 | ok |
| target tracks / seq | 3.09 | 3.03 | ok |
| persistence | 0.144 | 0.139 | ok |

Null check (unconditional e3_long, same requests and seeds): hit rate 0.013 -> ok (conditional hit rate is at least 0.20 higher)

**Verdict (pre-set rule): the conditional DiT follows requested trajectories.**
