# Full data: longer training of the passing model (2026-09-15)

Written before the run.

## Why

The factorized full-data model (`experiments/dit_64/configs/e3_standard.yaml`, 62,625 steps)
passes Ari's rule only with EMA weights and DDIM 20-30, with a thin
persistence margin (0.121 vs limit 0.108). Continuity is lost in the
high-noise stage (t >= 700). Joint attention made it worse
(`2026-09-15-e3-full-attention.md`: 0.080). Larger factorized models
(dim 512-640) run out of memory at batch 32 on the shared 3090 and would be
~2x slower with smaller batches, too slow for one session. The standard
remedy that fits: train the same model longer. In diffusion, sample quality
keeps improving after validation loss flattens, and EMA averages over more
of that late training.

## Setup

`experiments/dit_64/configs/e3_long.yaml`: identical to `e3_standard` except epochs 180. Starts
from a copy of `checkpoints/e3_standard_bs32/last.pt` (step 62,625, epoch 100; model,
EMA and optimizer state) in `checkpoints/e3_long_bs32/`, adding 80 epochs
(50,000 steps, ~3.5 h). The original checkpoint is not modified (sha256
prefix 8f1b9a3f7e6ed0b9; the copy was verified weight-identical).
Scored by `experiments/dit_64/score_hard_standard.py`: raw and EMA, DDIM 30 and 50,
seeds 1-6 x 32 sequences.

## Decision rule (Ari's rule, unchanged)

Passes if std within 0.1 of real, marginal L1 <= 0.15, target tracks/seq
within 15% of real, persistence within 25% of real.

**Replaces `e3_standard`** if EMA DDIM 30 passes all four with persistence
>= 0.121, or EMA DDIM 50 passes all four. Otherwise `e3_standard` with
EMA DDIM 30 stays.

## If the session closes

At most ~1000 steps (~4 min) are lost. Continue with either
- a new session: `setsid nohup bash experiments/dit_64/run_e3_long.sh > experiments/dit_64/logs/e3_long_pipeline.console.log 2>&1 < /dev/null &`
- the login host `ece-hpc`: `sbatch experiments/dit_64/e3_long.sbatch`

Both resume from `checkpoints/e3_long_bs32/last.pt` and refuse to start if a
training process is already running. Progress: `experiments/dit_64/logs/e3_long_pipeline.log`
(ends in DONE or FAILED) and `experiments/dit_64/logs/e3_long_bs32.log`.

## Result (`experiments/dit_64/results/e3_long_scores.json`)

Checkpoint `checkpoints/e3_long_bs32/last.pt`, step 112625, 32 sequences x seeds 1,2,3,4,5,6.

| arm | std | marginal L1 | target tracks/seq | persistence | verdict |
|---|---:|---:|---:|---:|---|
| real | 1.004 | 0.065 (floor) | 3.09 | 0.144 | |
| raw_ddim30 | 0.943 ok | 0.119 ok | 3.32 ok | 0.130 ok | PASSES |
| raw_ddim50 | 0.964 ok | 0.107 ok | 3.18 ok | 0.116 ok | PASSES |
| ema_ddim30 | 0.940 ok | 0.084 ok | 3.25 ok | 0.124 ok | PASSES |
| ema_ddim50 | 0.961 ok | 0.061 ok | 3.14 ok | 0.115 ok | PASSES |

**Verdict (pre-set rule): the DiT WORKS on the full data.**

## Reading (added after the result)

**Replaces `e3_standard`.** Both replacement conditions hold: EMA DDIM 30
passes with persistence 0.124 (>= 0.121), and EMA DDIM 50 passes (it failed
before). Raw weights now pass too, so the result no longer depends on EMA or
on the step count. Validation loss improved from 0.382 to 0.378.

Per-seed persistence (limit 0.108):

| arm | range over 6 seeds | seeds below the limit |
|---|---:|---:|
| raw_ddim30 | 0.107-0.149 | 1 of 6 |
| raw_ddim50 | 0.102-0.130 | 3 of 6 |
| ema_ddim30 | 0.111-0.145 | 0 of 6 |
| ema_ddim50 | 0.099-0.129 | 2 of 6 |

What changed with 80 more epochs (EMA, seeds 1-6 here vs seeds 4-9 for
e3_standard): DDIM 50 persistence 0.096 -> 0.115, all tracks/seq 28.1 -> 25.9
(fewer blips), target tracks/seq 3.14 (real 3.09).

Still below real: persistence 0.115-0.130 vs 0.144, and velocity consistency
1.63-1.86 vs 1.36 (motion slightly jerkier). Recommended setting: EMA weights,
DDIM 50 (best frame-level match: std 0.961, marginal L1 0.061, target tracks
3.14) or DDIM 30 for more persistence margin.

**Setting to use: EMA weights, DDIM 30.** It is the only arm with every seed
above the persistence limit (0.111-0.145, 0 of 6 below). EMA DDIM 50 matches
frames slightly better but 2 of 6 seeds fall just under the limit (0.099, 0.108).
