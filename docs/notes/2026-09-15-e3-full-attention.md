# Full data with joint space-time attention (2026-09-15)

Written before the run.

## Why

`configs/e3_standard.yaml` (factorized attention) passes Ari's rule on the full
data only with EMA weights and 20-30 DDIM steps
(`2026-09-14-hard-standard-recipe.md`, seeds 4-9). Its persistence margin is
thin (0.121 vs limit 0.108; 2 of 6 seeds below on their own), and continuity is
lost in the high-noise stage (t >= 700) where the global temporal layout is
decided. Factorized attention decides time and space in alternating sublayers;
joint attention over all 16 x 64 tokens is the standard video-DiT choice and
the most direct training-side fix for that stage.

## Setup

`configs/e3_full.yaml`: identical to `e3_standard` except `attn_mode: full`
(`FullBlock` in `src/dit.py`). 100 epochs, batch 32, bf16, EMA 0.9999.
Scored by `scripts/score_hard_standard.py`: raw and EMA weights, DDIM 30 and
50, seeds 1-6 x 32 sequences (fresh for this model).

## Decision rule (Ari's rule, unchanged)

Passes if std within 0.1 of real, marginal L1 <= 0.15, target tracks/seq within
15% of real, persistence within 25% of real.

**Replaces the e3_standard model** if EMA DDIM 30 passes all four with
persistence >= 0.121 (e3_standard's value), or EMA DDIM 50 passes all four
(which e3_standard does not). Otherwise e3_standard with EMA DDIM 30 stays.

## If the session closes

Nothing is lost beyond at most ~1000 steps (~4 min). To continue, either
- in a new session: `setsid nohup bash scripts/run_e3_full.sh > logs/e3_full_pipeline.console.log 2>&1 < /dev/null &`
- or from the login host `ece-hpc`: `sbatch scripts/e3_full.sbatch`

Both resume from `checkpoints/e3_full_bs32/last.pt`; the script refuses to
start if a training process is already running. Progress:
`logs/e3_full_pipeline.log` (ends in DONE or FAILED) and `logs/e3_full_bs32.log`.
Note: resuming restarts the interrupted epoch, so the run may take up to 625
extra steps.

## Launch (2026-09-15 ~08:12, session job 9687, ends 14:55)

Smoke test before launch: 26 unit tests pass; 40 training steps at 5.2
steps/s, 8.3 GB peak; resume from a checkpoint verified (step 40 -> 60);
scoring wrote JSON, image and note. Expected training time ~3.2 h (62,500
steps), so the result should land around 11:30, well inside the session.

**Declared confound:** the joint-attention model has 32.6 M parameters vs
45.0 M for factorized (one attention per block instead of two). If it wins,
that is despite fewer parameters; if it loses, capacity is a possible cause.

## Result (`samples/e3_full_scores.json`)

Checkpoint `checkpoints/e3_full_bs32/last.pt`, step 62500, 32 sequences x seeds 1,2,3,4,5,6.

| arm | std | marginal L1 | target tracks/seq | persistence | verdict |
|---|---:|---:|---:|---:|---|
| real | 1.004 | 0.065 (floor) | 3.09 | 0.144 | |
| raw_ddim30 | 0.922 ok | 0.120 ok | 2.81 ok | 0.069 FAIL | fails |
| raw_ddim50 | 0.945 ok | 0.097 ok | 2.66 ok | 0.058 FAIL | fails |
| ema_ddim30 | 0.945 ok | 0.079 ok | 2.96 ok | 0.080 FAIL | fails |
| ema_ddim50 | 0.968 ok | 0.056 ok | 2.82 ok | 0.075 FAIL | fails |

**Verdict (pre-set rule): the DiT does NOT yet pass on the full data; see which checks fail.**

## Reading (added after the result)

**Not a replacement; `e3_standard` (factorized) with EMA DDIM 30 stays.**
Joint attention matches or beats factorized on frame-level checks (EMA DDIM 50:
std 0.968, marginal L1 0.056) but is clearly worse on continuity:

| EMA DDIM 30 | persistence | target tracks/seq | all tracks/seq | velocity consistency |
|---|---:|---:|---:|---:|
| factorized, seeds 1-3 / 4-9 | 0.113 / 0.121 | 3.07 / 3.27 | 25.7 / 24.5 | 1.75 / 1.84 |
| joint attention, seeds 1-6 | 0.080 | 2.96 | 26.9 | 2.19 |
| real | 0.144 | 3.09 | 23.7 | 1.36 |

Final validation loss is also slightly worse (0.388 vs 0.382). Joint attention
does not fix the high-noise continuity loss. The declared confound applies:
it has 32.6 M parameters vs 45.0 M, so this does not separate "attention
pattern" from "capacity".
