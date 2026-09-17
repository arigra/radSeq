# Step 2: does synthetic data from the conditional DiT help a detector? (2026-09-17)

Written before the run. Ari approved the setup (N = 2,000, the rule below) and
asked for it to run unattended.

## Why the generator is retrained

The step-1 conditional DiT was trained on all 20,000 sequences, so using it to
augment a small set would leak data. Here the generator sees **only the first
2,000 training sequences** (`data.train_subset: 2000`), trained from scratch
(no warm start). Synthetic requests are those same 2,000 labels, each rendered
4 times with different seeds (8,000 synthetic sequences). This mirrors the
RADIal situation (~4,600 usable frames) on simulator data.

Minor leak, accepted: map normalisation uses the full training cache's mean and
std (two numbers).

## Setup

- Generator: `configs/cond_traj_n2000.yaml` (the step-1 recipe, 640 epochs,
  ~39,700 steps), EMA weights, DDIM 30, guidance w = 1 (step 1's choice).
- Generator quality check: the step-1 trajectory rule on held-out requests
  (reported; it does not gate the detector experiment).
- Detector: `src/detector.py` (small U-Net, single frames, heatmap of target
  centres, CenterNet focal loss), 3,000 steps x 256 frames, AdamW 1e-3,
  3 seeds per arm. Metric: average precision, detection correct within 2 bins,
  on val sequences 1000-1511 (8,192 frames).
- Arms: `real_n` (2,000 real), `real_n_synth` (2,000 real + 8,000 synthetic),
  `synth_only` (8,000 synthetic), `real_full` (20,000 real, upper bound).

## Decision rule

Synthetic data **helps** if mean AP(`real_n_synth`) - mean AP(`real_n`) is
larger than the bigger of the two arms' seed ranges (max - min over 3 seeds).
`synth_only` and `real_full` are reported for context, not used in the rule.

## If the session closes

Rerun `setsid nohup bash scripts/run_step2.sh > logs/step2_pipeline.console.log 2>&1 < /dev/null &`
in a new session, or `sbatch scripts/step2.sbatch` from `ece-hpc`. Training
resumes from its last checkpoint; finished stages are skipped.

## Found in the smoke test, before launch: the detection task is saturated

With 8 real training sequences and 20 training steps the detector already
scores AP 0.997 (8 test sequences), and a detector with **no learning at all**
(score = normalised brightness, local maxima) scores **AP 0.94** on val
1000-1127 (128 sequences, 2.96 targets per frame), at 2 bins and at 1 bin. On
this simulator the targets are too bright relative to clutter for detection AP
to measure any benefit of extra training data, so the rule above could only
return "does not help" regardless of the generator's quality.

**Change:** only the first stages run now (`GENERATOR_ONLY=1`): the
small-data generator (2,000 sequences, from scratch) and its trajectory-rule
quality check, which answers whether the recipe works at RADIal-like data size.
The detector experiment is on hold until a task with headroom is chosen (for
example weak targets only, target class, or much smaller N), and its rule will
be re-registered then.
