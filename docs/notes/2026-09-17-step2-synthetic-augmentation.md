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

## Result (`samples/cond_traj_n2000_scores.json`)

Checkpoint `checkpoints/cond_traj_n2000_bs32/last.pt`. Guidance w = 1.0, chosen on val 0-95 / seeds 1-3 (w=1.0: hit 0.976, passes True). Verdict on val 96-287, seeds 4-9, 192 sequences.

| check | simulator / real | conditional DiT | |
|---|---:|---:|---|
| hit rate | 0.976 | 0.985 | ok |
| unrequested lasting tracks / seq | 0.33 | 0.24 | ok |
| std | 1.004 | 0.944 | ok |
| marginal L1 | 0.065 | 0.105 | ok |
| target tracks / seq | 3.09 | 3.04 | ok |
| persistence | 0.144 | 0.159 | ok |

Null check (unconditional e3_long, same requests and seeds): hit rate 0.013 -> ok (conditional hit rate is at least 0.20 higher)

**Verdict (pre-set rule): the conditional DiT follows requested trajectories.**

## Step 2b, registered before running (Ari chose the target-class task, 2026-09-17)

**Memorisation check** (`scripts/memorization_check.py`): 128 requests from the
generator's own training labels and 128 from held-out labels (val 1000-1127);
RMS difference (normalised) between each generated sequence and the real
sequence with the same labels. **Memorising if** median (training labels) <
0.8 x median (held-out labels).

**Target-class detector** (`scripts/detector_class_augmentation.py`): the
detector sees all 16 frames and outputs one heatmap per class (steady,
Swerling-1, extended) per frame; a detection counts only in its class channel,
within 2 bins of a true target of that class. Metric: mean AP over classes on
val 1000-1511. Same four arms, 3 seeds, 3,000 steps x 32 sequences. Synthetic
data: 8,000 sequences from the small-data generator (2,000 training labels x 4
seeds, w = 1).

**Decision rule:**
- *Informative* only if mean mAP(`real_full`) - mean mAP(`real_n`) exceeds the
  larger seed range of those two arms (the task has headroom). Otherwise the
  verdict is "uninformative", not "does not help".
- If informative, synthetic data **helps** if mean mAP(`real_n_synth`) - mean
  mAP(`real_n`) exceeds the larger seed range of those two arms.
- If the memorisation check says the generator copies, a positive result is
  reported but flagged as possibly reflecting copied real sequences.

Run: `scripts/run_step2b.sh` (resumable; `sbatch scripts/step2b.sbatch` from `ece-hpc`).

## Memorisation check (`samples/memorization_n2000.json`)

128 requests each, EMA, DDIM 30, w=1.0. RMS difference (normalised units) between a generated sequence and the real sequence with the same labels: training labels 0.795, held-out labels 1.112 (unrelated real pairs 1.423). Ratio 0.72 (memorising if < 0.8).

**Verdict (pre-set rule): the generator IS copying its training sequences.**

## Result, target-class task (`samples/detector_class_augmentation.json`)

Mean AP over classes (within 2 bins, correct class) on val 1000-1511, 3000 steps x 32 sequences per arm.

| arm | training sequences | mAP per seed | mean mAP | seed range |
|---|---:|---|---:|---:|
| real_n | 2000 | 0.676, 0.658, 0.660 | 0.665 | 0.018 |
| real_n_synth | 10000 | 0.701, 0.706, 0.707 | 0.705 | 0.006 |
| synth_only | 8000 | 0.678, 0.686, 0.703 | 0.689 | 0.025 |
| real_full | 20000 | 0.731, 0.766, 0.783 | 0.760 | 0.052 |

Headroom (real_full - real_n): +0.095 vs bar 0.052. Gain (real_n_synth - real_n): +0.040 vs bar 0.018.

**Verdict (pre-set rule): synthetic data from the conditional DiT HELPS the class detector.**
