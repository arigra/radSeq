#!/usr/bin/env bash
# Research plan step 2b, unattended: memorisation check of the small-data
# generator -> 8,000 synthetic sequences -> target-class detector experiment.
# Safe to rerun: finished stages are skipped, synthetic parts are kept.
# Progress: experiments/step2_detector/logs/step2b_pipeline.log   Rules/verdicts: docs/notes/2026-09-17-step2-synthetic-augmentation.md
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:experiments/step2_detector:experiments/dit_64
PY=/truenas/home/arigra/.venv/bin/python
LOG=experiments/step2_detector/logs/step2b_pipeline.log
NOTE=docs/notes/2026-09-17-step2-synthetic-augmentation.md
CKPT=checkpoints/cond_traj_n2000_bs32/last.pt
say() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

# anchored to the python executable so the launching shell does not count
if pgrep -f "^$PY scripts/(memorization_check|gen_synthetic|detector_class_augmentation).py" > /dev/null; then
  say "REFUSED: a step-2b process is already running"; exit 1
fi
say "start"
if ! CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests/test_detector.py >> "$LOG" 2>&1; then
  say "FAILED: unit tests"; exit 1
fi
if [ ! -f experiments/step2_detector/results/memorization_n2000.json ]; then
  say "memorisation check"
  $PY scripts/memorization_check.py --ckpt "$CKPT" --out experiments/step2_detector/results/memorization_n2000.json --note "$NOTE" >> "$LOG" 2>&1 \
    || { say "FAILED: memorisation check (rerun)"; exit 1; }
fi
say "rendering synthetic sequences"
$PY scripts/gen_synthetic.py --ckpt "$CKPT" --n-labels 2000 --repeats 4 --guidance 1 \
    --out data/synthetic/cond_traj_n2000 >> "$LOG" 2>&1 \
  || { say "FAILED: synthetic generation (rerun; finished parts are kept)"; exit 1; }
say "target-class detector experiment"
$PY experiments/step2_detector/detector_class_augmentation.py --synthetic data/synthetic/cond_traj_n2000 --n-real 2000 \
    --out experiments/step2_detector/results/detector_class_augmentation.json --note "$NOTE" >> "$LOG" 2>&1 \
  || { say "FAILED: detector experiment (rerun)"; exit 1; }
say "DONE: verdicts appended to $NOTE"
