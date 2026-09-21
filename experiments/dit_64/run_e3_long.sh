#!/usr/bin/env bash
# Unattended longer training of the passing factorized full-data model.
# Safe to rerun at any time: training resumes from the newest checkpoint
# (last.pt, written atomically every 1000 steps and at each epoch end), and a
# finished training run skips straight to scoring.
# Progress: experiments/dit_64/logs/e3_long_pipeline.log   Rule/verdict: docs/notes/2026-09-15-e3-longer-training.md
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:experiments/dit_64:experiments/dit_64
PY=/truenas/home/arigra/.venv/bin/python
LOG=experiments/dit_64/logs/e3_long_pipeline.log
NOTE=docs/notes/2026-09-15-e3-longer-training.md
say() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

if pgrep -f "src.train --config experiments/dit_64/configs/e3_long_bs" > /dev/null; then
  say "REFUSED: an e3_long training process is already running"; exit 1
fi
say "start"
if ! CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests/test_diffusion.py tests/test_dit.py >> "$LOG" 2>&1; then
  say "FAILED: unit tests"; exit 1
fi

CKPT=""
for BS in 32 16 8; do
  CFG=experiments/dit_64/configs/e3_long_bs$BS.yaml
  sed -e "s/^  batch_size: .*/  batch_size: $BS/" \
      -e "s#checkpoints/e3_long\$#checkpoints/e3_long_bs$BS#" \
      -e "s#experiments/dit_64/logs/e3_long.log#experiments/dit_64/logs/e3_long_bs$BS.log#" \
      experiments/dit_64/configs/e3_long.yaml > "$CFG"
  RESUME=()
  if [ -f "checkpoints/e3_long_bs$BS/last.pt" ]; then
    RESUME=(--resume "checkpoints/e3_long_bs$BS/last.pt")
  fi
  say "training batch=$BS ($CFG) ${RESUME[*]}"
  if $PY -m src.train --config "$CFG" "${RESUME[@]}" >> "experiments/dit_64/logs/e3_long_bs$BS.console.log" 2>&1; then
    CKPT=checkpoints/e3_long_bs$BS/last.pt; break
  fi
  if tail -50 "experiments/dit_64/logs/e3_long_bs$BS.console.log" | grep -qiE "out of memory|OutOfMemoryError"; then
    say "OOM at batch=$BS, retrying smaller"; continue
  fi
  say "FAILED: training crashed, see experiments/dit_64/logs/e3_long_bs$BS.console.log (rerun this script to resume)"; exit 1
done
[ -z "$CKPT" ] && { say "FAILED: OOM at every batch size"; exit 1; }

say "scoring $CKPT"
if ! $PY experiments/dit_64/score_hard_standard.py --ckpt "$CKPT" --steps 30,50 --seeds 1,2,3,4,5,6 \
     --out experiments/dit_64/results/e3_long_scores.json --note "$NOTE" --png experiments/dit_64/results/e3_long.png >> "$LOG" 2>&1; then
  say "FAILED: scoring (rerun this script; training will be skipped)"; exit 1
fi
say "DONE: verdict appended to $NOTE"
