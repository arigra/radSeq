#!/usr/bin/env bash
# Unattended full-data (E3) standard-recipe run: unit tests -> train (smaller batch on OOM)
# -> score -> append verdict to the note. Progress: experiments/dit_64/logs/e3_standard_pipeline.log
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:experiments/dit_64:experiments/dit_64
PY=/truenas/home/arigra/.venv/bin/python
LOG=experiments/dit_64/logs/e3_standard_pipeline.log
NOTE=docs/notes/2026-09-14-hard-standard-recipe.md
say() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

say "start"
if ! CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests/test_diffusion.py tests/test_dit.py >> "$LOG" 2>&1; then
  say "FAILED: unit tests"; exit 1
fi

CKPT=""
for BS in 32 16 8; do
  CFG=experiments/dit_64/configs/e3_standard_bs$BS.yaml
  sed -e "s/^  batch_size: .*/  batch_size: $BS/" \
      -e "s#checkpoints/e3_standard\$#checkpoints/e3_standard_bs$BS#" \
      -e "s#experiments/dit_64/logs/e3_standard.log#experiments/dit_64/logs/e3_standard_bs$BS.log#" \
      experiments/dit_64/configs/e3_standard.yaml > "$CFG"
  RESUME=()
  if [ -f "checkpoints/e3_standard_bs$BS/last.pt" ]; then
    RESUME=(--resume "checkpoints/e3_standard_bs$BS/last.pt")   # continue after a killed session
  fi
  say "training batch=$BS ($CFG) ${RESUME[*]}"
  if $PY -m src.train --config "$CFG" "${RESUME[@]}" >> "experiments/dit_64/logs/e3_standard_bs$BS.console.log" 2>&1; then
    CKPT=checkpoints/e3_standard_bs$BS/last.pt; break
  fi
  if grep -qiE "out of memory|OutOfMemoryError" "experiments/dit_64/logs/e3_standard_bs$BS.console.log"; then
    say "OOM at batch=$BS, retrying smaller"; continue
  fi
  say "FAILED: training crashed, see experiments/dit_64/logs/e3_standard_bs$BS.console.log"; exit 1
done
[ -z "$CKPT" ] && { say "FAILED: OOM at every batch size"; exit 1; }

say "scoring $CKPT"
if ! $PY experiments/dit_64/score_hard_standard.py --ckpt "$CKPT" --note "$NOTE" --png experiments/dit_64/results/e3_standard.png >> "$LOG" 2>&1; then
  say "FAILED: scoring"; exit 1
fi
say "DONE: verdict appended to $NOTE"
