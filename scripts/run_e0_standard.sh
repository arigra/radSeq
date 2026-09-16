#!/usr/bin/env bash
# Unattended E0 standard-recipe run: unit tests -> train (smaller batch on OOM)
# -> score -> append verdict to the note. Progress: logs/e0_standard_pipeline.log
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:scripts
PY=/truenas/home/arigra/.venv/bin/python
LOG=logs/e0_standard_pipeline.log
NOTE=docs/notes/2026-09-14-e0-dit-standard-setup.md
say() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

say "start"
if ! CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests/test_diffusion.py tests/test_dit.py >> "$LOG" 2>&1; then
  say "FAILED: unit tests"; exit 1
fi

CKPT=""
for BS in 32 16 8; do
  CFG=configs/e0_standard_bs$BS.yaml
  sed -e "s/^  batch_size: .*/  batch_size: $BS/" \
      -e "s#checkpoints/e0_standard\$#checkpoints/e0_standard_bs$BS#" \
      -e "s#logs/e0_standard.log#logs/e0_standard_bs$BS.log#" \
      configs/e0_standard.yaml > "$CFG"
  say "training batch=$BS ($CFG)"
  if $PY -m src.train --config "$CFG" > "logs/e0_standard_bs$BS.console.log" 2>&1; then
    CKPT=checkpoints/e0_standard_bs$BS/last.pt; break
  fi
  if grep -qiE "out of memory|OutOfMemoryError" "logs/e0_standard_bs$BS.console.log"; then
    say "OOM at batch=$BS, retrying smaller"; continue
  fi
  say "FAILED: training crashed, see logs/e0_standard_bs$BS.console.log"; exit 1
done
[ -z "$CKPT" ] && { say "FAILED: OOM at every batch size"; exit 1; }

say "scoring $CKPT"
if ! $PY scripts/score_e0_standard.py --ckpt "$CKPT" --note "$NOTE" --png samples/e0_standard.png >> "$LOG" 2>&1; then
  say "FAILED: scoring"; exit 1
fi
say "DONE: verdict appended to $NOTE"
