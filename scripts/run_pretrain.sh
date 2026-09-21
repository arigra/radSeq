#!/usr/bin/env bash
# Step 3: pretrain the conditional DiT on each scene-simulator variant, in turn.
# Resumable: a finished variant is skipped, an interrupted one continues from
# its last checkpoint. One at a time -- two do not fit on the shared GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/pretrain.log
if pgrep -f "^$PY -m src.train --config configs/pretrain_" >/dev/null 2>&1; then
  echo "pretraining already running" >&2; exit 1
fi
for variant in engineer fitted; do
  ckpt=checkpoints/pretrain_${variant}
  if [ -f "$ckpt/STAGE_DONE" ]; then
    echo "$(date '+%F %T') | $variant already done" >> "$LOG"; continue
  fi
  args=(--config "configs/pretrain_${variant}.yaml")
  [ -f "$ckpt/last.pt" ] && args+=(--resume "$ckpt/last.pt")
  echo "$(date '+%F %T') | $variant: ${args[*]}" >> "$LOG"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
  touch "$ckpt/STAGE_DONE"
done
echo "$(date '+%F %T') | DONE" >> "$LOG"
