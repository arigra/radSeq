#!/usr/bin/env bash
# Unattended trajectory-conditioned DiT run (research plan step 1).
# Safe to rerun: training resumes from the newest checkpoint (last.pt, written
# atomically every 1000 steps and at each epoch end); a finished run skips to scoring.
# Progress: logs/cond_traj_pipeline.log   Rule/verdict: docs/notes/2026-09-17-trajectory-conditioning.md
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:scripts
PY=/truenas/home/arigra/.venv/bin/python
LOG=logs/cond_traj_pipeline.log
NOTE=docs/notes/2026-09-17-trajectory-conditioning.md
say() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

if pgrep -f "src.train --config configs/cond_traj_bs" > /dev/null; then
  say "REFUSED: a cond_traj training process is already running"; exit 1
fi
say "start"
if ! CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests/test_trajectory_condition.py tests/test_dit.py \
     tests/test_trajectory_adherence.py >> "$LOG" 2>&1; then
  say "FAILED: unit tests"; exit 1
fi

CKPT=""
for BS in 32 16 8; do
  CFG=configs/cond_traj_bs$BS.yaml
  sed -e "s/^  batch_size: .*/  batch_size: $BS/" \
      -e "s#checkpoints/cond_traj\$#checkpoints/cond_traj_bs$BS#" \
      -e "s#logs/cond_traj.log#logs/cond_traj_bs$BS.log#" \
      configs/cond_traj.yaml > "$CFG"
  RESUME=()
  if [ -f "checkpoints/cond_traj_bs$BS/last.pt" ]; then
    RESUME=(--resume "checkpoints/cond_traj_bs$BS/last.pt")
  fi
  say "training batch=$BS ($CFG) ${RESUME[*]}"
  if $PY -m src.train --config "$CFG" "${RESUME[@]}" >> "logs/cond_traj_bs$BS.console.log" 2>&1; then
    CKPT=checkpoints/cond_traj_bs$BS/last.pt; break
  fi
  if tail -50 "logs/cond_traj_bs$BS.console.log" | grep -qiE "out of memory|OutOfMemoryError"; then
    say "OOM at batch=$BS, retrying smaller"; continue
  fi
  say "FAILED: training crashed, see logs/cond_traj_bs$BS.console.log (rerun this script to resume)"; exit 1
done
[ -z "$CKPT" ] && { say "FAILED: OOM at every batch size"; exit 1; }

say "scoring $CKPT"
if ! $PY scripts/score_conditional.py --ckpt "$CKPT" --uncond-ckpt checkpoints/e3_long_bs32/last.pt \
     --out samples/cond_traj_scores.json --note "$NOTE" --png samples/cond_traj.png >> "$LOG" 2>&1; then
  say "FAILED: scoring (rerun this script; training will be skipped)"; exit 1
fi
say "DONE: verdict appended to $NOTE"
