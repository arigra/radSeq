#!/usr/bin/env bash
# Step 3 recipe check, started once the engineer's scene cache is complete.
# Resumable: finished runs are skipped on re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/scene_smoke.log
until [ -f data/scene_engineer/DONE ]; do sleep 60; done
echo "$(date '+%F %T') | start" >> "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  "$PY" scripts/scene_dit_smoke.py >> "$LOG" 2>&1
echo "$(date '+%F %T') | DONE" >> "$LOG"
