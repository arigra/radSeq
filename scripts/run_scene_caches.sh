#!/usr/bin/env bash
# Build the 8-frame training caches for both scene-simulator variants.
# Resumable: re-running continues each variant from its last finished chunk.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/scene_caches.log
for variant in engineer fitted; do
  if [ -f "data/scene_${variant}/DONE" ]; then
    echo "$(date '+%F %T') | $variant already built" >> "$LOG"; continue
  fi
  echo "$(date '+%F %T') | building $variant" >> "$LOG"
  PYTHONPATH=. "$PY" scripts/build_scene_cache.py --variant "$variant" >> "$LOG" 2>&1
done
echo "$(date '+%F %T') | DONE" >> "$LOG"
