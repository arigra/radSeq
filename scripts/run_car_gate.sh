#!/usr/bin/env bash
# Check of the new vehicle model before any GPU-heavy redo
# (docs/design/2026-10-22-vehicle-scattering-design.md): a small engineer's
# cache with the new car, then a detector trained on it alone, scored on the
# real VAL recordings. Pass: sim-only AP >= 0.1. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/car_gate.log
export PYTHONPATH=.
cache=data/scene_engineer_cargate
if [ ! -f "$cache/DONE" ]; then
  echo "$(date '+%F %T') | building $cache" >> "$LOG"
  "$PY" scripts/build_scene_cache.py --variant engineer --n-train 300 --n-val 30 \
      --out "$cache" >> "$LOG" 2>&1
fi
out=samples/car_gate_sim_only.json
if ! grep -q '"complete": true' "$out" 2>/dev/null; then
  echo "$(date '+%F %T') | sim-only detector -> $out" >> "$LOG"
  "$PY" scripts/radial_detector_arms.py --variant engineer --sim-cache "$cache" \
      --arms sim --eval-split val --out "$out" >> "$LOG" 2>&1
fi
echo "$(date '+%F %T') | DONE" >> "$LOG"
