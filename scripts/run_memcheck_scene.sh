#!/usr/bin/env bash
# Memorisation check for each RADIal-fine-tuned generator, once step 4 is done.
# Reading the detector arms depends on it: a copying generator adds little
# beyond the real data the detector already has.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/memcheck_scene.log
export PYTHONPATH=.
until grep -q "| DONE" logs/step4.log 2>/dev/null; do sleep 120; done
for variant in engineer fitted; do
  out=samples/memorization_radial_${variant}.json
  ckpt=checkpoints/finetune_${variant}/last.pt
  if [ -f "$out" ] || [ ! -f "$ckpt" ]; then continue; fi
  echo "$(date '+%F %T') | $variant" >> "$LOG"
  "$PY" scripts/memorization_check_scene.py --ckpt "$ckpt" --n 32 --out "$out" >> "$LOG" 2>&1
done
echo "$(date '+%F %T') | DONE" >> "$LOG"
