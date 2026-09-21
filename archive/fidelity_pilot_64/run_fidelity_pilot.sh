#!/usr/bin/env bash
# Sim-to-sim fidelity pilot: does generative fine-tuning substitute for
# simulator fidelity? See archive/fidelity_pilot_64/fidelity_pilot.py for the design.
#
# Every stage is resumable, so this can be re-run after a killed session and
# will continue from the last completed stage.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-$(command -v python3)}"
LOG=archive/fidelity_pilot_64/logs/fidelity_pilot.log

# Refuse to start a second copy: two runs would fight over the same checkpoints.
if pgrep -f "^$PY archive/fidelity_pilot_64/fidelity_pilot.py" >/dev/null 2>&1; then
  echo "fidelity pilot already running" >&2
  exit 1
fi

echo "$(date '+%F %T') | start" >> "$LOG"
PYTHONPATH=. "$PY" archive/fidelity_pilot_64/fidelity_pilot.py \
  --deltas "${DELTAS:-0,4,8,16}" \
  --n-real "${N_REAL:-200}" \
  --n-sim "${N_SIM:-6000}" \
  --pretrain-epochs "${PRETRAIN_EPOCHS:-40}" \
  --finetune-epochs "${FINETUNE_EPOCHS:-300}" \
  --detector-steps "${DETECTOR_STEPS:-2500}" \
  --out archive/fidelity_pilot_64/results/fidelity_pilot.json >> "$LOG" 2>&1
echo "$(date '+%F %T') | DONE" >> "$LOG"
