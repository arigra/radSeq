#!/usr/bin/env bash
# Start the real RADIal experiment as soon as its inputs exist: the real
# sequence cache and both pretrained DiTs.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/step4.log
echo "$(date '+%F %T') | waiting for the real cache and both pretrainings" >> "$LOG"
until [ -f data/radial_seq/DONE ] \
   && [ -f checkpoints/pretrain_engineer/STAGE_DONE ] \
   && [ -f checkpoints/pretrain_fitted/STAGE_DONE ]; do sleep 60; done
exec bash scripts/run_step4.sh
