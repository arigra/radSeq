#!/usr/bin/env bash
# Research plan step 2, unattended: small-data conditional DiT -> quality check
# -> synthetic data -> detector augmentation experiment. Every stage is safe to
# rerun: training resumes from last.pt, finished stages are skipped.
# GENERATOR_ONLY=1 stops after the generator quality check.
# Progress: logs/step2_pipeline.log   Rule/verdict: docs/notes/2026-09-17-step2-synthetic-augmentation.md
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:scripts
PY=/truenas/home/arigra/.venv/bin/python
LOG=logs/step2_pipeline.log
NOTE=docs/notes/2026-09-17-step2-synthetic-augmentation.md
say() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

# anchored to the python executable so a shell whose command line merely mentions
# these scripts (e.g. the one launching this pipeline) does not count
if pgrep -f "^$PY (-m src.train --config configs/cond_traj_n2000_bs|scripts/gen_synthetic.py|scripts/detector_augmentation.py)" > /dev/null; then
  say "REFUSED: a step-2 process is already running"; exit 1
fi
say "start"
if ! CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests/test_detector.py tests/test_trajectory_condition.py \
     tests/test_trajectory_adherence.py >> "$LOG" 2>&1; then
  say "FAILED: unit tests"; exit 1
fi

CKPT=""
for BS in 32 16 8; do
  CFG=configs/cond_traj_n2000_bs$BS.yaml
  sed -e "s/^  batch_size: .*/  batch_size: $BS/" \
      -e "s#checkpoints/cond_traj_n2000\$#checkpoints/cond_traj_n2000_bs$BS#" \
      -e "s#logs/cond_traj_n2000.log#logs/cond_traj_n2000_bs$BS.log#" \
      configs/cond_traj_n2000.yaml > "$CFG"
  RESUME=()
  if [ -f "checkpoints/cond_traj_n2000_bs$BS/last.pt" ]; then
    RESUME=(--resume "checkpoints/cond_traj_n2000_bs$BS/last.pt")
  fi
  say "generator training batch=$BS ($CFG) ${RESUME[*]}"
  if $PY -m src.train --config "$CFG" "${RESUME[@]}" >> "logs/cond_traj_n2000_bs$BS.console.log" 2>&1; then
    CKPT=checkpoints/cond_traj_n2000_bs$BS/last.pt; break
  fi
  if tail -50 "logs/cond_traj_n2000_bs$BS.console.log" | grep -qiE "out of memory|OutOfMemoryError"; then
    say "OOM at batch=$BS, retrying smaller"; continue
  fi
  say "FAILED: generator training crashed, see logs/cond_traj_n2000_bs$BS.console.log (rerun to resume)"; exit 1
done
[ -z "$CKPT" ] && { say "FAILED: OOM at every batch size"; exit 1; }

if [ ! -f samples/cond_traj_n2000_scores.json ]; then
  say "generator quality check (trajectory rule, w=1)"
  if ! $PY scripts/score_conditional.py --ckpt "$CKPT" --uncond-ckpt checkpoints/e3_long_bs32/last.pt \
       --guidance 1 --out samples/cond_traj_n2000_scores.json --note "$NOTE" \
       --png samples/cond_traj_n2000.png >> "$LOG" 2>&1; then
    say "FAILED: quality check (rerun this script)"; exit 1
  fi
fi

if [ "${GENERATOR_ONLY:-0}" = "1" ]; then
  say "DONE (GENERATOR_ONLY): small-data generator trained and quality-checked; detector stages skipped"
  exit 0
fi

say "rendering synthetic sequences"
if ! $PY scripts/gen_synthetic.py --ckpt "$CKPT" --n-labels 2000 --repeats 4 --guidance 1 \
     --out data/synthetic/cond_traj_n2000 >> "$LOG" 2>&1; then
  say "FAILED: synthetic generation (rerun this script; finished parts are kept)"; exit 1
fi

say "detector experiment"
if ! $PY scripts/detector_augmentation.py --synthetic data/synthetic/cond_traj_n2000 --n-real 2000 \
     --out samples/detector_augmentation.json --note "$NOTE" >> "$LOG" 2>&1; then
  say "FAILED: detector experiment (rerun this script)"; exit 1
fi
say "DONE: verdict appended to $NOTE"
