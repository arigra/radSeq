#!/usr/bin/env bash
# Step 3's real experiment: for each simulator variant, fine-tune its
# pretrained DiT on real RADIal sequences, render labelled sequences from it,
# then score three detector arms (real / real+sim / real+DiT) on the held-out
# RADIal test recordings.
#
# Every stage is resumable: a finished one is skipped, an interrupted training
# continues from its checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/step4.log
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }

# Refuse to run twice (e.g. an interactive copy and a batch job): two copies
# would train into the same checkpoints.
if pgrep -f "^bash scripts/run_step4.sh" | grep -vx "$$" >/dev/null; then
  say "REFUSED: another run_step4.sh is already running"; exit 1
fi

if [ ! -f data/radial_seq/DONE ]; then say "REFUSED: real RADIal cache not built"; exit 1; fi
say "start"
for variant in engineer fitted; do
  ckpt=checkpoints/finetune_${variant}
  if [ ! -f "$ckpt/STAGE_DONE" ]; then
    args=(--config "configs/finetune_${variant}.yaml")
    [ -f "$ckpt/last.pt" ] && args+=(--resume "$ckpt/last.pt")
    say "$variant: fine-tuning"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
    touch "$ckpt/STAGE_DONE"
  fi
  say "$variant: rendering generated sequences"
  "$PY" scripts/gen_scene_synthetic.py --ckpt "$ckpt/last.pt" \
      --out "data/radial_synth_${variant}" --repeats 4 --guidance 1 >> "$LOG" 2>&1
  out=samples/step4_${variant}.json
  if [ ! -f "$out" ]; then
    say "$variant: detector arms"
    "$PY" scripts/radial_detector_arms.py --variant "$variant" \
        --synthetic "data/radial_synth_${variant}" --out "$out" >> "$LOG" 2>&1
  fi
  say "$variant: done"
done
say "DONE"
