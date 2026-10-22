#!/usr/bin/env bash
# Step 6: the redo with the new vehicle model
# (docs/design/2026-10-22-vehicle-scattering-design.md):
#  0  refuse unless the car gate passed (sim-only AP >= 0.1 on real val)
#  1  move the old-car simulator outputs aside (caches, pretrained DiTs, fit)
#  2  refit the fitted variant with the new car
#  3  rebuild both scene caches; 4  re-pretrain both DiTs (60k steps each)
#  5  fine-tune each on the 50-sequence real subset, render 4 x 50 sequences,
#     score five detector arms on real test, run the memorisation check
#
# Every stage is resumable. A heartbeat file marks a running copy: a second copy
# (an interactive one and a batch job, possibly on another node) refuses to
# start while the heartbeat is fresh, so two never train into the same
# checkpoints; step6.sbatch waits for it to go stale instead.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/step6.log
LOCK=logs/step6.heartbeat
N=50
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }

if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another run_step6.sh is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT

gate=samples/car_gate_sim_only.json
if ! "$PY" -c "import json,sys; sys.exit(json.load(open('$gate'))['arms']['sim']['mean'] < 0.1)" 2>/dev/null; then
  say "REFUSED: car gate not passed ($gate)"; exit 1
fi
say "start"

# 1. old-car outputs, kept as a record (once)
old=archive/old_car
if [ ! -f "$old/MOVED" ]; then
  mkdir -p "$old" data/old_car checkpoints/old_car
  for d in data/scene_engineer data/scene_fitted; do [ -d "$d" ] && mv "$d" data/old_car/; done
  for d in checkpoints/pretrain_engineer checkpoints/pretrain_fitted; do
    [ -d "$d" ] && mv "$d" checkpoints/old_car/
  done
  [ -f configs/scene_fitted.json ] && mv configs/scene_fitted.json "$old/"
  [ -f samples/scene_sim_fit.json ] && mv samples/scene_sim_fit.json "$old/"
  date '+%F %T' > "$old/MOVED"
  say "old-car caches -> data/old_car, pretrained DiTs -> checkpoints/old_car, fit -> $old"
fi

# 2. refit the fitted variant (train recordings only)
if [ ! -f configs/scene_fitted.json ]; then
  say "refitting the fitted variant"
  "$PY" scripts/fit_scene_simulator.py --out samples/scene_sim_fit.json >> "$LOG" 2>&1
  "$PY" - <<'EOF'
import json
fit = json.load(open("samples/scene_sim_fit.json"))
json.dump({"note": "Scene-simulator knobs fitted to RADIal TRAINING recordings by "
                   "scripts/fit_scene_simulator.py, with the 2026-10-22 vehicle model. "
                   "The 'fitted' variant: an upper bound on calibration, not the baseline.",
           "knobs": fit["fitted"]["knobs"]}, open("configs/scene_fitted.json", "w"), indent=2)
EOF
fi

# 3-4. caches and pretraining (both scripts skip finished variants and resume)
say "scene caches"
bash scripts/run_scene_caches.sh
say "pretraining"
bash scripts/run_pretrain.sh

# 5. fine-tune on the 50-sequence subset and score
"$PY" scripts/make_radial_subset.py --n "$N" >> "$LOG" 2>&1
cache="data/radial_seq_n${N}"
for variant in engineer fitted; do
  tag="${variant}_n${N}"
  cfg="configs/finetune_${tag}.yaml"
  ckpt="checkpoints/finetune_${tag}"
  # same recipe and 9,600 optimizer steps as the 387-sequence fine-tune; the
  # subset is repeated so an epoch stays ~50 steps
  "$PY" - "$variant" "$N" "$cfg" <<'EOF'
import math, sys, yaml
variant, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open(f"configs/finetune_{variant}.yaml"))
rep = math.ceil(400 / n)
per_epoch = n * rep // cfg["train"]["batch_size"]
cfg["data"].update(cache_dir=f"data/radial_seq_n{n}", train_repeat=rep)
cfg["train"].update(epochs=math.ceil(9600 / per_epoch), snapshot_every_epochs=0,
                    ckpt_dir=f"checkpoints/finetune_{variant}_n{n}",
                    log_file=f"logs/finetune_{variant}_n{n}.log")
open(out, "w").write(f"# Step 6: fine-tune the new-car pretrained DiT on the {n}-sequence "
                     f"real subset (written by run_step6.sh from finetune_{variant}.yaml)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
EOF
  if [ ! -f "$ckpt/STAGE_DONE" ]; then
    args=(--config "$cfg")
    [ -f "$ckpt/last.pt" ] && args+=(--resume "$ckpt/last.pt")
    say "$tag: fine-tuning"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
    touch "$ckpt/STAGE_DONE"
  fi
  say "$tag: rendering generated sequences"
  "$PY" scripts/gen_scene_synthetic.py --ckpt "$ckpt/last.pt" --cache "$cache" \
      --out "data/radial_synth_${tag}" --repeats 4 --guidance 1 >> "$LOG" 2>&1
  out="samples/step6_arms_${tag}.json"
  if ! grep -q '"complete": true' "$out" 2>/dev/null; then
    say "$tag: detector arms"
    "$PY" scripts/radial_detector_arms.py --variant "$variant" --real "$cache" \
        --synthetic "data/radial_synth_${tag}" --arms real,real_sim,real_dit,sim,dit \
        --out "$out" >> "$LOG" 2>&1
  fi
  mem="samples/step6_memorization_${tag}.json"
  if [ ! -f "$mem" ]; then
    say "$tag: memorisation check"
    "$PY" scripts/memorization_check_scene.py --ckpt "$ckpt/last.pt" --cache "$cache" \
        --n 32 --out "$mem" >> "$LOG" 2>&1
  fi
  say "$tag: done"
done
say "DONE"
