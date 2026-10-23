#!/usr/bin/env bash
# Step 7, first pass (docs/design/2026-10-22-evaluation-redesign.md, approved by
# Ari 2026-10-22): engineer variant, folds 0 and 1 of a 5-fold recording-level
# cross-validation, vehicle-free frames included, same-direction traffic
# moving with the ego car.
#  0  freeze: manifest of the commit (and any uncommitted diff) per launch
#  1  diagnostic: does the DiT pretrained on the simulator already under-produce
#     clutter relative to SIMULATOR data? (old-traffic pretrained DiT, sim val)
#  2  move the old-traffic engineer cache and pretrained DiT aside
#  3  real caches: all recordings, folds 0/1, 50-sequence subsets
#  4  engineer scene cache with the new traffic; 5  re-pretrain (60k steps)
#  6  per fold: fine-tune on its 50 real sequences, early-stopped on val loss
#     (2026-10-23: 9,600 fixed steps overfit 50 sequences from the first
#     validation and failed the gate; the val-best checkpoint passed it);
#     7  realism gate on its val, with the val-best checkpoint.
#     A failed gate stops the study (protocol: no detector study on a generator
#     whose maps do not look real)
#  8  if scripts/run_step7_study.sh exists, the detector study per fold, then
#     verdicts pooled over the folds
# Every stage is resumable; a heartbeat file keeps two copies from running at once.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/step7.log
LOCK=logs/step7.heartbeat
N=50
FOLDS="0 1"
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }

if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another run_step7.sh is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT
say "start"

# 0. freeze what this launch runs
mkdir -p samples/step7_manifests
stamp=$(date +%Y%m%d_%H%M%S)
"$PY" scripts/write_manifest.py --allow-dirty --paths configs src scripts \
    --out "samples/step7_manifests/launch_${stamp}.json" >> "$LOG" 2>&1
git diff > "samples/step7_manifests/launch_${stamp}.diff"

# 1-2. the old step 6 pipeline must have stopped after engineer pretraining
until grep -q "STOPPED by Ari after engineer pretraining" logs/step6.log; do
  say "waiting for step 6 to stop"; sleep 120
done
old=checkpoints/old_traffic
if [ ! -f "$old/MOVED" ]; then
  diag=samples/step7_diag_pretrain_realism.json
  if ! grep -q '"chosen"' "$diag" 2>/dev/null; then
    say "diagnostic: pretrained DiT vs simulator val"
    "$PY" scripts/select_sampler.py --ckpt checkpoints/pretrain_engineer/last.pt \
        --cache data/scene_engineer --out "$diag" >> "$LOG" 2>&1
  fi
  mkdir -p "$old" data/old_traffic
  [ -d data/scene_engineer ] && mv data/scene_engineer data/old_traffic/
  [ -d checkpoints/pretrain_engineer ] && mv checkpoints/pretrain_engineer "$old/"
  date '+%F %T' > "$old/MOVED"
  say "old-traffic engineer cache -> data/old_traffic, pretrained DiT -> $old"
fi

# 3. real caches
if [ ! -f data/radial_all/DONE ]; then
  say "real cache, all recordings"
  "$PY" scripts/build_radial_cache.py --split-json data/radial/all.json --out data/radial_all >> "$LOG" 2>&1
fi
for k in $FOLDS; do
  "$PY" scripts/make_folds.py --build "$k" >> "$LOG" 2>&1
  "$PY" scripts/make_radial_subset.py --full "data/radial_fold${k}" --n "$N" >> "$LOG" 2>&1
done

# 4-5. engineer scene cache (new traffic) and pretraining
if [ ! -f data/scene_engineer/DONE ]; then
  say "engineer scene cache"
  "$PY" scripts/build_scene_cache.py --variant engineer >> "$LOG" 2>&1
fi
ckpt=checkpoints/pretrain_engineer
if [ ! -f "$ckpt/STAGE_DONE" ]; then
  args=(--config configs/pretrain_engineer.yaml)
  [ -f "$ckpt/last.pt" ] && args+=(--resume "$ckpt/last.pt")
  say "engineer pretraining ${args[*]}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> logs/pretrain.log 2>&1
  touch "$ckpt/STAGE_DONE"
fi

# 6-8. per fold
for k in $FOLDS; do
  tag="engineer_fold${k}_n${N}_es"
  cache="data/radial_fold${k}_n${N}"
  cfg="configs/step7_finetune_${tag}.yaml"
  ft="checkpoints/step7_finetune_${tag}"
  "$PY" - "$N" "$cache" "$ft" "$cfg" "$tag" <<'PYEOF'
import math, sys, yaml
n, cache, ft, out, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
cfg = yaml.safe_load(open("configs/finetune_engineer.yaml"))
rep = math.ceil(400 / n)
per_epoch = n * rep // cfg["train"]["batch_size"]
cfg["data"].update(cache_dir=cache, train_repeat=rep)
# at most the old 9,600 steps; validate every ~200 and stop after 4 without
# improvement; downstream stages use best.pt
cfg["train"].update(epochs=math.ceil(9600 / per_epoch), snapshot_every_epochs=0,
                    val_every_epochs=max(1, round(200 / per_epoch)),
                    early_stopping_patience=4,
                    init_from="checkpoints/pretrain_engineer/last.pt",
                    ckpt_dir=ft, log_file=f"logs/step7_finetune_{tag}.log")
open(out, "w").write(f"# Step 7: fine-tune on {cache} (written by run_step7.sh)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
  if [ ! -f "$ft/STAGE_DONE" ]; then
    args=(--config "$cfg")
    [ -f "$ft/last.pt" ] && args+=(--resume "$ft/last.pt")
    say "$tag: fine-tuning"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
    touch "$ft/STAGE_DONE"
  fi
  sel="samples/step7_sampler_${tag}.json"
  if ! grep -q '"chosen"' "$sel" 2>/dev/null; then
    say "$tag: realism gate"
    "$PY" scripts/select_sampler.py --ckpt "$ft/best.pt" --cache "$cache" --out "$sel" >> "$LOG" 2>&1
  fi
  if ! "$PY" -c "import json,sys; sys.exit(not json.load(open('$sel'))['chosen']['accepted'])"; then
    say "$tag: REALISM GATE FAILED ($sel) -- no detector study, per protocol"
    say "DONE (stopped at the realism gate)"
    exit 0
  fi
  say "$tag: realism gate passed"
  if [ -f scripts/run_step7_study.sh ]; then
    bash scripts/run_step7_study.sh "$k" "$tag" "$cache" "$ft/best.pt" "$sel" >> "$LOG" 2>&1
  else
    say "$tag: detector study not implemented yet (scripts/run_step7_study.sh missing); stopping"
    exit 0
  fi
done
for det in frame sequence; do
  reports=()
  for k in $FOLDS; do reports+=("samples/step7_study_engineer_fold${k}_n${N}_es_${det}.json"); done
  "$PY" scripts/step6_score.py --report "${reports[@]}" --dets-dir samples/step7_dets \
      --out "samples/step7_verdict_pooled_${det}.json" >> "$LOG" 2>&1
  say "pooled verdicts ($det) -> samples/step7_verdict_pooled_${det}.json"
done
say "DONE"
