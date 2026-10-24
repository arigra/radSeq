#!/usr/bin/env bash
# Step 8 (docs/notes/2026-10-24-step8-realistic-finetuning.md): three fine-tuning
# variants on fold 0, realism-chosen snapshot, held-out gate, then the step 7
# detector study on folds 0 and 1 and pooled verdicts. Resumable; a heartbeat
# file keeps two copies from running at once.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/step8.log
LOCK=logs/step8.heartbeat
N=50
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }

if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another run_step8.sh is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT
say "start"
mkdir -p samples/step8_manifests
stamp=$(date +%Y%m%d_%H%M%S)
"$PY" scripts/write_manifest.py --allow-dirty --paths configs src scripts \
    --out "samples/step8_manifests/launch_${stamp}.json" >> "$LOG" 2>&1
git diff > "samples/step8_manifests/launch_${stamp}.diff"

# finetune <variant> <fold> <epochs> <snapshot_every>: writes the config, trains, prints the dir
finetune() {
  local variant=$1 k=$2 epochs=$3 snap=$4
  local dir="checkpoints/step8_${variant}_fold${k}" cfg="configs/step8_${variant}_fold${k}.yaml"
  "$PY" - "$variant" "$k" "$epochs" "$snap" "$cfg" "$dir" "$N" <<'PYEOF'
import math, sys, yaml
variant, k, epochs, snap, out, ckpt, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7])
cfg = yaml.safe_load(open("configs/finetune_engineer.yaml"))
cfg["data"].update(cache_dir=f"data/radial_fold{k}_n{n}", train_repeat=math.ceil(400 / n))
cfg["train"].update(epochs=epochs, snapshot_every_epochs=snap, val_every_epochs=4,
                    init_from="checkpoints/pretrain_engineer/last.pt",
                    ckpt_dir=ckpt, log_file=f"logs/step8_{variant}_fold{k}.log")
if variant == "lowlr":
    cfg["train"]["lr"] = 1.0e-5
if variant == "partial":
    cfg["train"]["trainable"] = ["^proj[.]", "adaLN", "^out[.]"]
open(out, "w").write(f"# Step 8 fine-tune '{variant}' on fold {k} (written by run_step8.sh)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
  if [ ! -f "$dir/STAGE_DONE" ]; then
    local args=(--config "$cfg")
    [ -f "$dir/last.pt" ] && args+=(--resume "$dir/last.pt")
    say "fine-tuning $variant on fold $k ($epochs epochs of 50 steps)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
    touch "$dir/STAGE_DONE"
  fi
}

# 1. three variants on fold 0, 2,000 steps, a snapshot every 200
for v in base lowlr partial; do finetune "$v" 0 40 4; done

# 2. choose by realism on val 0-15, gate on val 16-31
sweep0=samples/step8_sweep_fold0.json
if ! grep -q '"chosen"' "$sweep0" 2>/dev/null; then
  cands=()
  for v in base lowlr partial; do
    for ep in 4 8 12 16 20 24 28 32 36 40; do
      cands+=(--ckpt "${v}_s$((ep * 50))=checkpoints/step8_${v}_fold0/epoch_$(printf %04d "$ep").pt")
    done
  done
  say "realism sweep, fold 0 (${#cands[@]} args)"
  "$PY" scripts/realism_sweep.py "${cands[@]}" --reference pretrained=checkpoints/pretrain_engineer/last.pt \
      --cache "data/radial_fold0_n${N}" --out "$sweep0" >> "$LOG" 2>&1
fi
read -r key ckpt accepted < <("$PY" -c "import json; c=json.load(open('$sweep0'))['chosen']; print(c['key'], c['ckpt'], c['accepted'])")
if [ "$accepted" != "True" ]; then
  say "fold 0: chosen $key FAILED the held-out realism gate ($sweep0) -- stopping, per protocol"
  say "DONE (no variant keeps real-looking clutter)"; exit 0
fi
say "fold 0: chosen $key ($ckpt) passed the held-out gate"
variant=${key%%_s*}; rest=${key#*_s}; steps_ft=${rest%%_*}

# 3. fold 0 study
bash scripts/run_step7_study.sh 0 "engineer_fold0_n${N}_s8" "data/radial_fold0_n${N}" "$ckpt" "$sweep0" >> "$LOG" 2>&1

# 4. fold 1: the same recipe, not re-chosen, gated the same way
finetune "$variant" 1 $((steps_ft / 50)) $((steps_ft / 50))
ckpt1="checkpoints/step8_${variant}_fold1/epoch_$(printf %04d $((steps_ft / 50))).pt"
weights=${key##*_}
sweep1=samples/step8_sweep_fold1.json
if ! grep -q '"chosen"' "$sweep1" 2>/dev/null; then
  say "fold 1: gate for ${variant}_s${steps_ft}"
  "$PY" scripts/realism_sweep.py --ckpt "${variant}_s${steps_ft}=$ckpt1" --weights "$weights" \
      --cache "data/radial_fold1_n${N}" --out "$sweep1" >> "$LOG" 2>&1
fi
read -r acc1 < <("$PY" -c "import json; print(json.load(open('$sweep1'))['chosen']['accepted'])")
if [ "$acc1" != "True" ]; then
  say "fold 1: the recipe FAILED the held-out gate ($sweep1); pooled verdicts use fold 0 only"
else
  bash scripts/run_step7_study.sh 1 "engineer_fold1_n${N}_s8" "data/radial_fold1_n${N}" "$ckpt1" "$sweep1" >> "$LOG" 2>&1
fi

# 5. verdicts pooled over the folds that ran
for det in frame sequence; do
  reports=()
  for k in 0 1; do
    f="samples/step7_study_engineer_fold${k}_n${N}_s8_${det}.json"
    [ -f "$f" ] && reports+=("$f")
  done
  "$PY" scripts/step6_score.py --report "${reports[@]}" --dets-dir samples/step7_dets \
      --out "samples/step8_verdict_pooled_${det}.json" >> "$LOG" 2>&1
  say "pooled verdicts ($det, ${#reports[@]} folds) -> samples/step8_verdict_pooled_${det}.json"
done
say "DONE"
