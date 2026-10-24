#!/usr/bin/env bash
# Step 9 part 3b (docs/notes/2026-10-24-step9-generator-realism.md): fine-tune
# the simulator-realistic snapshot (pretrain_engineer epoch 40 = 30k steps, raw
# weights) on all of fold 0's real training sequences, then choose a snapshot by
# realism on real val 0-15 and gate it on val 16-31. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/step9.log
LOCK=logs/step9.heartbeat
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }
if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another step 9 copy is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT
mkdir -p samples/step9_manifests
stamp=$(date +%Y%m%d_%H%M%S)
"$PY" scripts/write_manifest.py --allow-dirty --paths configs src scripts \
    --out "samples/step9_manifests/launch_${stamp}.json" >> "$LOG" 2>&1
git diff > "samples/step9_manifests/launch_${stamp}.diff"

ft=checkpoints/step9_fullreal_fold0 cfg=configs/step9_fullreal_fold0.yaml
"$PY" - "$ft" "$cfg" <<'PYEOF'
import sys, yaml
ft, out = sys.argv[1:3]
cfg = yaml.safe_load(open("configs/finetune_engineer.yaml"))
cfg["data"].update(cache_dir="data/radial_fold0")
cfg["train"].update(epochs=200, init_from="checkpoints/pretrain_engineer/epoch_0040.pt",
                    init_weights="raw", ckpt_dir=ft, snapshot_every_epochs=20,
                    log_file="logs/step9_fullreal_fold0.log")
open(out, "w").write("# Step 9: from the 30k-step raw weights, fine-tuned on all of fold 0's "
                     "real training data (written by run_step9_fullreal.sh)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
if [ ! -f "$ft/STAGE_DONE" ]; then
  args=(--config "$cfg"); [ -f "$ft/last.pt" ] && args+=(--resume "$ft/last.pt")
  say "full real data: fine-tuning from the 30k raw snapshot"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
  touch "$ft/STAGE_DONE"
fi
gate=samples/step9_realgate_fullreal_fold0.json
if ! grep -q '"chosen"' "$gate" 2>/dev/null; then
  cands=()
  for ep in 20 40 60 80 100 120 140 160 180 200; do
    cands+=(--ckpt "full_ep${ep}=$ft/epoch_$(printf %04d $ep).pt")
  done
  say "full real data: realism sweep"
  "$PY" scripts/realism_sweep.py "${cands[@]}" \
      --reference pre30k=checkpoints/pretrain_engineer/epoch_0040.pt \
      --cache data/radial_fold0 --out "$gate" >> "$LOG" 2>&1
fi
read -r acc key < <("$PY" -c "import json; c=json.load(open('$gate'))['chosen']; print(c['accepted'], c['key'])")
say "full real data: chosen $key, held-out real gate accepted=$acc ($gate)"
say "DONE"
