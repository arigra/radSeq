#!/usr/bin/env bash
# Step 9 part 4 (docs/notes/2026-10-24-step9-generator-realism.md): the same DiT
# trained from scratch on fold 0's real training sequences only, then realism
# in-sample and held-out. Resumable.
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

dir=checkpoints/step9_scratch_real_fold0 cfg=configs/step9_scratch_real_fold0.yaml
"$PY" - "$dir" "$cfg" <<'PYEOF'
import sys, yaml
d, out = sys.argv[1:3]
cfg = yaml.safe_load(open("configs/pretrain_engineer.yaml"))
cfg["data"].update(cache_dir="data/radial_fold0")
per_epoch = 427 // cfg["train"]["batch_size"]          # 53 steps
cfg["train"].update(epochs=60000 // per_epoch + 1, snapshot_every_epochs=100,
                    val_every_epochs=50, ckpt_dir=d, log_file="logs/step9_scratch_real_fold0.log")
open(out, "w").write("# Step 9 part 4: from scratch on fold 0's real training data only, "
                     "60k steps (written by run_step9_scratch.sh)\n" + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
if [ ! -f "$dir/STAGE_DONE" ]; then
  args=(--config "$cfg"); [ -f "$dir/last.pt" ] && args+=(--resume "$dir/last.pt")
  say "part 4: training from scratch on real data only"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
  touch "$dir/STAGE_DONE"
fi
out=samples/step9_scratch_realism_fold0.json
if ! grep -q '"chosen"' "$out" 2>/dev/null; then
  cands=()
  for ep in 300 600 900 1100; do cands+=(--ckpt "scratch_ep${ep}=$dir/epoch_$(printf %04d $ep).pt"); done
  say "part 4: realism sweep (held-out val, 32+32)"
  "$PY" scripts/realism_sweep.py "${cands[@]}" --n 32 --cache data/radial_fold0 --out "$out" >> "$LOG" 2>&1
fi
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import json, torch
from scripts.realism_sweep import generated_stats, items_of
from src.eval.scene_realism import mismatch, realism, with_median
from src.scene_data import SceneSequenceDataset
r = json.load(open("samples/step9_scratch_realism_fold0.json"))
c = r["chosen"]
tr = SceneSequenceDataset("data/radial_fold0", "train", normalise=False)
idx = list(range(32))
real = with_median(realism(items_of(tr, idx)), items_of(tr, idx))
g = generated_stats(c["ckpt"], c["weights"], tr, idx, torch.device("cuda"), 250, 1.0)
rel, score = mismatch(g, real)
r["in_sample"] = {"key": c["key"], "relative_error": rel, "score": score, "stats": g, "real_train": real}
json.dump(r, open("samples/step9_scratch_realism_fold0.json", "w"), indent=2)
print("in-sample", c["key"], {k: round(v, 2) for k, v in rel.items()})
PYEOF
read -r acc key < <("$PY" -c "import json; c=json.load(open('$out'))['chosen']; print(c['accepted'], c['key'])")
say "part 4: from scratch, chosen $key, held-out real gate accepted=$acc ($out)"
say "DONE"
