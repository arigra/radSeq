#!/usr/bin/env bash
# Step 9 (docs/notes/2026-10-24-step9-generator-realism.md): screen recipe
# changes on simulator data at 7,500 steps, train the winner to 60k, gate it
# against simulator val, then fine-tune on fold 0's full real training set and
# gate against real val. Resumable; heartbeat keeps two copies apart.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/step9.log
LOCK=logs/step9.heartbeat
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }
if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another run_step9.sh is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT
say "start"
mkdir -p samples/step9_manifests
stamp=$(date +%Y%m%d_%H%M%S)
"$PY" scripts/write_manifest.py --allow-dirty --paths configs src scripts \
    --out "samples/step9_manifests/launch_${stamp}.json" >> "$LOG" 2>&1
git diff > "samples/step9_manifests/launch_${stamp}.diff"

# pretrain <variant> <epochs>: config from pretrain_engineer.yaml with one change
pretrain() {
  local v=$1 epochs=$2 dir="checkpoints/step9_$1" cfg="configs/step9_$1.yaml"
  "$PY" - "$v" "$epochs" "$cfg" "$dir" <<'PYEOF'
import sys, yaml
v, epochs, out, ckpt = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = yaml.safe_load(open("configs/pretrain_engineer.yaml"))
if v == "shift4":
    cfg["diffusion"]["schedule_shift"] = 4.0
elif v == "patch8":
    cfg["model"].update(patch=8, stride=8)
elif v != "base":
    raise SystemExit(f"unknown variant {v}")
cfg["train"].update(epochs=epochs, snapshot_every_epochs=10, ckpt_dir=ckpt,
                    log_file=f"logs/step9_{v}.log")
open(out, "w").write(f"# Step 9 pretraining '{v}' (written by run_step9.sh)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
  if [ "$v" = base ]; then return; fi            # the 60k-step pretrain_engineer run
  local done_marker="$dir/DONE_${epochs}"
  if [ ! -f "$done_marker" ]; then
    local args=(--config "$cfg")
    [ -f "$dir/last.pt" ] && args+=(--resume "$dir/last.pt")
    say "pretraining $v to $epochs epochs"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
    touch "$done_marker"
  fi
}
ckpt_of() {  # ckpt_of <variant> <epoch>
  if [ "$1" = base ]; then echo "checkpoints/pretrain_engineer/epoch_$(printf %04d "$2").pt"
  else echo "checkpoints/step9_$1/epoch_$(printf %04d "$2").pt"; fi
}

# part 2: screen at 7,500 steps (10 epochs)
VARIANTS="${VARIANTS:-shift4 patch8}"
for v in $VARIANTS; do
  pretrain "$v" 10
  out="samples/step9_screen_${v}.json"
  grep -q '"chosen"' "$out" 2>/dev/null || "$PY" scripts/realism_sweep.py \
      --ckpt "${v}_s7500=$(ckpt_of "$v" 10)" --weights raw --cache data/scene_engineer --out "$out" >> "$LOG" 2>&1
done
winner=$("$PY" - $VARIANTS <<'PYEOF'
import json, sys
base = json.load(open("samples/step9_sim_realism_by_steps.json"))["candidates"]["base_s7500_raw"]["relative_error"]
wins = {}
for v in sys.argv[1:]:
    e = json.load(open(f"samples/step9_screen_{v}.json"))["candidates"][f"{v}_s7500_raw"]["relative_error"]
    if (base["off_label_peaks"] - e["off_label_peaks"] >= 0.10
            and all(e[k] - base[k] <= 0.10 for k in e if k != "off_label_peaks")):
        wins[v] = sum(e.values()) / len(e)
print(min(wins, key=wins.get) if wins else "base")
PYEOF
)
say "screen winner: $winner"

# part 3a: the winner to 60k steps, gated against simulator val
pretrain "$winner" 80
simgate="samples/step9_simgate_${winner}.json"
grep -q '"chosen"' "$simgate" 2>/dev/null || "$PY" scripts/realism_sweep.py \
    --ckpt "${winner}_s60000=$(ckpt_of "$winner" 80)" --cache data/scene_engineer --out "$simgate" >> "$LOG" 2>&1
read -r acc key < <("$PY" -c "import json; c=json.load(open('$simgate'))['chosen']; print(c['accepted'], c['key'])")
if [ "$acc" != True ]; then
  say "$winner at 60k FAILED the simulator realism gate ($simgate)"; say "DONE"; exit 0
fi
say "$winner at 60k passed the simulator realism gate ($key)"

# part 3b: fine-tune on all of fold 0's real training sequences (427)
pre=$(ckpt_of "$winner" 80)
ft=checkpoints/step9_${winner}_fullreal_fold0 cfg=configs/step9_${winner}_fullreal_fold0.yaml
"$PY" - "$winner" "$pre" "$ft" "$cfg" <<'PYEOF'
import sys, yaml
v, pre, ft, out = sys.argv[1:5]
cfg = yaml.safe_load(open("configs/finetune_engineer.yaml"))
src = yaml.safe_load(open(f"configs/step9_{v}.yaml"))
cfg["model"], cfg["diffusion"] = src["model"], src["diffusion"]
cfg["data"].update(cache_dir="data/radial_fold0")
cfg["train"].update(epochs=200, init_from=pre, ckpt_dir=ft, snapshot_every_epochs=20,
                    log_file=f"logs/step9_{v}_fullreal_fold0.log")
open(out, "w").write(f"# Step 9: '{v}' fine-tuned on all of fold 0's real training data\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
if [ ! -f "$ft/STAGE_DONE" ]; then
  args=(--config "$cfg"); [ -f "$ft/last.pt" ] && args+=(--resume "$ft/last.pt")
  say "fine-tuning $winner on fold 0's full real training set"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
  touch "$ft/STAGE_DONE"
fi
realgate=samples/step9_realgate_${winner}_fullreal_fold0.json
if ! grep -q '"chosen"' "$realgate" 2>/dev/null; then
  cands=()
  for ep in 20 40 60 80 100 120 140 160 180 200; do
    cands+=(--ckpt "${winner}_full_ep${ep}=$ft/epoch_$(printf %04d $ep).pt")
  done
  "$PY" scripts/realism_sweep.py "${cands[@]}" --cache data/radial_fold0 --out "$realgate" >> "$LOG" 2>&1
fi
read -r acc key < <("$PY" -c "import json; c=json.load(open('$realgate'))['chosen']; print(c['accepted'], c['key'])")
say "full real data: chosen $key, held-out real gate accepted=$acc ($realgate)"
say "DONE"
