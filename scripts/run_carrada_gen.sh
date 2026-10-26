#!/usr/bin/env bash
# docs/notes/2026-10-26-carrada-generator.md: per patch size, pretrain the DiT
# on the CARRADA simulator, realism vs simulator, fine-tune on all real train,
# realism vs real val. Resumable; heartbeat keeps two copies apart.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/carrada_gen.log
LOCK=logs/carrada_gen.heartbeat
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }
if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; then
  say "REFUSED: another copy is running ($(cat "$LOCK"))"; exit 1
fi
echo "$(hostname -s) pid $$" > "$LOCK"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK"; sleep 60; done ) &
trap 'rm -f "$LOCK"' EXIT
say "start"
mkdir -p samples/carrada_manifests
stamp=$(date +%Y%m%d_%H%M%S)
"$PY" scripts/write_manifest.py --allow-dirty --paths configs src scripts \
    --out "samples/carrada_manifests/launch_${stamp}.json" >> "$LOG" 2>&1
git diff > "samples/carrada_manifests/launch_${stamp}.diff"

sim=data/carrada_sim_engineer
if [ ! -f "$sim/DONE" ]; then
  say "simulator cache 6000 + 300"
  "$PY" scripts/build_carrada_cache.py --sim --n-train 6000 --n-val 300 --out "$sim" >> "$LOG" 2>&1
fi
train() {  # train <config> <dir>
  local cfg=$1 dir=$2
  if [ ! -f "$dir/STAGE_DONE" ]; then
    local args=(--config "$cfg"); [ -f "$dir/last.pt" ] && args+=(--resume "$dir/last.pt")
    say "training $cfg"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
    touch "$dir/STAGE_DONE"
  fi
}
for p in 16 8; do
  pre=checkpoints/carrada_pre_p${p} pcfg=configs/carrada_pre_p${p}.yaml
  "$PY" - "$p" "$pre" "$pcfg" <<'PYEOF'
import sys, yaml
p, d, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open("configs/pretrain_engineer.yaml"))
cfg["data"].update(cache_dir="data/carrada_sim_engineer", n_range=256, n_doppler=64)
cfg["model"].update(patch=p, stride=p)
cfg["diffusion"]["schedule_shift"] = 8.0
cfg["train"].update(epochs=80, snapshot_every_epochs=10, ckpt_dir=d, log_file=f"logs/carrada_pre_p{p}.log")
open(out, "w").write(f"# CARRADA: DiT pretrained on the engineer simulator, patch {p} (run_carrada_gen.sh)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
  train "$pcfg" "$pre"
  simr=samples/carrada_simrealism_p${p}.json
  grep -q '"chosen"' "$simr" 2>/dev/null || "$PY" scripts/realism_sweep.py \
      --ckpt "p${p}_s30000=$pre/epoch_0040.pt" --ckpt "p${p}_s60000=$pre/epoch_0080.pt" \
      --cache "$sim" --n 32 --spread --ddma-slot 0 --peak-db 7.7 --out "$simr" >> "$LOG" 2>&1
  read -r skey sckpt sw sacc < <("$PY" -c "import json; c=json.load(open('$simr'))['chosen']; print(c['key'], c['ckpt'], c['weights'], c['accepted'])")
  say "p${p}: simulator realism, chosen $skey, held-out simulator gate accepted=$sacc"

  ft=checkpoints/carrada_ft_p${p} fcfg=configs/carrada_ft_p${p}.yaml
  "$PY" - "$p" "$sckpt" "$sw" "$ft" "$fcfg" <<'PYEOF'
import sys, yaml
p, init, w, d, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
cfg = yaml.safe_load(open("configs/finetune_engineer.yaml"))
src = yaml.safe_load(open(f"configs/carrada_pre_p{p}.yaml"))
cfg["model"], cfg["diffusion"] = src["model"], src["diffusion"]
cfg["data"].update(cache_dir="data/carrada_seq", n_range=256, n_doppler=64)
cfg["train"].update(epochs=77, snapshot_every_epochs=8, val_every_epochs=8, init_from=init,
                    init_weights=w, ckpt_dir=d, log_file=f"logs/carrada_ft_p{p}.log")
open(out, "w").write(f"# CARRADA: patch {p}, fine-tuned on all real train from {init} ({w})\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
  train "$fcfg" "$ft"
  realr=samples/carrada_realrealism_p${p}.json
  if ! grep -q '"chosen"' "$realr" 2>/dev/null; then
    c=(); for ep in 8 16 24 32 40 48 56 64 72; do c+=(--ckpt "p${p}_ft_ep${ep}=$ft/epoch_$(printf %04d $ep).pt"); done
    "$PY" scripts/realism_sweep.py "${c[@]}" --reference "p${p}_pre=$sckpt" \
        --cache data/carrada_seq --n 32 --spread --ddma-slot 0 --peak-db 5.0 --out "$realr" >> "$LOG" 2>&1
  fi
  read -r rkey racc < <("$PY" -c "import json; c=json.load(open('$realr'))['chosen']; print(c['key'], c['accepted'])")
  say "p${p}: real realism, chosen $rkey, held-out real gate accepted=$racc ($realr)"
done
say "DONE"
