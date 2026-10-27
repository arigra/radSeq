#!/usr/bin/env bash
# docs/notes/2026-10-27-carrada-detector-study.md: fine-tune the CARRADA DiT on
# 50 real sequences, choose a snapshot by realism, render the simulator's scenes
# and the real labels, then the RADIal step 6 detector study on CARRADA.
# Resumable; heartbeat keeps two copies apart.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/carrada_detector.log
LOCK=logs/carrada_detector.heartbeat
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
    --out "samples/carrada_manifests/detector_${stamp}.json" >> "$LOG" 2>&1

N=50
real=data/carrada_seq_n${N}
[ -f "$real/DONE" ] || "$PY" scripts/make_radial_subset.py --n $N --full data/carrada_seq >> "$LOG" 2>&1
pre=checkpoints/carrada_pre_p16/epoch_0040.pt      # the simulator-gate choice (raw)
ft=checkpoints/carrada_ft50_p16 fcfg=configs/carrada_ft50_p16.yaml
"$PY" - "$pre" "$ft" "$fcfg" "$real" <<'PYEOF'
import sys, yaml
init, d, out, cache = sys.argv[1:5]
cfg = yaml.safe_load(open("configs/carrada_ft_p16.yaml"))
cfg["data"]["cache_dir"] = cache
cfg["train"].update(epochs=384, snapshot_every_epochs=32, val_every_epochs=32, init_from=init,
                    init_weights="raw", ckpt_dir=d, log_file="logs/carrada_ft50_p16.log")
open(out, "w").write(f"# CARRADA: patch 16, fine-tuned on {cache} from {init} (raw)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
PYEOF
if [ ! -f "$ft/STAGE_DONE" ]; then
  args=(--config "$fcfg"); [ -f "$ft/last.pt" ] && args+=(--resume "$ft/last.pt")
  say "training $fcfg"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m src.train "${args[@]}" >> "$LOG" 2>&1
  touch "$ft/STAGE_DONE"
fi
sel=samples/carrada_realrealism_n${N}.json
if ! grep -q '"chosen"' "$sel" 2>/dev/null; then
  c=(); for ep in 32 64 96 128 160 192 224 256 288 320 352 384; do c+=(--ckpt "n${N}_ep${ep}=$ft/epoch_$(printf %04d $ep).pt"); done
  "$PY" scripts/realism_sweep.py "${c[@]}" --reference "p16_pre=$pre" \
      --cache "$real" --n 32 --spread --ddma-slot 0 --peak-db 5.0 --out "$sel" >> "$LOG" 2>&1
fi
read -r key gen_ckpt steps eta weights acc < <("$PY" -c "import json; c=json.load(open('$sel'))['chosen']; print(c['key'], c['ckpt'], c['steps'], c['eta'], c['weights'], c['accepted'])")
say "n${N}: real realism, chosen $key, held-out real gate accepted=$acc ($sel)"
sampler=(--steps "$steps" --eta "$eta" --weights "$weights" --guidance 1)

simlab=data/carrada_synth_n${N}_simlabels reallab=data/carrada_synth_n${N}_reallabels
say "rendering the simulator's scenes"
"$PY" scripts/gen_scene_synthetic.py --ckpt "$gen_ckpt" --cache data/carrada_sim_engineer \
    --split train --limit $((4 * N)) --repeats 1 "${sampler[@]}" --out "$simlab" >> "$LOG" 2>&1
say "rendering the real labels"
"$PY" scripts/gen_scene_synthetic.py --ckpt "$gen_ckpt" --cache "$real" \
    --repeats 4 "${sampler[@]}" --out "$reallab" >> "$LOG" 2>&1
mem=samples/carrada_memorization_n${N}.json
[ -f "$mem" ] || "$PY" scripts/memorization_check_scene.py --ckpt "$gen_ckpt" --cache "$real" \
    --n 32 --steps "$steps" --eta "$eta" --weights "$weights" --out "$mem" >> "$LOG" 2>&1 \
    || say "memorization check failed (see log); study continues"
tf=samples/carrada_temporal_n${N}.json
[ -f "$tf" ] || "$PY" scripts/temporal_fidelity.py --real "$real" --generated "$reallab" --out "$tf" >> "$LOG" 2>&1 \
    || say "temporal fidelity failed (see log); study continues"

for det in frame sequence; do
  out=samples/carrada_study_n${N}_${det}.json
  arms=real,sim_mix,dit_mix,ditrl_mix,sim_pt,dit_pt,sim_only,dit_only
  [ "$det" = sequence ] && arms=real,sim_mix,dit_mix
  if ! grep -q '"complete": true' "$out" 2>/dev/null; then
    say "detector study ($det)"
    "$PY" scripts/radial_detector_study.py --variant engineer --real "$real" \
        --sim-cache data/carrada_sim_engineer --dit "$simlab" --dit-rl "$reallab" \
        --arms "$arms" --detector "$det" --val-spread \
        --dets-dir samples/carrada_dets --out "$out" >> "$LOG" 2>&1
  fi
  "$PY" scripts/step6_score.py --report "$out" --dets-dir samples/carrada_dets --n-doppler 64 \
      --out "samples/carrada_verdict_n${N}_${det}.json" >> "$LOG" 2>&1
  say "verdicts ($det) -> samples/carrada_verdict_n${N}_${det}.json"
done
say "DONE"
