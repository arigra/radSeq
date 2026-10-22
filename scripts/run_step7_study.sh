#!/usr/bin/env bash
# Step 7 detector study for one fold, called by run_step7.sh once the fold's
# generator has passed the realism gate. Args: fold tag cache finetune-dir sampler-json.
# Resumable: generation skips finished parts, the study keeps scored seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
k=$1 tag=$2 cache=$3 ft=$4 sel=$5
PY="${PY:-$(command -v python3)}"
LOG=logs/step7.log
N=50
export PYTHONPATH=.
say() { echo "$(date '+%F %T') | $*" >> "$LOG"; }

read -r steps eta weights < <("$PY" -c "import json; c=json.load(open('$sel'))['chosen']; print(c['steps'], c['eta'], c['weights'])")
sampler=(--steps "$steps" --eta "$eta" --weights "$weights" --guidance 1)
say "$tag: sampler ddim${steps} eta${eta} ${weights}"

# the simulator's own scenes (new layouts; the simulator arm uses exactly these)
# and the real training labels
simlab="data/step7_synth_${tag}_simlabels"
reallab="data/step7_synth_${tag}_reallabels"
say "$tag: rendering the simulator's scenes"
"$PY" scripts/gen_scene_synthetic.py --ckpt "$ft/last.pt" --cache data/scene_engineer \
    --split train --limit $((4 * N)) --repeats 1 "${sampler[@]}" --out "$simlab"
say "$tag: rendering the real labels"
"$PY" scripts/gen_scene_synthetic.py --ckpt "$ft/last.pt" --cache "$cache" \
    --repeats 4 "${sampler[@]}" --out "$reallab"

mem="samples/step7_memorization_${tag}.json"
[ -f "$mem" ] || "$PY" scripts/memorization_check_scene.py --ckpt "$ft/last.pt" --cache "$cache" \
    --n 32 --steps "$steps" --eta "$eta" --weights "$weights" --out "$mem"
tf="samples/step7_temporal_${tag}.json"
[ -f "$tf" ] || "$PY" scripts/temporal_fidelity.py --real "$cache" --generated "$reallab" --out "$tf"

for det in frame sequence; do
  out="samples/step7_study_${tag}_${det}.json"
  arms=real,sim_mix,dit_mix,ditrl_mix,sim_pt,dit_pt,sim_only,dit_only
  [ "$det" = sequence ] && arms=real,sim_mix,dit_mix
  if ! grep -q '"complete": true' "$out" 2>/dev/null; then
    say "$tag: detector study ($det)"
    "$PY" scripts/radial_detector_study.py --variant engineer --real "$cache" \
        --dit "$simlab" --dit-rl "$reallab" --arms "$arms" --detector "$det" \
        --dets-dir samples/step7_dets --out "$out"
  fi
  "$PY" scripts/step6_score.py --report "$out" --dets-dir samples/step7_dets \
      --out "samples/step7_verdict_${tag}_${det}.json"
  say "$tag: verdicts ($det) -> samples/step7_verdict_${tag}_${det}.json"
done
