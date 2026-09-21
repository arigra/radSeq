# radSeq — Radar Sequence Generator

This research is about generating radar RD sequences. as in previous research we have seen diffusion model generate impresive RD maps, we decided to try making a sequences generator, making RD sequences closer to reality (relative to simulators)

If this generator will produce the sequences properly, and we could fine-tune it on a small specific dataset, this can help tramendously in radar neural network training.

**Read first:** `notebooks/radSeq.ipynb` — the whole project as notes, with every comparison.

## Map

| folder | what's in it | status |
|---|---|---|
| `notebooks/` | `radSeq.ipynb` | current |
| `src/` | all library code (below) | current |
| `scripts/` | step 3 pipeline: RADIal scene simulator → caches → DiT pretraining | **current** |
| `configs/` | configs for step 3 + `base.yaml` (defaults, used by tests) | current |
| `samples/` | step 3 results | current |
| `logs/` | step 3 run logs (not in git) | current |
| `docs/notes/` | dated lab notes, one per experiment, with pre-set rules and verdicts | reference |
| `experiments/dit_64/` | the 64×64 DiT: training pipelines, scoring, diagnostics, results | finished, runnable |
| `experiments/step2_detector/` | step 2: does generated data help a detector | finished, runnable |
| `experiments/radial_facts/` | how RADIal's grid and mislabelled label columns were established | finished |
| `archive/` | dead ends and superseded work — see `archive/INDEX.md` | record only |
| `tests/` | `python -m pytest tests/ -q` | current |
| `checkpoints/`, `data/` | model weights and datasets (not in git) — inventory in `archive/INDEX.md` | — |

## src/

| module | what |
|---|---|
| `simulator.py` | the 64×64 simulator (default behaviour pinned bit-for-bit by a test); also RADIal-grid options used by the first spec simulator |
| `scene_sim.py` | **RADIal scene simulator**: DDMA replication, static world from a moving car, receiver filters, antenna patterns; engineer's and fitted variants |
| `radar_physics.py` | published RADIal radar spec, radar equation, window |
| `radial.py` | RADIal access: labels, split, sequences, power maps, camera frames |
| `scene_data.py` | memory-mapped 512×256 scene caches |
| `dataset.py` | 64×64 caches |
| `dit.py`, `patching.py`, `diffusion.py`, `ema.py`, `losses.py` | the DiT and diffusion |
| `train.py`, `sample.py` | training and sampling (both grids) |
| `trajectory_condition.py` | target / vehicle conditioning maps |
| `detector.py`, `eval/` | detectors and metrics (four checks, adherence, CFAR) |
| `viz.py` | all plotting used by the notebook |
| `conditioning.py`, `research_losses.py`, `unet.py` | optional hooks in `train.py` (phase-3 encoder, extra losses, U-Net ablation); unused by the current recipe |

## Current pipeline (step 3, RADIal)

Run from the repo root.

```bash
python scripts/fit_scene_simulator.py      # fitted variant -> configs/scene_fitted.json
bash scripts/run_scene_caches.sh           # 8-frame 512x256 caches, both variants
bash scripts/run_scene_smoke.sh            # recipe check -> samples/scene_dit_smoke.json
bash scripts/run_pretrain.sh               # DiT pretraining on both variants
```

Long runs must go through the batch system, or they die when the interactive
session ends. From the login host `ece-hpc`:

```bash
sbatch scripts/pretrain.sbatch             # resumes from the last checkpoint; safe to resubmit
```

## Finished experiments

- **64×64 DiT** — `experiments/dit_64/`; notes `docs/notes/2026-09-14-*`, `2026-09-15-*`,
  `2026-09-17-trajectory-conditioning.md`. Checkpoints: `e0_standard_bs32` (easy),
  `e3_long_bs32` (full, passes the four checks), `cond_traj_bs32` (trajectory-conditioned).
- **Step 2, generated data for a detector** — `experiments/step2_detector/`; note
  `docs/notes/2026-09-17-step2-synthetic-augmentation.md`. Checkpoint: `cond_traj_n2000_bs32`.

## Environment

Python 3.10+, CUDA PyTorch: `python -m pip install -r requirements.txt`, then `python -m pytest tests/ -q`.
