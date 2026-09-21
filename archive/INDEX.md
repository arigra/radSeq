# Archive index

Dead ends and superseded work, kept as a record. Nothing here is needed by the
current pipeline, and scripts here are not guaranteed to run from this location
(some import each other as `scripts.*`). Everything was moved with `git mv` on
2026-09-21; `git checkout before-cleanup` restores the old layout.

| folder | what it was | why it stopped | results |
|---|---|---|---|
| `codex_research/` | Codex-era screens (Sept 13–16): CFAR calibration, false-track rates, measured-RADIal temporal gate, tail risk; and Codex's self-contained teaching notebook `02_radseq_self_contained.ipynb` | exploratory screens for a paper idea; superseded by the working DiT and the RADIal plan | `results/`, notes `docs/notes/2026-09-13-*` |
| `ablations_aug/` | August ablations (smoothness weight, EMA, overlap, U-Net), phase 1–3 runs, attention control, wandb run | the original DiT did not generate properly; the fixes came from the diagnostics in `experiments/dit_64/` | `results/`, notes `docs/notes/2026-08-07-*`, `2026-09-05-*`..`2026-09-07-*` |
| `latent_generators/` | `easy_latent.py`, `hard_latent.py`: learned trajectory/scene models whose maps were rendered by the simulator | a stop-gap while the pixel DiT was broken; replaced by the working DiT | checkpoints `easy_latent.pt`, `hard_latent.pt` |
| `fidelity_pilot_64/` | sim-to-sim rehearsal of the fidelity study on the 64×64 simulator | the 64×64 simulator does not resemble real radar; the study moves to the scene simulator. First run invalid (EMA bug, see `pilot_emabug/`) | `results/` (only the corrected d0 point finished) |
| `first_radial_simulator/` | the 64×64 simulator moved onto RADIal's grid: first fitted to RADIal (512-bin range gain), then specs-only | fitting laundered real data into the baseline; the specs-only build looked nothing like RADIal (no DDMA, 1/R³ clutter bug). Replaced by `src/scene_sim.py` | `results/` |
| `notebook_tools/` | one-off scripts that appended and executed notebook cells | the notebook is now edited directly | — |

## Checkpoints and data inventory (not in git)

Sizes as of 2026-09-21. **Nothing has been deleted** — freeing space is Ari's call.

### In use or cited — keep

| path | size | what |
|---|---|---|
| `checkpoints/e0_standard_bs32` | 8.1G | 64×64 DiT, easy regime (notebook section 3) |
| `checkpoints/e3_long_bs32` | 6.8G | 64×64 DiT, full regime, passes the four checks (notebook) |
| `checkpoints/cond_traj_bs32` | 4.1G | trajectory-conditioned DiT |
| `checkpoints/cond_traj_n2000_bs32` | 4.1G | step 2 small-data generator |
| `checkpoints/pretrain_engineer`, `pretrain_fitted` | 2.8G each | step 3 pretraining (running) |
| `checkpoints/scene_smoke_A_long` | 2.8G | winning 512×256 recipe run |
| `data/cache`, `data/cache_easy` | 5.5G each | 64×64 datasets |
| `data/scene_engineer`, `data/scene_fitted` | 13G each | step 3 caches |
| `data/radial` | 12K | pinned RADIal split |

### Candidates to free (~76 GB)

| path | size | what |
|---|---|---|
| `archive/fidelity_pilot_64/pilot_emabug/pilot_ft_d*` | 26G | fine-tuned checkpoints of the **invalid** pilot run (EMA bug); its logs and JSONs already document the bug |
| `checkpoints/e3_standard_bs32` | 8.1G | shorter run superseded by `e3_long` |
| `checkpoints/e3_full_bs32` | 5.9G | full-attention variant, worse than factorized |
| `checkpoints/pilot_*` (6 dirs) | 14.4G | dropped 64×64 pilot |
| `data/pilot` | 7.1G | dropped 64×64 pilot caches |
| `checkpoints/scene_smoke_A_p16_s16`, `B_p16_s4`, `C_p32_s16`, `B_long` | 7.0G | recipe-check runs that lost |
| `checkpoints/abl_*`, `ctrl_*`, `obj_*`, `phase*`, `e0_fact_s0`, `e0_temp_s0`, `easy_regime`, `research_easy_*` | ~9G | August/September ablations and screens |
| `data/measured` | 4.5G | Codex-era measured-RADIal power cache |
| `data/synthetic` | 1.0G | step 2 synthetic sequences (regenerable from `cond_traj_n2000_bs32`) |
