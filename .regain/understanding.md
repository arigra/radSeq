# radSeq — AI's understanding (phase 1, pass 1, 2026-09-26)

For the AI, not the user. Every line is marked: ✓ verified against code/results, ~ inferred, ? unknown.

## Coverage — what this pass actually read
- Read in full: README, notebook markdown (120 cells), docs/prop.md, 2026-09-21 step-3 design note (first 60 lines), src/scene_sim.py, src/dit.py, src/patching.py, src/scene_data.py, src/train.py (loop, first ~330 lines), scripts/build_scene_cache.py, scripts/*.sh, *.sbatch, configs/pretrain_*.yaml, configs/base.yaml.
- Results read: samples/scene_dit_smoke.json, samples/scene_sim_fit.json, experiments/step2_detector/results/*.json, experiments/dit_64/results/e3_long_scores.json (head).
- NOT read: src/diffusion.py, src/sample.py, src/trajectory_condition.py, src/radial.py body, src/eval/*, src/simulator.py, src/detector.py, scripts/fit_scene_simulator.py body, experiments/* code, tests, docs/notes except one, archive/.
- Runs locally (✓ 2026-09-28): conda envs `3dc` (torch 2.3) and base (torch 2.1) run the simulator, a tiny scene cache, the DiT forward pass and one training step on CPU; see `.regain/radseq.regain.md`. (Pass 1 said "no torch": wrong, it only checked the system python3.)
- Not local: RADIal data, full caches, checkpoints and logs. Those are on HPC (`ece-hpc`, `/truenas/home/arigra/permuter/ariGranevich/radSeq`).

## User
- Relation (Ari, 2026-09-26): built it with agents, understands the direction but not what is inside.
- Confirmed as the essence: "DiT generating RD sequences; tests whether sim-pretrained + RADIal-fine-tuned is a better detector-training source than sim data."

## 1. Purpose (✓ essence confirmed by Ari; which form of the question is still open)
Can a DiT pretrained on a radar simulator and then fine-tuned on scarce real data (RADIal) provide better extra training data for a detector than the simulator itself, and does that make simulator fidelity matter less?
- The question has two forms in the docs. The design note says "claim: D > B". Notebook §5 says "interaction: B drops with worse sim, D flat". ? Which one is Ari's?

## 2. Live / dormant / dead
**Live, step 3 (RADIal, 512×256):**
`scripts/fit_scene_simulator.py` → `configs/scene_fitted.json`
`scripts/build_scene_cache.py` (via `run_scene_caches.sh`, `scene_pipeline.sbatch`) → `data/scene_{engineer,fitted}/`
`scripts/scene_dit_smoke.py` (recipe check, finished) → `samples/scene_dit_smoke.json`
`python -m src.train --config configs/pretrain_*.yaml` (via `run_pretrain.sh`, `pretrain.sbatch`)
src used: scene_sim, radar_physics, radial, scene_data, train, dit, patching, diffusion, ema, losses, trajectory_condition, sample.

**Finished, runnable (64×64):** experiments/dit_64, experiments/step2_detector, experiments/radial_facts; src/simulator.py, dataset.py, detector.py, eval/*; scripts/gen_synthetic.py, memorization_check.py.

**Dormant:** code paths that exist but the current recipe switches off:
- `conditioning.py` (phase 3), `research_losses.py`, `unet.py`
- `TemporalBlock` and `FullBlock` in dit.py
- `hann` and `tile` unpatchify modes
- trajectory/Doppler losses (phase ≥ 2)
- `smooth_loss`: computed every step but multiplied by `lambda_smooth: 0`, so logged only.

**Missing for the next step (✓ grep):** there is no code for fine-tuning on RADIal, no training cache of real RADIal sequences, and no detector or arms A/B/D on RADIal. The only trace is a legend label in `viz.py:407`.

## 3. Flow, step 3 (✓ from code)
1. RADIal `radar_FFT` (512,256,16) complex + labels.csv → `radial.py` → dB power maps. The split is by recording, pinned in `data/radial/split.json`.
2. `SceneSimulator(scenario).gen_sequence()` builds one sequence:
   - picks a road type (city / highway / countryside, equal odds) and an ego speed
   - builds static scatterers (curb, guardrail, facades, trees, poles, ground) plus moving vehicles (Poisson, mean 1.16) plus parked cars
   - for each of 8 frames (0.2 s apart): radar equation, elevation, azimuth cos², surface law sin^n → windowed 2D tones summed over 16 Rx (random phase + array phase) → DDMA copies rolled into 12 slots (offsets 0, 80..240) → + thermal noise, receiver high-pass/low-pass, ADC noise → dB
   - output: x (8,512,256) dB. **Labels = moving vehicles only; parked cars are scene** (✓ `scene_sim.py:346`).
3. `build_scene_cache`: 6000 train / 300 val per variant, float16 memmap. Mean/std come from every 12th training sequence. Each sequence has its own seed, so the cache is reproducible.
4. `SceneSequenceDataset` → x normalised (8,512,256); traj (8 vehicles, 8 frames, 2); present (8,8).
5. `train`:
   - condition: `render_vehicle_condition` → cond_map (B,8,2,512,256), dropped for 10% of samples
   - model: patchify 16×16 stride 16 → 32×16 = 512 patches per frame, 4096 tokens per sequence, token = 256×3 values → Linear to 384 → 12 FactorizedBlocks (attention over the 8 frames at each patch; attention over the 512 patches in each frame; MLP; adaLN-Zero on t, 9·d outputs) → unpatchify
   - objective: predict v; cosine schedule with shift 16, T = 1000; MSE on v
   - optimisation: AdamW 1e-4, batch 8, 750 steps/epoch × 80 = 60k steps, bf16, EMA 0.9999 (initial weights left in the EMA: e^-6 ≈ 0.25%, fine)
   - ckpt every 1000 steps; validation every 5 epochs
6. Sampling: DDIM 30 steps, EMA weights (~ from config; sample.py not read).

## 4. Claims vs evidence
| claim (where) | evidence | status |
|---|---|---|
| shift 16 beat 4 on dynamic range, 50 vs 42 dB, sim 57 (pretrain yaml) | scene_dit_smoke.json A_long 50.0, B_long 42.06, sim 57.2 | ✓ |
| "showed DDMA copies, arcs, empty-slot band" (pretrain yaml) | figures not in git | ? |
| recipe = 45M params, 8×8 patches, factorized (notebook §3) | e3_long config dim 384, depth 12, factorized; 25·d²·12 ≈ 44M | ✓ |
| generated data +0.040 mAP (notebook §4) | real_n 0.665, real_n_synth 0.705 | ✓ |
| memorisation ratio 0.715 | memorization_n2000.json | ✓ |
| fitted distance 0.32, engineer 1.14 | distance_fresh 0.324, engineer 1.136 | ✓ |
| "scene sim matches real range structure" (notebook §2.4) | far range vs mid: real −16.6, engineer −5.1, fitted −9.9 dB; near: real −6.6, engineer −13.0 | ✗ overstated |
| status "recipe check running" (notebook §6, README table) | finished 21/9 21:29; pretraining started 21:39 | ✗ stale |
| dit.py docstring "Temporal-only… per the proposal" | recipe uses `attn_mode: factorized` | ✗ stale |
| FactorizedBlock = "control variant for the Phase-1 diagnosis" | it became the main recipe | ✗ stale |

Pattern: numbers match their result files. Status and structure descriptions lag behind the code.

## 5. Decisions (who decided: ? for all; 103/112 commits carry an agent co-author trailer, and the other 9 read as agent-written too)
1. Proposal's temporal-only attention → factorized time + space (Phase-1 diagnosis, ~early Sept).
2. Proposal's overlapping patches → non-overlapping (stride = patch).
3. Proposal's smoothness/physics losses → switched off (λ = 0).
4. ε-prediction → v-prediction; shift 4 at 64×64 → 16 at 512×256 (decided 10 min after the evidence came in).
5. 16 → 8 frames, because RADIal has only 219 independent 16-frame sequences.
6. Split by whole recording, pinned.
7. Fidelity levels = engineer vs fitted scene sims. Replaces the 64×64 dB-mismatch pilot, which was dropped after only its 0 dB arm finished.
8. Labels/conditioning = moving vehicles only, positions only, no classes.
9. After the memorisation flag in step 2, moved on to RADIal with fewer sequences (387 train); no mitigation in code.

## 6. Open state
- ? Did pretraining finish? It was started 21/9 21:39: 60k steps × 2 variants, run one after the other under 24 h sbatch limits (resumable). The logs are on HPC.
- The next step (fine-tune on RADIal + detector arms) has no code yet.
- Risk ~: memorisation at 387 real sequences (it was already flagged at 2000).
- Risk ~: RADIal labels vs sim labels. The sim labels only moving vehicles; if RADIal labels parked cars too, the condition maps disagree between pretraining and fine-tuning. The notebook states the assumption, but I did not check it against the labels.
- Stale docs (§4) mislead any agent that reads them first.

## Details that bite (ranked by likely damage; candidates from pass 1)
1. Sim labels = moving vehicles only (`scene_sim.py:346`). If RADIal labels also include parked or stopped cars, the condition maps mean different things in pretraining and in fine-tuning. Unchecked.
2. Normalisation is per cache (`stats.pt`: mean/std from that cache's own train split). Fine-tuning on real data will have to choose between the sim stats and the real stats. Either choice silently shifts the dB scale the model learned.
3. Engineer sim: far range is 11 dB too bright and near range 6.5 dB too dark relative to RADIal. The notebook says it "matches range structure".
4. RADIal `labels.csv`: `radar_D_mps` holds Doppler bins, and `radar_P_db` holds linear power, despite the names. Velocity per bin is marked "TO VERIFY" in the design note.
5. `smooth` appears in every training log line, but `lambda_smooth: 0` means it has no effect on training.
6. dit.py docstring and the FactorizedBlock comment describe a temporal-only model; the recipe is factorized.
7. EMA 0.9999: it was the cause of the invalid pilot. It is fine at 60k steps, but must not be used for short fine-tunes (train.py warns about this).
8. Fitted sim knob `surface_incidence_power` = 0.0 sits at its lower bound, so the fit is pinned by the search range there.

## Self-test (predictions made before checking)
| prediction | result |
|---|---|
| pretrain tokens per sequence = 8·32·16 = 4096 | ✓ num_patches formula |
| smoothness term has no effect on training | ✓ total = dit + 0·smooth |
| changing the schedule shift touches only the yaml | ✓ read through `cfg["diffusion"]` in train.py |
| fine-tune on RADIal exists somewhere | ✗ it does not; I had assumed it from the notebook's "next" |
| engineer sim roughly matches real range profile (from the notebook) | ✗ far range off by 11 dB |
