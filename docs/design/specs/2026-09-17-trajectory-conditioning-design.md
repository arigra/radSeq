# Trajectory-conditioned DiT — design (2026-09-17)

Step 1 of the research plan: make the working DiT generate the targets it is
asked for, so every synthetic sequence comes with exact target labels (needed
by step 2, detector training) in the form RADIal provides (step 3).

Decisions: Ari chose *target trajectories* as the control and *match the
simulator* as the pass bar, and delegated the rest ("go on with what you
think"); sections 1-3 below were not individually reviewed.

## 1. What is controlled

Per target: range and Doppler bin in each of the 16 frames, and class
(0 steady, 1 Swerling-1, 2 extended). Target count follows from the list
(1-5). Not controlled: target brightness (not stored in the cache), clutter
and noise (not labelled in real radar); the model invents them.

## 2. Model

- **Condition map** (`src/trajectory_condition.py`): `render_condition(traj,
  n_targets, cls)` -> `(B, L, 4, 64, 64)`. Channels 0-2: a Gaussian blob
  (sigma = 1 bin, peak 1, max-combined) at each target's sub-bin position, one
  channel per class. Channel 3: all ones when a condition is present.
  `drop_condition(cond, p)` zeroes all 4 channels for a random fraction p of
  sequences (classifier-free guidance training).
- **Input** (`src/dit.py`): `TemporalDiT(cond_channels=4)` (config `model.cond_channels: 4`)
  patchifies the noisy map and the condition channels together, so each 8x8
  patch token carries `64 * (1 + 4)` values. Only the input projection grows;
  factorized attention, depth 12, dim 384 are unchanged.
- **Warm start** (`train.init_from`): load the passing unconditional
  checkpoint (`checkpoints/e3_long_bs32/last.pt`, EMA weights), copy its input
  projection into the map columns and zero the condition columns. At step 0 the
  conditional model's output equals the unconditional model's for any
  condition.
- **Guidance at sampling**: `pred = uncond + w * (cond - uncond)` on the
  network output (v-prediction), with `uncond` using the dropped condition.

## 3. Training

Same data (`data/cache`, 20k train / 2k val) and recipe as `experiments/dit_64/configs/e3_long.yaml`
(v-prediction, schedule shift 4, no clamp, bf16, batch 32, lr 1e-4, EMA 0.9999,
no smoothness loss), plus `model.cond_channels: 4`,
`train.cond_dropout: 0.1`, `train.init_from`. Fresh optimizer. 40 epochs
(25,000 steps, ~2 h). Resumable pipeline `experiments/dit_64/run_cond_traj.sh` and
`experiments/dit_64/cond_traj.sbatch`, checkpoints every 1000 steps.

## 4. Evaluation and decision rule

`src/eval/adherence.py`: `trajectory_adherence(x_db, traj, n_targets,
radius=2.0)` using the evaluator's own detector and linker
(`src/eval/metrics`: peaks 12 dB over the frame median, top 5, 3-bin gate):

- **hit rate**: fraction of requested (target, frame) pairs with a detected
  peak within 2 bins (Euclidean, range/Doppler bins).
- **unrequested lasting tracks / sequence**: linked tracks of >= 8 frames
  whose peaks are within 2 bins of a requested target on fewer than half of
  their frames.

Requests are the labels of held-out validation sequences. The **simulator
reference** is the same measurement on those real sequences with their own
labels. Generation: EMA weights, DDIM 30.

- **Guidance sweep**: w in {1, 2, 3}, requests from val 0-95, sampling seeds
  1-3. Choose the w that passes the rule with the highest hit rate (if none
  passes, the one with the highest hit rate).
- **Verdict**: chosen w, requests from val 96-287, seeds 4-9.
- **Passes** if all hold: hit rate >= simulator hit rate - 0.05; unrequested
  lasting tracks/seq <= simulator's; Ari's four checks (std within 0.1 of real,
  marginal L1 <= 0.15, target tracks/seq within 15% of real, persistence within
  25% of real) pass against held-out real data.
- **Null check**: the unconditional `e3_long` model on the same requests must
  score a hit rate at least 0.20 below the conditional model's, else the test
  is too easy and no result is reported.

Not measured: whether generated targets look like their requested class.

## 5. Code units and tests

| unit | test |
|---|---|
| `render_condition`, `drop_condition` | blob maximum at requested bin; class routing; padded targets ignored; presence channel; dropout zeroes whole sequences |
| `TemporalDiT(cond_channels)`, input-projection expansion | expanded model output equals unconditional output exactly |
| `train` conditioning + `init_from` | tiny run from a tiny unconditional checkpoint trains; config recorded |
| `generate_trajectory_conditioned` | shape and finiteness; w=0 equals the dropped-condition prediction |
| `trajectory_adherence` | simulator sequences with own labels score high; labels shifted by 10 bins score low |
| scorer + pipeline | smoke test (train, resume, score) before launch |

The decision rule is copied into `docs/notes/2026-09-17-trajectory-conditioning.md`
before the run, and the result is appended there.
