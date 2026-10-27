# CARRADA: can the generator make real-looking maps here? (pre-registered 2026-10-26)

Ari, 2026-10-26: "yes" to the proposal after the simulator gate passed
(sim-only AP 0.299 on real val, real-only reference 0.560;
samples/carrada_gate.json). The RADIal finding to test: the DiT matches the
simulator's texture but makes 30-75% too few clutter peaks on real data, whatever
it starts from (docs/notes/2026-10-24-step9-generator-realism.md).

## Runs (scripts/run_carrada_gen.sh)
For each patch size, **16x16 first (as on RADIal), then 8x8** (a suspected cause
on RADIal, affordable on the 256 x 64 grid):
1. Pretrain the DiT on 6,000 engineer simulator sequences, 60k steps, the RADIal
   recipe (dim 384, 12 blocks, factorized attention, v-prediction, EMA 0.9999),
   schedule shift 8 (the resolution rule sqrt(pixels / 64x64) x 4 that gives
   4 at 64x64 and ~16-23 at 512x256).
2. Realism against simulator val, snapshots at 30k and 60k, raw and EMA weights.
3. Fine-tune the chosen snapshot on all 1,005 real CARRADA training sequences,
   9,600 steps (the RADIal recipe), a snapshot every ~1,000.
4. Realism against real val; in-sample against real training sequences.

## Measurement (fixed before any generated map exists)
Speckle, frame change, bright tail, clutter peaks (src/eval/scene_realism.py).
CARRADA has no DDMA copies, so only a vehicle's own cell is excluded from the
clutter count. Peak threshold = 2.7 std of the reference's own training maps,
as RADIal's 15 dB is for its real maps: real CARRADA 2.7 x 1.84 = 5.0 dB (~49
peaks per frame on real val), CARRADA simulator 2.7 x 2.84 = 7.7 dB. Val halves:
64 evenly spaced val sequences, alternating into selection (A) and gate (B), so
both halves span all six val recordings. Gate: every statistic within 25% on B.

## What each outcome means
- Both patch sizes pass on real data: the RADIal failure is RADIal-specific
  (dense street clutter, its grid), not a property of the generator.
- Only 8x8 passes: patch size is the cause; RADIal is retried with 8x8.
- Both fail: a generator problem in general; a different approach is needed.

## Results (2026-10-27)
**Both patch sizes pass on held-out real data** -> by the rule above, the RADIal
failure is RADIal-specific, not a property of the generator.

Relative error on gate half B (limit 25%):

| | speckle | frame change | brightest | clutter peaks |
|---|---|---|---|---|
| 16x16, simulator gate (30k raw) | 6% | 7% | 0.2% | 2% |
| 16x16, real gate (ft epoch 32, EMA) | 2% | 11% | 0.3% | 6% |
| 8x8, simulator gate (30k raw) | 7% | 8% | 3% | 2% |
| 8x8, real gate (ft epoch 72, raw) | 1% | 9% | 2% | 8% |

Fine-tuning does the work: the pretrained model alone is 54-69% off on real
data (selection-half mean). Unlike RADIal, realism did not decay over
fine-tuning (16x16: every epoch 16-72 within ~14% mean). Frame change is the
largest residual (9-14%) at every epoch.

Not done: the in-sample check against real training sequences (step 4). Realism
is not usefulness: whether the maps help a detector is the detector study.
