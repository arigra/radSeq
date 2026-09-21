# Step 3 design: fitted simulator vs fine-tuned DiT on RADIal (draft for Ari)

## Question
Given a simulator fitted to real data (imperfect) and a DiT pretrained on it then fine-tuned on
N real sequences: which is the better extra-data source for a detector trained with N real sequences?

## RADIal facts (measured from labels.csv, 2026-09-21)
- 8,252 labelled frames, 91 recordings, 10,661 vehicle boxes (median 1 per frame, max 9).
- Only labelled frames are on disk; per-frame radar_FFT is (512 range, 256 Doppler, 16 rx) complex64.
- Runs of >=16 *consecutive* labelled frames: 109 runs holding 5,115 frames.
  - non-overlapping 16-frame sequences: ~320; overlapping windows (stride 1): ~3,480 (highly correlated).
- Largest recordings: 432 / 427 / 336 frames; the longest single run is 288 frames.

## Consequence for sizes
Real sequences are far scarcer than in the simulator rehearsal (2,000 -> ~320 independent ones).
So "scarce" is the natural regime, and the test set must be few whole recordings.

## Split (proposal, by recording, fixed before any model is fitted)
- test: ~20% of usable recordings (whole recordings, never used by simulator fit, generator, detector selection)
- val: ~10% of recordings (early stopping / memorisation check)
- train: the rest; nested real subsets for the size curve (e.g. 10, 25, 50, 100 % of train sequences)
- report bootstrap CI over test recordings; no claim if arms differ by less than the CI.

## Arms (same real subset N, same test, 3 seeds)
A real only; B real + fitted-simulator data; C real + pretrained-not-fine-tuned DiT; D real + fine-tuned DiT.
Claim: D > B (and D > C shows fine-tuning matters).

## Open decisions for Ari
1. Detector task / conditioning: RADIal gives vehicle boxes + radar points (no target classes/trajectories).
2. Map format: 512x256 -> our 64x64 (crop/resize) and how RD map power is normalised.
3. What the simulator is fitted to (noise/clutter statistics, target RCS, speeds) and how.
