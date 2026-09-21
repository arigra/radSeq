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

## Measured RADIal grid and statistics (2026-09-21)

### Grid, established empirically (scripts/radial_calibrate.py)
`radar_FFT/fft_NNNNNN.npy` is (512, 256, 16) complex64 = (range, Doppler, Rx).
- **axis0 = range at exactly 0.2 m/bin** (512 bins -> 102.4 m). Confirmed by
  regressing argmax against `radar_R_m`.
- **axis1 = Doppler in unshifted FFT order** (bin 0 = zero relative velocity,
  bins 128-255 = negative).
- **`labels.csv` column `radar_D_mps` is an integer Doppler BIN INDEX, not m/s**,
  despite its name. Likewise `radar_P_db` holds linear power (~1.4e7), not dB.
  Both column names are wrong; trusting them would have silently corrupted the fit.
- Velocity per bin is NOT recoverable from the labels (they are bins). Published
  spec is ~0.1 m/s per bin; TO VERIFY before it is used for kinematics.
- Sanity check: labelled vehicles sit a median 24.5 dB above their range ring
  (p10 18.1, p90 31.7) at the predicted cell, over 64 vehicles.

### Label distribution (10,661 vehicles), which justifies any crop
- range: p50 44.3 m, p90 66.9 m, p99 78.4 m, max 90.8 m -> the range axis is
  genuinely used; cropping range below ~450 bins discards real targets.
- Doppler: 81.4 % of vehicles within +/-32 bins of zero, 93.6 % within +/-64.
- some rows carry radar_R_m = -1 (invalid) and must be filtered.

### Simulator vs RADIal, measured in the same terms (scripts/radial_stats.py)
| quantity | our simulator | RADIal | verdict |
|---|---|---|---|
| range bin | 3.0 m | 0.2 m | 15x mismatch |
| grid | 64 x 64 | 512 x 256 | mismatch |
| background flatness in Doppler | 0 bins >3 dB | 0 bins >3 dB | **already matches** |
| background spread over range | 5.7 dB | ~6 dB | **already matches** |
| dynamic range (min..max) | 134 dB | 76 dB | ours ~1.8x too wide |
| target peak above range ring | 38.7 dB | 24.5 dB | **ours ~14 dB too bright** |

The background statistics need no structural change: RADIal has no zero-Doppler
clutter ridge (ego motion smears static clutter across Doppler) and our AR(1)
clutter already produces a comparably flat background.

The two real gaps are geometric (grid) and the target contrast: **our targets are
~14 dB easier to see than real vehicles**. That single number is the clearest
quantification of the sim-to-real gap so far, and it is a CNR/SCNR fit parameter.
