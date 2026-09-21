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

## Fitted simulator (2026-09-21)

Fit on TRAINING recordings only (scripts/fit_simulator_to_radial.py); val/test
recordings are untouched so the "simulator" arm carries no real test data.

Three model deficiencies were found by measurement, not assumed:
1. **Single look.** RADIal's maps sum power over 16 receive channels, which
   suppresses speckle; our single-look maps spanned 131 dB against RADIal's 80.
   No clutter setting closes that -- only non-coherent integration does.
   Fitted `n_looks = 8` (the 16 channels are correlated, so they are not 16
   independent looks).
2. **No receiver range response.** RADIal has a near blind zone (bins 0-3,
   ~-21 dB), a peak near 12 m, a gentle ~5 dB falloff, and a sharp roll-off
   past ~96 m (~-18 dB). Our clutter was uniform in range. The response is
   *fitted from the training recordings* as a per-bin dB gain, transferring
   only the shape (both profiles are referenced to their own median).
3. **Targets far too bright.** The shipped target gain U(-5, 10) dB had to drop
   to **U(-35, -20) dB** -- a 25 dB correction. The original simulator made
   targets roughly 25 dB easier to see than real vehicles.

### Residuals of the fitted simulator (sim - real, dB)
| statistic | RADIal | fitted sim | residual |
|---|---|---|---|
| target prominence over range ring | 23.98 | 22.54 | -1.44 |
| dynamic range | 80.27 | 76.60 | -3.67 |
| Doppler profile spread | 2.54 | 2.32 | -0.22 |
| range profile spread | 24.26 | 25.34 | +1.08 |

Parameters in `configs/radial_sim.yaml`, range response in
`data/radial/range_gain_db.npy`, full sweep in `samples/simulator_fit_radial.json`.

This is deliberately a *good* baseline: a strawman simulator would make the
generator look good for the wrong reason. What it still cannot model is real
vehicle scattering (multiple scatterers, micro-Doppler, aspect dependence),
which is the residual the DiT is supposed to capture and what the
target-centred check will test.

## Split, pinned (data/radial/split.json)
Whole recordings, balanced by the sequences each contributes (recordings differ
~20x in length; assigning them independently left val with 6 of 219 sequences).

| split | recordings | frames | 16-frame sequences |
|---|---|---|---|
| train | 67 | 4544 | 153 |
| val | 9 | 1016 | 22 |
| test | 13 | 1627 | 44 |

**Only 219 independent 16-frame sequences exist in all of RADIal.** That is far
scarcer than the 2,000 used in the step 2 rehearsal, and it is the dominant
constraint on everything downstream: overlapping windows (stride 4) raise train
to 525 but they are near-duplicates, and 8-frame sequences would give 387.
