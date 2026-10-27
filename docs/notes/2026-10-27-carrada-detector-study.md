# CARRADA detector study: with 50 real sequences, does DiT data beat simulator data?

Decided by Ari 2026-10-27 (option 1 after the generator study,
docs/notes/2026-10-26-carrada-generator.md). Written before any result.

## Setup (scripts/run_carrada_detector.sh)
- Real data: 50 of CARRADA's 1,005 training sequences (fixed random subset,
  15 of 18 training recordings; data/carrada_seq_n50).
- Generator: the patch-16 DiT pretrained on the simulator (epoch 40, raw, the
  simulator-gate choice), fine-tuned on those 50 sequences for 384 epochs
  (~2,400 steps), a snapshot every 32 epochs. The snapshot and weights are
  chosen by realism on val half A and gated on half B, as before. **The detector
  study runs on the chosen snapshot whether or not it passes the gate**; the gate
  result is reported beside it.
- Extra data, 4 per real sequence (200): the simulator's own scenes (`sim`), the
  DiT rendering those same scenes (`dit`), and the DiT rendering the 50 real
  label sets 4 times (`dit_rl`, secondary).
- Detector study = RADIal step 6 (scripts/radial_detector_study.py): arms real,
  sim_mix, dit_mix, ditrl_mix, sim_pt, dit_pt, sim_only, dit_only; 5 seeds for
  primary arms, 3 for the rest; early stopping on real val; then the 8-frame
  sequence detector on real, sim_mix, dit_mix. Class-agnostic AP within 2 bins.
- Changed from RADIal: early stopping used the first 32 val sequences, which on
  CARRADA are all one recording (on RADIal mostly two). Here they are evenly
  spaced over all six val recordings (`--val-spread`).
- Test: CARRADA's official test split, 264 sequences from **6 recordings**.
  The bootstrap resamples recordings, so with 6 the intervals will be wide; a
  difference that does not resolve may only mean too few test recordings.

## What each outcome means (primary: dit_mix vs sim_mix)
- dit_mix > sim_mix: with 50 real sequences, generator data helps a detector
  more than simulator data -- the project's claim, on one dataset.
- not resolved: no detectable difference at this test size.
- dit_mix < sim_mix: generator data is worse than the simulator it learned from.
- sim_mix and dit_mix vs real say whether any extra data helps at 50 sequences.
- If the generator fails its gate but dit_mix still wins (or loses), the
  4-statistic gate is not predicting usefulness -- itself a finding.

## Results
(appended after the runs)
