# Invalid results

`step4_engineer_no_positive_targets.json` (2026-09-23): every detector arm scored
AP 0.000, including real data only. Cause: the training heatmaps centred each
Gaussian on the fractional label position, so no cell ever equalled 1.0, and
the CenterNet focal loss counts only cells equal to 1.0 as vehicles -- 596
vehicles produced a single positive cell. Fixed in `src/detector_scene.py`
(centres rounded to a cell); a detector trained on real data then reached
AP 0.868 after 600 steps.

`step4_engineer_sim_wrong_normalisation.json` (2026-10-19): the `real_sim` arm
(0.769) is invalid; `real` (0.828) and `real_dit` (0.768) were computed
correctly. Cause: `scripts/radial_detector_arms.py` normalised every pool by the
real data's statistics (mean 97.0 dB, std 5.6), including the simulated frames,
which sit near 12.5 dB -- so 80% of that arm's training frames were ~15 std off
the real data. Fixed by normalising each source by its own statistics, as DiT
training does.
