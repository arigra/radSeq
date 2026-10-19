# Invalid results

`step4_engineer_no_positive_targets.json` (2026-09-23): every detector arm scored
AP 0.000, including real data only. Cause: the training heatmaps centred each
Gaussian on the fractional label position, so no cell ever equalled 1.0, and
the CenterNet focal loss counts only cells equal to 1.0 as vehicles -- 596
vehicles produced a single positive cell. Fixed in `src/detector_scene.py`
(centres rounded to a cell); a detector trained on real data then reached
AP 0.868 after 600 steps.
