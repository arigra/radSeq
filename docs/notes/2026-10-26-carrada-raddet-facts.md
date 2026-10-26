# CARRADA and RADDet: facts for a second and third test bed (2026-10-26)

Ari, 2026-10-26: build a simulator for another dataset and repeat the RADIal
experiments, to see whether the same issues appear, new ones, or none. Both
datasets are already on disk (`radar data/Carrada`, `radar data/RADDet`); the
CARRADA layout is documented in `vis to radar/carrada_dataset_reference.md`.
Order agreed: CARRADA first, RADDet second.

## CARRADA (measured on the files unless marked)
| fact | value | how known |
|---|---|---|
| RD map | 256 range x 64 Doppler per frame; `range_doppler_raw` = linear power (aggregated view), `processed` ~ its dB (corr 0.995) | measured |
| range | 0.2 m/bin, 50 m; **row 0 = near** (objects 5.5 dB stronger in rows < 128; the near-range dip fits a receiver high-pass). Range-angle maps run the other way (same car: RD row 239, RA row 17). To confirm against camera images | measured, moderate confidence |
| Doppler | 0.42 m/s/bin, +/-13.4 m/s, **zero at bin 32** (centred) | paper Table I + measured ridge |
| static clutter | weak zero-Doppler ridge, +3.9 dB over the median | measured |
| frame rate | 10 Hz (ADC timestamps 0.1 s apart, 1,019 for 1,017 frames) | measured |
| Doppler sign | approaching objects above bin 32; slow objects move -0.206 rows per frame per Doppler bin (grid predicts -0.21) | measured on 4,140 frame pairs |
| radar | static, on a parked car, test track in Canada; 77 GHz, 2 Tx x 4 Rx (8 virtual), 64 chirps x 256 samples; chip AWR1843 per Ouaknine's thesis | paper, arXiv 2005.01456 / 2203.08038 |
| frames | 12,666 in 30 recordings; 7,193 (57%) annotated, in 53 consecutive runs = 875 non-overlapping 8-frame sequences | measured |
| unannotated frames | **empty of moving objects**: no peak > 9 dB off the ridge (annotated frames: median 12.1 dB) -> real negatives | measured on 120 + 120 frames |
| objects per frame | 0: 5,473 frames, 1: 5,636, 2: 1,557 | measured |
| labels | per instance (IDs across frames): RD and RA sparse points, box, dense mask; classes pedestrian / cyclist / car; the box sits on the object's peak (example: 9.5 dB above its row) | reference doc + measured |

## RADDet
| fact | value | how known |
|---|---|---|
| data | RAD tensors 256 x 256 x 64 complex64 per frame; RD = power summed over angle | measured |
| radar | **static**: strong zero-Doppler ridge in one bin, +14 to +20 dB over the median (streets: far more static clutter than CARRADA) | measured |
| grid | 0.195 m/bin (50 m), 0.42 m/s/bin, 64 Doppler bins | RADDet README |
| frames | 10,158; the official split interleaves train and test frame by frame (temporal leakage for a sequence study); merged they form **26 consecutive runs** holding 10,124 frames | measured |
| labels | 6 classes; 3D boxes [range, angle, Doppler centre + sizes]; no instance IDs (tracks must be linked, as for RADIal) | measured |
