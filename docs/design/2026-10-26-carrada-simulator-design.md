# CARRADA simulator, engineer's variant (draft for Ari, 2026-10-26)

Built like the RADIal scene simulator: from published radar parameters and the
dataset paper's description of the scenes; **nothing fitted to the recordings**
(Ari's rule, 2026-09-21). Facts in docs/notes/2026-10-26-carrada-raddet-facts.md.
Code: a new `src/carrada_sim.py`, reusing `scene_sim.py`'s renderer pieces and the
self-shadowing car (docs/design/2026-10-22-vehicle-scattering-design.md).

## Radar (published)
- 77 GHz FMCW, TDM MIMO 2 Tx x 4 Rx -> 8 virtual channels, no DDMA copies.
- Grid as the dataset: 256 range bins of 0.2 m (row 0 near), 64 Doppler bins of
  0.42 m/s, zero at bin 32, velocities beyond +/-13.4 m/s alias (wrap).
- RD map = power summed over the 8 channels (the "aggregated view"), in dB.
- Link budget: Tx 12 dBm, receiver noise figure 14-15 dB (AWR1843 datasheet).
- 10 Hz frames, 8-frame sequences (0.8 s).

## Scene (from the paper's description)
- Static radar on a parked car; flat asphalt test track: diffuse ground return
  (the RADIal ground model) and a few isolated poles; no buildings, no parked cars.
- One or two moving objects per sequence, on straight paths that approach,
  recede or cross, inside 50 m and the field of view; an object leaving the view
  leaves empty frames, as in the recordings.
- Car: the self-shadowing body outline, 4.5 x 1.8 m, 10 dBsm, 3-10 m/s.
- Pedestrian: torso plus four limb scatterers swinging at the gait frequency
  (~1.8 Hz), body 0.5-2 m/s, limbs to ~2x body speed (micro-Doppler).
- Cyclist: bicycle + rider, 3-7 m/s; wheel-spoke micro-Doppler to ~2x bike speed.
- Labels as CARRADA: per instance and frame, class, RD box around the object's
  return; centre on its nearest visible point; instance IDs across frames.

## Gate before any generator work
A detector trained on simulator data alone must reach AP >= 0.1 on CARRADA's
real val recordings (the RADIal car gate). Below: back to this spec.

## Then the RADIal experiments, same rules
5-fold cross-validation by recording (30 recordings, balanced by sequences);
DiT pretrained on the simulator; realism vs simulator; fine-tune on full real
data and on 50 sequences, realism chosen on val half A and gated on half B
(32 + 32 sequences); controlled detector study (per-frame and multi-frame,
bootstrap over recordings). The small grid also makes 8x8 patches affordable,
one of the suspected causes of the RADIal clutter gap.

## Decisions for Ari
- (a) **Pedestrian and cyclist RCS and micro-Doppler.** The values above are
  approximate, from memory of the literature (pedestrian about -10 to 0 dBsm,
  cyclist about -5 to +5). Look up published values first, or accept these?
- (b) **Near-range receiver high-pass.** Its corner depends on the chirp slope,
  which the paper does not give. Leave it out, or reuse RADIal's filter shape?
- (c) **Detector task.** Class-agnostic objects, as on RADIal (comparable), or
  pedestrian / cyclist / car (CARRADA's own task)? Recommendation: class-agnostic
  first, 3-class as a second readout.
- (d) **Antenna gain.** No official figure for the board; TI's estimator default
  is 8 dBi. Accept, or leave the absolute level free (only relative levels then
  matter; each source is normalised by its own statistics anyway)?

## Decisions taken (Ari, 2026-10-26: "do the recommendations")
- (a) Looked up first. **Pedestrian:** RCS -10 to +5 dBsm at 77 GHz over a walking
  stride (co-pol; ray tracing validated against 77 GHz measurements), torso
  ~1.5 m/s, arms and legs up to 5 m/s, a range spread of ~1.5 m about the torso
  (Deep et al., arXiv 1910.13706). Modelled: total RCS drawn uniform in that
  range per frame, torso + four limbs, limb speed swinging to ~3.3x the body's.
  **Cyclist:** no published dBsm value found (the 77 GHz measurements behind
  MathWorks' backscatterBicyclist, Stolz et al. EURAD 2017, are not public).
  Assumption, flagged: rider = the pedestrian model; bicycle frame as extra
  scatterers at the rider's level; wheels from kinematics alone (a rolling rim
  runs from 0 to 2x the bike's speed); pedal cadence = wheel rate / 1.5 (the
  model's default gear ratio).
- (b) Near-range high-pass: **left out** (its corner would have to be guessed;
  the near-range dip stays a sim-to-real gap).
- (c) Detector: **class-agnostic first**, three classes as a second readout.
- (d) Antenna gain: **8 dBi Tx and Rx** (TI's estimator default, flagged).

## Conventions measured (docs/notes/2026-10-26-carrada-raddet-facts.md)
Approaching objects sit **above** bin 32. For slow objects the range change per
frame per Doppler bin is -0.206 rows (the published grid predicts -0.21), which
confirms 0.2 m, 0.42 m/s and 10 Hz together. Labels: the centre of the object's
RD box (real) / the power-weighted centre of its visible returns (simulated).
Empty frames: 43% of frames carry no object (paper Table II: 7,193 of 12,666
with instances), so 43% of simulated sequences start with none; otherwise one
or two objects, equally likely (paper: "one or two").
