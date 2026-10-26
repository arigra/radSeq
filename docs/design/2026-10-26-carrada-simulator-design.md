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
