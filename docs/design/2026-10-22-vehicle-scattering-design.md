# Vehicle scattering in the scene simulator (draft for Ari, 2026-10-22)

## Problem
`src/scene_sim.py` `_car` draws every vehicle as five equally bright points:
the four corners of its footprint and the centre, label at the geometric centre.
Seen from the radar, the near corners, the centre and the far corners fall in
three range cells 2.25 m (11 bins) apart, so a simulated car is three one-bin
spikes with the label on the middle one. A detector trained on that finds no
real cars (step 5, sim-only AP 0.000-0.002). Measured at the label's Doppler,
dB above the range ring, offsets -12..+12 bins in steps of 2:

    real  0  0  0  0  1 16 19 14  8  4  3  3  3    one blob, tail behind
    sim  14  7  0  0  0  1 21  0  0  0  0  5 13    three spikes

What is physically wrong: (1) the far half of the car is visible, though the
body shadows it; (2) every point is equally bright whatever its orientation;
(3) the label sits inside the car, where nothing reflects.

## Proposed model (physics only; nothing fitted)
1. **Body as a box surface, not corners.** Scatterers along the four faces of
   the footprint at the existing density (`scatterers_per_m` = 4, finer than the
   0.2 m range bin), each carrying its face's outward normal. Sizes unchanged:
   car 4.5 x 1.8 m, truck 10 x 2.5, two-wheeler 2 x 0.8.
2. **Self-shadowing.** A scatterer contributes only if its face points at the
   radar (normal . line-of-sight > 0). Behind or ahead of the ego car that leaves
   the near end; at an oblique angle, the near end plus one side.
3. **Orientation-dependent strength.** Each visible scatterer's strength goes as
   cos^n of its incidence angle, the same surface law the simulator already uses
   for static structures (`surface_incidence_power`, n = 2 in the engineer's
   variant). A face seen head-on reflects strongly; a side seen at grazing angle
   barely does.
4. **Total RCS preserved.** Visible scatterers are scaled so the vehicle's total
   is the published value (car 10 dBsm, truck 20, two-wheeler 0), with the
   existing +/-3 dB per-scatterer scintillation.
5. **Label at the visible return.** Range = the nearest visible point of the
   body, i.e. where the radar sees the car; Doppler and azimuth unchanged.
6. Parked cars use the same model (they are scene, not labels).

Expected appearance, before any measurement: one compact return at the near
face, 1-3 range bins wide, with the label on it; a weaker streak along the
visible side for oblique cars.

## Decisions for Ari
- **(a) Label position.** Item 5 follows from physics, but it is also what the
  measured RADIal labels show (peak a median 1 bin from the label). Is matching
  the label convention acceptable? It is an annotation convention, like the
  mislabelled `radar_D_mps` column, not a radar parameter.
- **(b) The tail behind the peak.** Real cars have a weak tail 2-12 bins behind
  the peak (wheel housings, underbody, multipath under the car at 0.8 m radar
  height). Leave it out (the tail is then a sim-to-real gap the DiT has to
  learn), or add interior scatterers at a strength set *before* comparing with
  RADIal? I have already seen the real profile, so choosing that strength now
  would be partly fitting by eye. My recommendation: leave it out.
- **(c) The fitted variant.** Its knobs were tuned with the old car. Rerun
  `scripts/fit_scene_simulator.py` with the new car (cheap), or keep its knobs?
  Recommendation: refit, since the car is part of what it was fitted against.

## Check before spending GPU time (~1 h)
Small engineer cache (300 sequences) with the new car, then:
- the label-centred profile above, reported next to real (reported, not tuned);
- a detector trained on simulator data alone, scored on the real **val**
  recordings, not test, so the simulator design never touches the test set.
  Pass: sim-only AP >= 0.1 (it is 0.002 now). If it fails, back to this note.

## Then the redo (~16 h GPU, by sbatch)
1. Rebuild both scene caches (6,000 + 300 sequences each).
2. Re-pretrain both DiTs, 60k steps each (~6 h each).
3. Fine-tune each on the **50-sequence** real subset (Ari, 2026-10-22), 9,600
   steps; render 4 x 50 sequences.
4. Detector arms on real test: real (50), real + sim, real + DiT, sim only,
   DiT only; step 4 and step 5 rules; memorisation check.

## Outcome (2026-10-22)
Approved by Ari as drafted: (a) label at the nearest visible point, (b) no tail,
(c) refit the fitted variant. Implemented in `src/scene_sim.py` (`_car`,
`_car_returns`, `Scenario.vehicle_incidence_power` = 2, kept out of the fit,
which drives the rough-surface law to 0).

Label-centred profile, offsets -12..+12 bins (reported, not tuned):
`real 0 0 0 0 1 16 19 14 8 4 3 3 3`; `new car 0 0 0 0 0 1 24 3 1 1 0 0 0`.
Peak within 2 bins of the label: real 69%, old car 24%, new car 92%.

**Gate passed:** detector on 300 new-car engineer sequences only, real val:
AP 0.277 / 0.393 / 0.377, mean 0.349 (old car 0.002; bar 0.1;
`samples/car_gate_sim_only.json`). Step 6 started 13:12 (`scripts/run_step6.sh`).
