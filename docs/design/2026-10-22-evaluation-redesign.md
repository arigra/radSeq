# Redesign of the RADIal augmentation study (draft for Ari, 2026-10-22)

Step 6 was stopped after engineer pretraining, before fine-tuning or any
detector ran. An external review (pasted by Ari, 2026-10-22) and my own audit
found that, even if it had finished, it could not answer "does the generator
beat its simulator?". This note lists every problem found, what is already
fixed, and a proposed protocol. Nothing below is implemented until Ari approves
it.

## Problems found

| # | problem | status |
|---|---|---|
| 1 | Arms not controlled: real+sim used the simulator's own scenes, real+DiT repeated the 50 real label sets 4x; label distributions differ (sim 34% empty frames, 32% of vehicles within +/-32 Doppler bins; real 0% and 86%) | detector study script renders the simulator's label sets with the DiT, so B and D share scenes exactly; distribution vs real still open (see 2 and 8) |
| 2 | Simulator traffic: vehicle speeds are drawn independently of the ego speed, so relative velocities are wide; on real roads most traffic drives with the ego car | open: fix from road knowledge, not by matching RADIal |
| 3 | Real-data dilution: uniform mixing (80% extra data per batch) at a fixed 3,000 steps | fixed in `scripts/radial_detector_study.py`: balanced batches, pretrain-then-fine-tune arms, early stopping on real val |
| 4 | Generator background too clean (speckle, frame change, bright tail, clutter peaks); the hit-rate gate cannot see it | measured; stochastic sampling is not enough (below); realism gate written (`src/eval/scene_realism.py`, `scripts/select_sampler.py`) |
| 5 | No confidence interval: "largest seed range" of 3 seeds, ignoring variation across test recordings and across the choice of real subset | bootstrap over test recordings written and tested (`scripts/step6_score.py`); subset variation open |
| 6 | Test set contaminated: step 5's test AP (sim-only ~0) prompted the car redesign | open: step 6-type results are exploratory unless the protocol changes (below) |
| 7 | Temporal generation never tested on real data: every detector is per-frame | open |
| 8 | **Vehicle-free frames dropped**: RADIal marks them with a -1 row and `src/radial.py:read_labels` discards those rows, so 1,065 of 8,252 labelled frames (12.9%) never enter any real cache; real data has 0% empty frames by construction, the test set has no empty scenes, and dropping them also breaks consecutive runs | open (found 2026-10-22) |
| 9 | Peak detection did not wrap the Doppler axis: a vehicle across bins 255/0 gave two detections, one a false alarm | fixed (`detect(..., circular_doppler=True)`), applied to every RADIal scorer |
| 10 | Provenance: code was edited while step 6 ran | `scripts/write_manifest.py` freezes commit + hashes before a run; pipelines must not be edited while running |
| 11 | `docs/prop.md` no longer describes the model (factorized attention, non-overlapping patches, v-prediction, no physics losses); its "distant RD cells are independent" premise is contradicted by RADIal's 12 distant DDMA copies | open: documentation |

## Evidence on problem 4 (step 4 generator, 16 real val label sets)

| | real val | DDIM 30, deterministic (used so far) | best: DDIM 250, eta 1, raw weights |
|---|---|---|---|
| speckle | 1.58 dB | 1.04 | 1.28 |
| frame-to-frame change | 2.67 dB | 1.73 | 2.05 |
| brightest cells above median | 26.9 dB | 20.7 | 23.0 |
| strong clutter peaks / frame | 247 | 84 | 119 |

The sampler explains a part; the model itself under-produces clutter.
Diagnostic of where (pretraining vs fine-tuning): see the section added below.

## Proposed protocol

**Order matters: fix the data and the generator before any detector study.**

1. **Real data (problem 8).** Keep vehicle-free frames; rebuild the real
   caches; re-pin the split by recording (the recording split itself is
   unchanged). Every earlier real-data number becomes "without empty frames".
2. **Simulator traffic (problem 2).** From road knowledge: same-direction
   traffic speed = ego speed plus a spread; oncoming traffic as now; parked
   cars unchanged. Stated before looking at RADIal's Doppler histogram again.
   Then rebuild caches and re-pretrain (engineer ~6 h, fitted ~6 h).
3. **Generator realism (problem 4).** Find the cause first (diagnostic below),
   then fix it. Gate: all four statistics within 25% of real val, else no
   detector study.
4. **Evaluation split (problems 5, 6).** Recording-level K-fold cross-validation
   (K = 5 over all usable recordings): in each fold, the real subset (N = 50
   sequences) comes from the training recordings, val from a separate slice,
   test from the held-out fold. This varies both the test recordings and the
   real subset, and the bootstrap runs over all held-out recordings. It does
   not undo the contamination of the old test recordings; the claim is then
   "consistent across folds", stated as such. Alternative: keep the old split
   and label every result exploratory.
5. **Controlled arms (problems 1, 3).** Arms share condition trajectories: the
   DiT renders exactly the simulator's scenes. Balanced batches and
   pretrain-then-fine-tune, early stopping on real val, 5 seeds on the primary
   arms, paired bootstrap verdicts.
6. **Temporal test (problem 7).** Add a multi-frame detector arm (8 stacked
   frames as input) next to the per-frame one, plus temporal fidelity of
   generated vs real sequences (frame change, vehicle persistence, blob
   correlation between frames). A generator that ignores time would score the
   same as ours on the per-frame detector, but not on these.
7. **Provenance (problem 10).** Commit, manifest, launch; no edits until done.

## Cost
Per fold and variant: fine-tune ~80 min, generation ~35 min, detectors ~3-4 h.
5 folds x 2 variants is ~45 h of GPU, plus re-pretraining (~12 h). Cheaper
first pass: 2 folds, engineer variant only (~10 h), then decide.

## Decisions for Ari
- (a) K-fold cross-validation, or keep the old split and call results exploratory?
- (b) Approve the traffic-speed change (item 2) and the empty-frame fix (item 1)?
- (c) Multi-frame detector as the temporal test, or something else?
- (d) Cheap first pass (2 folds, engineer only) before the full study?
