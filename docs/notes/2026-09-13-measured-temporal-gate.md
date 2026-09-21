# Measured-data gate: does temporal background structure matter? (2026-09-13)

Written before running `archive/codex_research/research_radial_gate.py`.

## Why not Rad-R

Codex's check (`archive/codex_research/results/research_real_track_gap.json`) used Rad-R
`training_cache.h5`: 9 captures x 200 frames, `rd_map` min-max normalised to
[0,1] and resized to 224x224, no target labels, mostly static fault-injection
scenes. Normalisation destroys absolute power, which CFAR statistics need, and
static scenes make an ordered-vs-shuffled test uninformative. Its ordered and
shuffled path counts differ by 0-8%. That result is not evidence either way.

## Data

RADIal ready-to-use (`radar_FFT`, 512 range x 256 Doppler x 16 channels;
measured driving, labelled vehicles). 84% of labelled frames follow their
predecessor; 111 runs are >= 16 consecutive frames (4,592 frames).
`archive/codex_research/cache_radial_power.py` caches channel-summed dB power for those runs.

## Question

A frame-wise generator conditioned on the vehicle list places vehicles
correctly in each frame but samples everything else independently per frame.
Does that change how many three-frame detection paths appear outside the
labelled vehicles, at equal detection occupancy?

## Test

Non-overlapping 16-frame windows. Hard CA-CFAR (4 training, 2 guard cells) on
linear power; the threshold is set per window to a fixed occupancy (1e-3,
1e-2) so only temporal arrangement differs. Cells within 8 range bins
(about 1.6 m) of a labelled vehicle are masked. A background path is a detection
linked to detections in the two previous frames within +/-k range and Doppler
cells (k = 1, 4). Null: permute the background detection frames within the
window, 20 permutations.

## Decision rule (occupancy 1e-3)

- **Premise holds**: median ordered/permuted ratio >= 1.5 for both k, and
  >= 75% of windows have ratio > 1.
- **Premise dropped**: median ratio < 1.2 for both k.
- Otherwise inconclusive.

## Result (added after running; `archive/codex_research/results/research_radial_gate.json`)

246 windows from 111 runs. Median ordered/permuted background-path ratio:

| Occupancy | k=1 | k=4 | Windows with ratio > 1 (k=1 / k=4) |
|---:|---:|---:|---:|
| 1e-3 | 1.00 | 1.18 | 39% / 68% |
| 1e-2 | 1.05 | 1.13 | 84% / 93% |

**Decision: premise dropped** (both k below 1.2 at 1e-3). On measured
RADIal, a vehicle-conditioned frame-wise background changes three-frame
background paths by a median 0-18%. A diagnostic on 20 runs found no
explanation from persistent cells (median 0% of background detections in cells
lit >= 8/16 frames) or TDM-MIMO Doppler replicas (largest mod-16 share 0.088
vs uniform 0.0625).

Caveat: unlabelled real objects (guardrails, pedestrians, parked cars) count as
background. That is intended: a frame-wise generator must reproduce them too.
