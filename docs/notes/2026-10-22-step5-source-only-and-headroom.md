# Step 5: source-only detectors and the headroom question (pre-registered 2026-10-22)

Step 4 (`samples/step4_*.json`) found that neither extra source helps: real 0.828,
real+sim 0.800 / 0.776, real+DiT 0.768 / 0.768 (engineer / fitted), with the
largest seed range 0.068. That experiment could not separate two explanations:
the generator adds nothing beyond its simulator, or the task leaves no room for
extra data at 387 real sequences. Step 5 tests each one directly.

Detector settings in every arm are those of step 4: 3,000 steps, 8 frames per
step, width 32, seeds 1/2/3, AP on the 121 real test sequences. Every source is
normalised by its own statistics. Pipeline: `scripts/run_step5.sh`
(`sbatch scripts/step5.sbatch` to resume), log `logs/step5.log`.

## A. Source only: did fine-tuning teach the generator real vehicles?
Detectors trained on one source alone, with no real data, tested on real:
`sim` = 1,548 simulator sequences; `dit` = the 1,548 sequences step 4 generated
from the DiT fine-tuned on all 387 real sequences.

Rule, per variant: bar = the larger seed range of the two arms.
- dit - sim > bar: **the mechanism is alive**; the generator learned real
  signatures, and step 4's null came from how real and extra data were combined.
- |dit - sim| <= bar: **the generator reproduces its simulator**; this is the
  falsifier set on 2026-09-21.
- sim - dit > bar: raw simulator data is the better source.

Caveat: the DiT was fine-tuned on the same 387 real sequences the
detector is compared against, so `dit` beating `sim` cannot show that it beats
real data; only that it carries something real the simulator lacks.

## B1. Headroom: how much does more real data help?
Real-only detectors on nested random subsets of the 387 training sequences:
25, 50, 100, 200 (`scripts/make_radial_subset.py`, permutation seed 0; each
subset's statistics computed on itself). N = 387 is step 4's real arm.

Rule (`scripts/step5_choose_n.py`): headroom(N) = AP(387) - AP(N); bar = the
largest seed range among the five real-only points. N* = the largest N in
200/100/50/25 with headroom > bar. If none: **the task cannot show an
augmentation gain at any size**, and B2 is skipped.

## B2. The three step 4 arms at N*
Both DiTs fine-tuned again from their pretrained weights, on the N* subset only
(the same 9,600 optimizer steps and recipe as the 387 fine-tune; the subset is
repeated so an epoch stays ~50 steps). 4 x N* sequences generated from the
subset's labels; the simulator arm gets 4 x N* simulator sequences. Same rule as
step 4: D vs B with the largest seed range as the bar, plus D - A and B - A.
Memorisation check on each generator (training labels = the subset, held out =
val); ratio < 0.8 means copying, and a positive D then counts for less.

## Results
(appended after the runs)

### A (finished 11:42): DiT-only > sim-only on both variants
| | sim only | DiT only | bar | verdict |
|---|---:|---:|---:|---|
| engineer | 0.002 | 0.132 | 0.068 | DiT > sim |
| fitted | 0.000 | 0.273 | 0.210 | DiT > sim |

### Stopped 11:47 by Ari: the simulated car is wrong
Sim-only AP ~0 led to the cause. `src/scene_sim.py` `_car` draws a car as five
equally bright points (four corners + centre), which appear in range as three
one-bin spikes 11 bins (2.25 m) apart with the label on the middle one. Real
RADIal cars are one blob: peak at the label, tail behind it. Median profile at
the label's Doppler, dB above the range ring, offsets -12..+12 bins step 2:
real `0 0 0 0 1 16 19 14 8 4 3 3 3`; sim `14 7 0 0 0 1 21 0 0 0 0 5 13`.
Physically the far corners are shadowed by the body. This invalidates every
simulator arm (step 4 real+sim, step 5 sim-only) and taints both pretrained
DiTs; A's DiT-only numbers are a DiT unlearning a wrong car from real data.
B1 was stopped at its first point (N=25); B2 never ran. Ari's choice for the
redo: fine-tune the DiT on a fixed small real subset, N = 50.
