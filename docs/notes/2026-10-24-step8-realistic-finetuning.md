# Step 8: fine-tuning that keeps the background real (pre-registered 2026-10-24)

## Why
Step 7's realism gate failed twice on fold 0 (`samples/step7_sampler_*`). Full
fine-tuning on 50 real sequences smooths away clutter as it trains: clutter-peak
error 15% at ~500 steps, 34% at 800 (the val-loss best, which failed), 68% at
9,600. Validation loss stays flat over that range (0.2648 vs 0.2640), so it
cannot pick a realistic checkpoint. The ~500-step pass was mostly the pretrained
model showing through the EMA (0.999^500 = 61% of it still pretrained), so it is
not counted as a pass.

## Variants (fold 0, 50 sequences, from the pretrained engineer DiT's EMA)
- `base`: the step 7 recipe (lr 5e-5, all parameters)
- `lowlr`: lr 1e-5, all parameters
- `partial`: lr 5e-5, only the input/condition projection, the adaLN
  modulation layers and the output layer (`train.trainable`); the attention and
  MLP weights stay pretrained
Each runs 2,000 steps, with a snapshot every 200.

## Selection and gate (scripts/realism_sweep.py)
Candidates: every snapshot x {EMA, raw} weights; sampler DDIM 250, eta 1.
Score: mean relative error of speckle, frame change, bright tail and clutter
peaks against real val sequences 0-15. The lowest score wins. **Gate: the
winner, on real val sequences 16-31 (never used for the choice), must have every
statistic within 25%.** The pretrained model with no fine-tuning is scored as a
reference and cannot be chosen.

## Then
Pass: detector study on fold 0 with the winner (as in step 7: controlled arms,
per-frame and multi-frame detectors, bootstrap verdicts); fold 1 repeats the
winning recipe (variant, step count, weights) without re-choosing, gated the
same way; verdicts pooled over both folds.
Fail: no fine-tuning variant keeps real-looking clutter; per the 2026-10-24
decision, this generator in its current form cannot learn from 50 real
sequences without losing realism, and the study stops there.

## Results
(appended after the runs)
