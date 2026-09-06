# Autonomous queue — pre-registration

**Fixed 2026-09-06 12:5x, before any queued run started.** Approved by the user
as a fixed queue; it is not to be extended without asking.

## Item 1 (already running): objective arms

`vpred` and `minsnr`, three seeds each. Rule already committed in
`2026-09-06-objective-arms-preregistration.md`: an arm is a fix only if, averaged
over its three seeds, generated std > **0.85** AND target tracks/seq >= **1.75**.

## Item 2 (revised): does the deficit close with training budget?

Every arm measured so far stopped at 12,500 steps with validation still
improving, so no negative result to date distinguishes "cannot" from "not yet".

**The planned 50,000-step retrain is unnecessary and has been cancelled.** The
finished Phase-1 run already went 97,500 steps / 78 epochs, and its epoch
snapshots are on disk. What was never done is scoring those snapshots on the
*corrected* detector — the epoch progression was only ever checked on generated
std, which is flat (0.659 at epoch 10 to 0.667 at epoch 70).

So item 2 becomes an evaluation, not a training run: score
`checkpoints/phase1_wandb/{epoch_0010, epoch_0030, epoch_0050, epoch_0070,
best}.pt` under the current paired protocol, reporting target tracks/seq and
persistence as well as the distribution statistics. Cost: minutes instead of
3.5 hours of GPU.

**Rule:** if target tracks/seq rises materially (> 0.22, the measured noise
floor for that metric) between epoch 10 and epoch 70, then every 12,500-step
negative result in this project is confounded by budget and must be re-run
longer before it is believed. If it is flat, budget is excluded and the
12,500-step comparisons stand.

## Item 3 (conditional): composition

Only if item 1 produced an arm clearing both bars. Sample that arm's checkpoint
with the hann decode, which independently gives +0.6 target tracks/seq.

**Rule:** the combination is worth adopting only if it beats *both* parents on
target tracks/seq without pushing marginal L1 above 0.60.

## Item 4: smoothness magnitude sweep

`lambda_smooth` in {0.03, 0.3}, single seed each, everything else at baseline.
Measured so far: 0.0 gives std 1.285 / 0.34 tracks; 0.01 gives 0.826 / 0.59;
0.1 (baseline) gives 0.655 / 1.75. The term is the dominant lever on variance
and the region between 0.01 and 0.1 has never been looked at.

**Rule:** a setting is interesting only if it reaches std > 0.85 while holding
target tracks/seq >= 1.75 — the same bar as item 1. A single seed cannot
establish that; anything that clears it gets two more seeds before it is
reported as a result.

## Stopping condition

When items 1, 2 and 4 have run (and 3 if triggered), the queue ends. If nothing
has cleared its bar, the outcome is written up as a negative result and the
project reframes around restoration. **No further arms are to be invented
without the user's approval** — the queue is not to be extended to keep
searching.
