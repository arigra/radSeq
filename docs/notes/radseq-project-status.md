---
title: radSeq — Project Status
tags:
  - radar
  - diffusion
  - research
---

# radSeq

## What it does

radSeq generates synthetic **16-frame, 64×64 Range–Doppler (RD) sequences**. The aim is to create controllable radar training data with realistic target motion and clutter when labeled real-world data is scarce.

## Pipeline

`Simulator → cached sequences → diffusion training → DDIM sampling → physics/statistical evaluation`

- **Simulator:** Generates 1–5 targets with velocity and acceleration, three target classes, and temporally correlated radar clutter. It also records exact trajectories and scene parameters.
- **Data:** 20,000 training and 2,000 validation sequences (about 5.5 GB), normalized using fixed training-set statistics.
- **Model:** A Diffusion Transformer using overlapping 8×8 patches. Attention runs **only through time at each spatial location**; distant regions within a frame never directly attend to one another.
- **Output:** Generated RD grids/GIFs plus trajectory, Doppler, persistence, intensity-distribution, and CFAR-related metrics.

## Training phases

1. **Backbone:** Diffusion denoising + temporal smoothness; unconditional generation.
2. **Physics:** Adds differentiable constant-velocity trajectory and Doppler-consistency losses.
3. **Conditioning:** Adds commanded motion, target class, clutter, and SCR using classifier-free guidance.

All phases are implemented; only **Phase 1 has a completed full-data experiment**.

## Current result (rescored 2026-08-07)

Phase 1 ran for **97,500 steps / 78 epochs** and stopped early after validation stopped improving. Use `checkpoints/phase1_wandb/best.pt` for evaluation.

Metrics below use the **corrected** detector (`max_peaks=5`, `min_track_len=8`), validated at 93.5% track precision / 90.7% target recall against ground truth. The earlier numbers were computed over tracks that were only 19.9% real targets and are superseded — see [the diagnosis note](2026-08-07-phase1-marginal-l1-diagnosis.md).

| Metric ↓ unless noted | Reference | Generated | Meaning |
|---|---:|---:|---|
| Velocity inconsistency | **1.358** | 2.147 | Target motion is *less* consistent than real |
| Doppler drift | 0.508 | **0.469** | Comparable; the one metric still holding up |
| Track persistence ↑ | **0.1437** | 0.0589 | Generated targets persist 2.4× less |
| Target tracks / seq | 3.09 | 2.59 | Slightly under-generates targets |
| Marginal intensity L1 | **0.0647** | 0.4084 | Intensity distribution does not match |

## Bottom line

Two established findings:

1. **The intensity gap is architectural.** The DiT gives every spatial site its own independent temporal transformer (`src/dit.py:40-42`), with no cross-spatial information flow. It denoises real noised data almost perfectly (`pred_x0` std 0.986 vs real 1.003) but cannot synthesize spatial structure from noise, so the bright tail collapses to 0.63× the real std. Converged by epoch 10 — not a training or sampling issue.
2. **The earlier "motion is learned" claim was a measurement artifact.** Correctly scored, the backbone is worse than real data on every target-level kinematic metric.

Phase 1 passes the letter of its exit criteria, but the persistence tolerance (0.2) is larger than the reference value (0.1437), so that criterion cannot fail. It should be tightened before it gates anything.

Phase 2 and Phase 3 remain implemented but untrained. The open decision is whether to test factorized spatial-temporal attention — which the spec lists under "Out of scope (possible later ablation)" — or to reframe around the restoration regime where the architecture demonstrably works.

## Key locations

- `configs/phase1_wandb.yaml` — completed experiment configuration
- `checkpoints/phase1_wandb/best.pt` — selected Phase 1 model
- `samples/phase1_wandb_best/` — generated and reference visualizations
- `src/simulator.py` — radar sequence simulator
- `src/dit.py` — temporal-only transformer
- `src/train.py` / `src/sample.py` — training and generation
- `src/eval/metrics.py` — evaluation metrics

