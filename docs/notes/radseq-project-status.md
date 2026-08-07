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

## Current result

Phase 1 ran for **97,500 steps / 78 epochs** and stopped early after validation stopped improving. Use `checkpoints/phase1_wandb/best.pt` for evaluation.

| Metric ↓ unless noted | Reference | Generated | Meaning |
|---|---:|---:|---|
| Velocity inconsistency | 3.04 | **2.67** | Motion is smooth, possibly too smooth |
| Doppler drift | 0.706 | **0.437** | Doppler is unusually stable |
| Track persistence ↑ | 0.075 | **0.083** | Comparable/slightly higher continuity |
| Marginal intensity L1 | **0.0647** | 0.4355 | Brightness and clutter statistics do not match well |

## Bottom line

The backbone has learned **temporally coherent target motion**, but not the simulator's full **intensity/clutter distribution**. The immediate research decision is whether to improve Phase 1's appearance/statistics before using it as the baseline for Phase 2. Phase 2 and Phase 3 currently have implementation evidence, not trained-result evidence.

## Key locations

- `configs/phase1_wandb.yaml` — completed experiment configuration
- `checkpoints/phase1_wandb/best.pt` — selected Phase 1 model
- `samples/phase1_wandb_best/` — generated and reference visualizations
- `src/simulator.py` — radar sequence simulator
- `src/dit.py` — temporal-only transformer
- `src/train.py` / `src/sample.py` — training and generation
- `src/eval/metrics.py` — evaluation metrics

