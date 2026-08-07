---
title: radSeq Simulator vs. RDDiffusion
tags:
  - radar
  - simulation
  - radseq
---

# radSeq Simulator vs. RDDiffusion

radSeq keeps RDDiffusion's basic radar formation model—**target IQ signal + SIRP/K-distributed clutter + Gaussian noise**, transformed into a 64×64 Range–Doppler map—but extends it from independent images into physically connected sequences.

## Main differences

- **RDDiffusion:** Each sample is one independent frame. Target range and velocity are randomly resampled every time.
- **radSeq:** Each sample is a 16-frame sequence. A target's initial range $r_0$, velocity $v_0$, and acceleration $a$ are sampled once and evolved using:

$$
r(t)=r_0+v_0t+\frac{1}{2}at^2, \qquad v(t)=v_0+at
$$

- Trajectories that leave the radar grid are rejected, keeping targets visible throughout the sequence.
- RDDiffusion generates fresh clutter per image. radSeq evolves clutter with an **AR(1) process** controlled by $\rho$: low $\rho$ changes quickly; high $\rho$ is nearly stationary.
- radSeq adds three target types: steady point targets, frame-varying Swerling-1 targets, and range-extended three-scatterer targets.
- radSeq records complete trajectories, initial velocity, acceleration, class, target count, measured SCNR, and clutter correlation instead of only a binary target-label map.
- The output is a sequence of **log-magnitude RD maps in dB**, ready for diffusion training.
- radSeq also makes the Doppler label grid exactly match its steering matrix. RDDiffusion's rounded/`linspace` grid could shift labels by approximately one bin.

> [!summary]
> **RDDiffusion generates unrelated radar snapshots. radSeq uses essentially the same radar physics to generate connected scenes whose targets and clutter evolve coherently through time.**

