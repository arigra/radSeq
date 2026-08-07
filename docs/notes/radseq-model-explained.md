---
title: radSeq Model Explained
tags:
  - radar
  - diffusion
  - transformer
  - radseq
---

# radSeq Model Explained

radSeq uses a **Diffusion Transformer (DiT)** to generate an entire 16-frame radar sequence simultaneously.

```text
Noisy RD sequence
      ↓
Split every frame into overlapping patches
      ↓
Represent patches as transformer tokens
      ↓
Compare each patch with the same location across time
      ↓
Predict and remove diffusion noise
      ↓
Reconstruct the 16 RD frames
```

## Temporal-only attention

The transformer's attention operates **only through time**. For example, it follows one spatial patch across the sequence:

```text
Patch (3,5): frame 1 → frame 2 → ... → frame 16
```

It does not directly connect that patch to distant patches in the same frame. This reflects the assumption that distant RD regions are physically independent, while the same region in consecutive frames is related through target motion.

## Patches and inputs

Each 64×64 frame is divided into overlapping **8×8 patches with stride 4**. The overlap preserves local spatial information and prevents moving targets from disappearing abruptly at patch boundaries.

Each token contains:

- The radar values inside its patch
- Its spatial position in the RD map
- Its frame number in the sequence

The model also receives the **diffusion timestep**, which tells it how noisy the current sequence is.

## Training

1. Take a clean simulated sequence.
2. Add a randomly selected amount of Gaussian noise.
3. Ask the model to predict the added noise.
4. Penalize incorrect noise predictions.
5. Add a smoothness penalty for abrupt changes between reconstructed frames.

Later training phases add explicit trajectory/Doppler losses and controllable scene conditioning.

## Generation

Generation begins with pure random noise. The model repeatedly removes predicted noise until a radar sequence appears. DDIM sampling reduces this process from the 1,000 training diffusion levels to approximately 50 generation steps.

> [!summary]
> **The model learns how local radar regions evolve through time, then uses that knowledge to turn random noise into a coherent 16-frame RD sequence.**

