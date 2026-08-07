---
title: radSeq explainer
---

# radSeq, in plain terms

radSeq generates fake radar data. Specifically: short movies (16 frames) of Range-Doppler maps — the kind of image a radar produces where one axis is distance to a target and the other is its speed. The goal is to train a diffusion model that can dream up realistic sequences of targets moving around, instead of needing real recorded radar data.

## Why this is a reasonable thing to build

Real radar datasets with good labels are scarce. If you can simulate physically plausible radar scenes and train a generative model on them, you get a controllable data source: you can ask for "a target approaching at this speed" or "with this much clutter" and get a matching sequence out, with ground truth for free since you made the scene up.

## The three pieces

**1. The simulator.** Code that fakes a radar scene: a few moving targets (following normal physics — constant velocity or acceleration) plus background clutter (like sea or ground return) that has some realistic texture and correlation from frame to frame. It writes out the image sequence and the true position/speed/class of every target, because it generated them itself.

**2. The model.** A diffusion model (the same family behind image generators like Stable Diffusion) built on a transformer. The one twist: the transformer's attention only looks across *time*, never across space. That's because in a Range-Doppler map, a pixel here has no physical relationship to a pixel somewhere else in the same frame — but a target's position at frame 5 is very related to its position at frame 6. So the model is built to only mix information along the time axis.

**3. Training happens in stages.**
- Stage 1: just teach the model to produce a coherent, smoothly-moving sequence at all.
- Stage 2: add a nudge that specifically rewards realistic target trajectories and stable speed readings.
- Stage 3: let you *tell* the model what you want (target speed, clutter level, target type) and have it follow instructions, using a technique called classifier-free guidance.

You're not supposed to jump to the next stage until the current one actually looks right — samples get compared against metrics computed from real-vs-real data as a sanity check.

## Where the project is right now

All three stages are coded and tested. Only stage 1 has actually been trained end to end, on the full dataset, and it ran until it stopped improving (early stopping). The output sequences look decent — the targets move smoothly and consistently, arguably even smoother than real data. But the overall pixel-value statistics (how bright/dark things are, including clutter texture) don't yet match real data closely. That's the open question before moving to stage 2: is stage 1 good enough, or does it need another pass first.

## If you want to poke at it yourself

- `notebooks/00_overview.ipynb` runs the whole pipeline step by step.
- `samples/phase1_wandb_best/` has the actual generated GIFs from the finished training run — just look at them.
- The plan and design spec (`docs/superpowers/`) explain every design decision in detail, if you want the "why" behind something.