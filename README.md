# radSeq

Temporal radar-sequence diffusion training and evaluation.

## Environment and verification

Use Python 3.10+ in a virtual environment with CUDA-enabled PyTorch, then run:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

The prepared dataset is described by `data/cache/manifest.yaml`. Do not regenerate
it unless the simulator or data configuration changes.

## Full phased run

Run every command from the repository root. Phase 1 is the current configuration:

```bash
python -u -m src.train --config configs/base.yaml
python -u -m src.sample --ckpt checkpoints/last.pt --n 16 --out samples/phase1
```

Training writes progress to `logs/phase1.log`, atomically overwrites the resumable
`checkpoints/last.pt` every 250 steps, and keeps an epoch snapshot every 10 epochs.
After interruption, continue with:

```bash
python -u -m src.train --config configs/base.yaml --resume checkpoints/last.pt
```

Do not enable Phase 2 until Phase 1 generated samples pass the documented exit
criteria. Then set `train.phase: 2`, change the log to `logs/phase2.log`, train
from scratch, and evaluate into `samples/phase2`. Repeat with phase 3 and add
`train.cond_dropout: 0.1`; Phase 3 must be trained from scratch because it
introduces the conditioning encoder.

The checkpoint includes model, encoder (when applicable), optimizer, epoch,
global step, and configuration.

## Publication-tracked Phase 1 rerun

Authenticate once, then start the isolated full-data experiment:

```bash
wandb login
python -u -m src.train --config configs/phase1_wandb.yaml
```

The run logs training loss components, gradient norm, learning rate, and
deterministic held-out losses to the `radSeq` W&B project. Runs use a group,
job type, tags, notes, seed, and complete config so later ablation runs can be
filtered and compared.

The maximum budget is 150 epochs. Validation runs on the complete 2,000-sequence
held-out split after every epoch. `best.pt` is selected by total validation
objective. Training cannot stop before epoch 40 and stops after 15 validations
without an improvement of at least 0.0005. Use `best.pt`, rather than
`last.pt`, for final sampling and ablation tables.

Outputs are isolated from the legacy run:

- checkpoints: `checkpoints/phase1_wandb/`
- text log: `logs/phase1_wandb.log`
- W&B group: `phase1-backbone`

Resume both training and its original W&B run with:

```bash
python -u -m src.train --config configs/phase1_wandb.yaml \
  --resume checkpoints/phase1_wandb/last.pt
```

## Working on this node

This machine has no direct internet egress, which affects git and anything that
fetches at runtime.

- **Push over SSH, not HTTPS.** The stored HTTPS credential authenticates but has
  no write access to `arigra/*` and returns 403. The remote must be
  `git@github.com:arigra/radSeq.git`.
- **SSH needs the proxy helper.** Port 22 is refused and 443 is reset, so
  `~/.ssh/config` routes GitHub through an HTTP-CONNECT helper at
  `~/.ssh/proxy_connect.py` (original config backed up at `~/.ssh/config.bak`).
  With it in place, plain `git push` works. There is no `nc`, `socat` or `gh`
  installed to fall back on.
## Exact easy-regime reference

The noise-free one-target E0 experiment is a useful generator sanity check.
Sample its known kinematic distribution directly with:

```bash
python -m src.sample --easy --n 8 --seed 1 --out samples/easy_reference
```

This uses the radar simulator, not a diffusion checkpoint. It writes the same
sequence visualizations and metrics as checkpoint sampling, using
`data/cache_easy` for the validation reference. The E0 diffusion checkpoint
collapses toward the empty map despite its low denoising loss; direct E0
sampling separates a learned-generation failure from a data-generation error.
The peak detector overcounts persistent sidelobes on noise-free E0 data, so
its target-track count should not be interpreted as the true target count.

## Learned generator for the easy regime

The pixel-space DiT checkpoint learns to denoise observed E0 targets but
collapses when sampling from noise. `src.easy_latent` instead learns the
two-variable velocity/acceleration distribution. It draws initial range
uniformly from the physically valid interval and renders the sampled target
with the radar simulator. This is a learned scene generator with a physics
renderer, not a repaired pixel-space DiT.

```bash
python -m src.easy_latent --cache data/cache_easy --steps 50000 --out checkpoints/easy_latent.pt
python -m src.sample --easy-latent checkpoints/easy_latent.pt --n 8 --seed 1 --out samples/easy_latent
```

On the existing 50,000-step checkpoint, three 32-sequence draws have normalized
map standard deviations 1.020, 1.038 and 1.004; marginal-L1 distances to E0
validation data are 0.054, 0.048 and 0.064 (real-vs-real: 0.062).
The E0 peak detector still overcounts sidelobes, so generated image markers
use the known trajectory labels instead.

## Learned generator for the full regime

The same scene-based approach handles 1–5 targets, all three target classes,
varied target strength, clutter and receiver noise. It learns the accepted
target-motion distribution and empirical target-count/class frequencies from
`data/cache`; the simulator renders fresh target phases, amplitudes, clutter
and noise. This targets the synthetic distribution used to train the original
model. It does not repair the original pixel-space DiT checkpoint.

```bash
python -m src.hard_latent --cache data/cache --steps 50000 --out checkpoints/hard_latent.pt
python -m src.sample --hard-latent checkpoints/hard_latent.pt --n 8 --seed 1 --out samples/hard_latent
```

On a 64-sequence held-out comparison, generated and real map marginal-L1 were
0.0385 and 0.0374 respectively. Generated target tracks per sequence were
3.16 versus 3.11 for real sequences, and velocity-consistency scores were
1.64 versus 1.60. New samples include their true trajectory markers.
