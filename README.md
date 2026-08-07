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
