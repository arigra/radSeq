# radSeq

A diffusion model (DiT) that generates sequences of radar Range-Doppler maps. The question: is a model pretrained on a simulator and fine-tuned on a little real data (RADIal) a better source of detector training data than the simulator itself?

Each part below shows the real code and runs a small piece of it here, on this Mac. Anything that needs RADIal data, a GPU or a checkpoint runs on the HPC and says so.

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cpu")
```

## 1 · RADIal (real data)
Real recordings from a 77 GHz radar: 512 range × 256 Doppler × 16 receive channels per frame, summed into one dB map. The data lives on the HPC; here there is only the split, which is by whole recording so no drive is in two splits.

```file src/radial.py 114-133
```

```python
import json

split = json.load(open("data/radial/split.json"))
{name: len(recs) for name, recs in split.items()}
```

## 2 · Simulator
Builds a road scene (city, highway or countryside) and renders 8 frames, 0.2 s apart, from radar physics. Two variants: *engineer*, never tuned to RADIal, and *fitted*, whose knobs were fitted to RADIal's training recordings. Only moving vehicles get labels, and many scenes have none: of the first 12 training seeds, 6 have no labelled vehicle. The one below (seed 10,000,007) has four.

```file src/scene_sim.py 321-380
```

```python
from src.scene_sim import SceneSimulator, Scenario, fitted_scenario

def one_sequence(scenario, seed):
    sim = SceneSimulator(scenario=scenario, seq_len=8, generator=torch.Generator().manual_seed(seed))
    return sim.gen_sequence()

SEED = 10_000_007                      # training sequence no. 7 of the engineer cache
eng = one_sequence(Scenario(), SEED)
fit = one_sequence(fitted_scenario(), SEED)
vehicles = {lab["id"] for frame in eng["labels"] for lab in frame}
print(eng["road_type"], f"· ego {eng['ego_speed']:.1f} m/s ·", len(vehicles), "moving vehicles ·", tuple(eng["x"].shape))
```

```python
# one colour scale for both, so brightness can be compared
lo = min(eng["x"][0].min(), fit["x"][0].min()); hi = max(eng["x"][0].max(), fit["x"][0].max())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, o) in zip(axes, [("engineer", eng), ("fitted", fit)]):
    im = ax.imshow(o["x"][0], aspect="auto", origin="lower", cmap="viridis", vmin=lo, vmax=hi)
    ax.set_title(f"{name} · frame 0 · {o['road_type']}")
    ax.set_xlabel("Doppler bin")
axes[0].set_ylabel("range bin")
fig.colorbar(im, ax=axes, label="dB")
plt.show()
```

What to look for: each return repeats across Doppler (the radar's DDMA copies, with an empty band where slots are unused), and the engineer variant's far range is brighter than the fitted one's. Against RADIal the engineer's far range is 11 dB too bright.

## 3 · Prepare
Renders 6,000 sequences per variant to disk (float16), with one seed per sequence so the cache is reproducible, and a mean/std for normalising. Vehicle positions become a condition map the model is trained with. Here: a tiny cache of 8 sequences.

```file scripts/build_scene_cache.py 21-64
```

```python
import os, subprocess, sys

run = subprocess.run([sys.executable, "scripts/build_scene_cache.py", "--variant", "engineer",
                      "--n-train", "8", "--n-val", "2", "--out", "data/scene_engineer_tiny"],
                     env={**os.environ, "PYTHONPATH": "."}, capture_output=True, text=True, check=True)
print(run.stdout.strip().splitlines()[-1])
```

```file src/scene_data.py 43-62
```

```python
from src.scene_data import SceneSequenceDataset

ds = SceneSequenceDataset("data/scene_engineer_tiny", "train")
item = ds[7]                            # the same scene as in part 2
print(len(ds), "sequences;", "x", tuple(item["x"].shape), "traj", tuple(item["traj"].shape), "present", tuple(item["present"].shape))
print(f"normalised: mean {item['x'].mean():.2f}, std {item['x'].std():.2f}  (stats: {ds.stats})")
```

```file src/trajectory_condition.py 51-69
```

```python
from src.trajectory_condition import render_vehicle_condition

batch = {k: v.unsqueeze(0) for k, v in item.items()}
cond_map = render_vehicle_condition(batch["traj"], batch["present"])
print("condition map", tuple(cond_map.shape), "· vehicle pixels > 0.5:", int((cond_map[:, :, 0] > 0.5).sum()))
```

## 4 · DiT model
Cuts each frame into 16×16 patches (512 per frame, 4,096 per sequence). Each of the 12 blocks runs attention over time (the same patch across the 8 frames), then over space (all patches in one frame), then an MLP.

```file src/dit.py 64-103
```

```python
import yaml
from src.train import build_model

cfg = yaml.safe_load(open("configs/pretrain_engineer.yaml"))
model = build_model(cfg, device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

with torch.no_grad():
    out = model(batch["x"], torch.tensor([500]), None, cond_map=cond_map)
print("in", tuple(batch["x"].shape), "→ out", tuple(out.shape))
```

## 5 · Training
Adds noise to the maps and teaches the model to predict v (the noise-and-signal mix) back: cosine schedule with shift 16, 60,000 steps per variant, EMA 0.9999, on the HPC. Here: the loss of one step on the tiny batch, without bf16.

```file src/train.py 104-129
```

```file configs/pretrain_engineer.yaml
```

```python
from src.diffusion import diffusion_from_config
from src.train import _loss_components

local = {**cfg, "train": {**cfg["train"], "amp": "off"}}   # bf16 is for the GPU
diff = diffusion_from_config(cfg["diffusion"])
total, parts = _loss_components(model, None, diff, batch, local, device, dropout_p=0.1)
total.backward()
print({k: round(float(v), 4) for k, v in parts.items()}, "→ total", round(float(total), 4))
```

`smooth` is computed on every step and printed in the training log, but `lambda_smooth: 0.0` in the config means the total is `dit` alone.

## 6 · Sample, eval
Starts from pure noise and removes it in 30 DDIM steps, guided by the vehicle condition. Needs a trained checkpoint, which only exists on the HPC.

```file src/sample.py 97-131
```

```python
from pathlib import Path

ckpts = sorted(Path("checkpoints/pretrain_engineer").glob("*.pt"))
print(f"{len(ckpts)} checkpoints here" if ckpts else "No checkpoint on this machine: training ran on the HPC (started 21/9; whether it finished is unknown from here).")
```

## 7 · Experiment (not built)
Fine-tune the pretrained DiT on RADIal, then train a detector three ways: real data only (A), real + simulator data (B), real + DiT data (D). No code exists for this yet.
