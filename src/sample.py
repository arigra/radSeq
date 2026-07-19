"""Sample sequences from a checkpoint; write viz + metrics."""
import argparse
from pathlib import Path

import torch
import yaml

from src.dataset import RadarSequenceDataset, denormalize
from src.diffusion import GaussianDiffusion
from src.train import build_model


def generate(ckpt_path, n_seq, device, steps=50, cond=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    L = cfg["data"]["seq_len"]
    x = diff.ddim_sample(model, (n_seq, L, 64, 64), device, steps=steps, cond=cond)
    stats = RadarSequenceDataset(cfg["data"]["cache_dir"], "val").stats
    return denormalize(x.cpu(), stats)


if __name__ == "__main__":
    from src.eval.metrics import evaluate_sequences
    from src.viz import sequence_gif, sequence_grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = generate(args.ckpt, args.n, device, steps=args.steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, seq in enumerate(x):
        sequence_grid(seq, out / f"seq_{i}.png")
        sequence_gif(seq, out / f"seq_{i}.gif")

    cfg = torch.load(args.ckpt, map_location="cpu")["config"]
    val = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
    x_real = torch.stack([denormalize(val[i]["x"], val.stats)
                          for i in range(min(len(val), args.n * 4))])
    metrics = evaluate_sequences(x, x_real, seq_len=cfg["data"]["seq_len"])
    with open(out / "metrics.yaml", "w") as fh:
        yaml.safe_dump(metrics, fh)
    print(yaml.safe_dump(metrics))
