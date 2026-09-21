"""Score DiT checkpoints on the easy E0 regime (1 steady target, no clutter/noise).

The corrected CFAR detector is unreliable on noise-free maps (sidelobes link
as tracks), so E0 is scored directly: distribution match plus whether each
frame has ONE dominant peak that moves on a straight range-Doppler line.
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset, denormalize
from src.diffusion import GaussianDiffusion
from src.sample import select_checkpoint_state
from src.train import build_model


def e0_metrics(x_db, stats):
    """x_db: (B, L, N, K) dB maps."""
    z = (x_db - stats["mean"]) / stats["std"]
    B, L, N, K = x_db.shape
    flat = x_db.reshape(B, L, -1)
    peak_db, idx = flat.max(-1)
    r, d = (idx // K).float(), (idx % K).float()
    # second-highest value at least 4 bins away from the peak
    rr = torch.arange(N)[:, None].expand(N, K).reshape(-1)
    dd = torch.arange(K)[None, :].expand(N, K).reshape(-1)
    far = ((rr[None, None] - r[..., None]).abs() > 4) | ((dd[None, None] - d[..., None]).abs() > 4)
    second = torch.where(far, flat, torch.full_like(flat, -1e9)).max(-1).values
    t = torch.arange(L, dtype=torch.float32)
    A = torch.stack([torch.ones(L), t], 1)
    proj = A @ torch.linalg.pinv(A)

    def line_rms(y):
        return (y - y @ proj.T).pow(2).mean(-1).sqrt()

    hist = torch.histc(z.flatten().float(), bins=100, min=-5, max=5)
    return {
        "mean": float(z.mean()), "std": float(z.std()),
        "frame_diff_std": float((z[:, 1:] - z[:, :-1]).std()),
        "peak_db_median": float(peak_db.median()),
        "peak_prominence_db_median": float((peak_db - second).median()),
        "range_line_rms_bins_median": float(line_rms(r).median()),
        "doppler_line_rms_bins_median": float(line_rms(d).median()),
        "frac_seq_on_line": float(((line_rms(r) < 1.0) & (line_rms(d) < 1.0)).float().mean()),
        "_hist": hist / hist.sum(),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=CHECKPOINT")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--terminal", default="mean", choices=("model", "mean"))
    ap.add_argument("--cache", default="data/cache_easy")
    ap.add_argument("--out", default="experiments/dit_64/results/e0_scores.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = RadarSequenceDataset(args.cache, "val")
    stats = val.stats
    real = torch.stack([val.items[i]["x"] for i in range(2 * args.n)]).float()
    ra, rb = e0_metrics(real[:args.n], stats), e0_metrics(real[args.n:], stats)
    report = {"real": {k: v for k, v in rb.items() if k != "_hist"},
              "real_vs_real_marginal_l1": float((ra["_hist"] - rb["_hist"]).abs().sum()),
              "terminal_x0": args.terminal, "arms": {}}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = build_model(cfg, device)
        model.load_state_dict(select_checkpoint_state(ckpt, weights="raw"))
        model.eval()
        diff = GaussianDiffusion(
            cfg["diffusion"]["timesteps"],
            parameterization=cfg["diffusion"].get("parameterization", "eps"),
            terminal_x0=args.terminal)
        per_seed = {}
        for seed in (int(s) for s in args.seeds.split(",")):
            torch.manual_seed(seed)
            x = diff.ddim_sample(model, (args.n, cfg["data"]["seq_len"], 64, 64),
                                 device, steps=args.steps)
            m = e0_metrics(denormalize(x.cpu(), stats), stats)
            m["marginal_l1"] = float((m.pop("_hist") - rb["_hist"]).abs().sum())
            per_seed[seed] = m
            if seed == 1:
                torch.save(denormalize(x.cpu(), stats), Path(args.out).with_suffix(f".{name}.pt"))
        report["arms"][name] = {"checkpoint": path, "step": ckpt.get("step"),
                                "attn_mode": cfg["model"].get("attn_mode", "temporal"),
                                "lambda_smooth": cfg["train"]["lambda_smooth"],
                                "per_seed": per_seed}
        print(name, json.dumps(per_seed[min(per_seed)], indent=None), flush=True)
    print("real", json.dumps(report["real"]), "real-vs-real L1",
          round(report["real_vs_real_marginal_l1"], 3))
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
