"""Locate WHERE the generated upper-tail compression originates.

diag_marginal_l1.py established the what: generated sequences have ~0.63x the
real standard deviation, with the bright tail (99th pct +1.24 sigma vs real
+2.50) collapsed. This script separates the candidate sources:

  A. denoise fidelity - take REAL x0, noise it to t, ask the model to denoise.
     Compares pred_x0 against the true x0 it came from. Isolates the model's
     one-shot denoising from any sampling-trajectory compounding. If the tail
     is already flat here at low t, the model is the problem, not the sampler.

  B. sampler budget - DDIM at 50/100/250 steps plus full ancestral DDPM.
     If the tail returns with more steps, it is a sampling artifact.

  C. training progression - epoch 10/30/50/70/best. If the tail keeps widening
     across checkpoints, the run is simply undertrained; if it plateaus early,
     the objective (not the compute) is the limit.

Read-only. Writes a JSON report.
"""
import argparse
import json
from pathlib import Path

import torch

from src.diffusion import GaussianDiffusion
from src.eval.metrics import marginal_l1
from src.train import build_model


def tail_stats(x, mean=None, std=None):
    """Summarize a tensor in normalized units (mean/std given => convert)."""
    f = x.flatten().float()
    if mean is not None:
        f = (f - mean) / std
    sub = f if f.numel() <= 16_000_000 else f[:: f.numel() // 16_000_000 + 1]
    q = torch.quantile(sub, torch.tensor([0.5, 0.75, 0.99, 0.999, 1.0]))
    return {"mean": round(float(f.mean()), 4), "std": round(float(f.std()), 4),
            "p50": round(float(q[0]), 4), "p75": round(float(q[1]), 4),
            "p99": round(float(q[2]), 4), "p999": round(float(q[3]), 4),
            "max": round(float(q[4]), 4)}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt.get("epoch"), ckpt.get("step")


def load_raw_val(cache_dir, limit):
    seqs = []
    for f in sorted(Path(cache_dir).glob("val_*.pt")):
        for item in torch.load(f):
            seqs.append(item["x"])
            if len(seqs) >= limit:
                return torch.stack(seqs)
    return torch.stack(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/phase1_wandb/best.pt")
    ap.add_argument("--ckpt-dir", default="checkpoints/phase1_wandb")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n-real", type=int, default=64)
    ap.add_argument("--n-ancestral", type=int, default=8)
    ap.add_argument("--out", default="samples/diag_tail_origin.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(Path(args.cache) / "stats.pt")
    mean, std = stats["mean"], stats["std"]
    report = {"config": vars(args)}

    real_raw = load_raw_val(args.cache, args.n_real)
    real_norm = ((real_raw - mean) / std)
    report["real_normalized"] = tail_stats(real_norm)
    print("real:", report["real_normalized"])

    model, cfg, epoch, step = load_model(args.ckpt, device)
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    L = cfg["data"]["seq_len"]
    report["best_ckpt"] = {"epoch": epoch, "step": step}

    # ---- A. one-shot denoising fidelity on REAL noised data -------------
    # No sampling loop involved: the only thing under test is the model's
    # eps prediction at a single timestep.
    x0 = real_norm[: args.n].to(device)
    denoise = {}
    torch.manual_seed(0)
    for t_val in [10, 50, 100, 200, 400, 600, 800, 999]:
        t = torch.full((x0.shape[0],), t_val, device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = diff.q_sample(x0, t, eps)
        with torch.no_grad():
            eps_hat = model(xt, t, None)
        x0_hat = diff.pred_x0(xt, t, eps_hat)
        denoise[t_val] = {
            "x0_hat": tail_stats(x0_hat.cpu()),
            "eps_mse": round(float((eps - eps_hat).pow(2).mean()), 5),
            "x0_mse": round(float((x0 - x0_hat).pow(2).mean()), 5),
        }
        print(f"  t={t_val:4d} x0_hat std={denoise[t_val]['x0_hat']['std']:.3f} "
              f"p999={denoise[t_val]['x0_hat']['p999']:.3f} "
              f"eps_mse={denoise[t_val]['eps_mse']:.4f}")
    report["A_denoise_fidelity"] = denoise
    report["A_true_x0"] = tail_stats(x0.cpu())

    # ---- B. sampler budget ---------------------------------------------
    sampler = {}
    for steps in [50, 100, 250]:
        torch.manual_seed(1)
        g = diff.ddim_sample(model, (args.n, L, 64, 64), device, steps=steps).cpu()
        sampler[f"ddim_{steps}"] = {
            "stats": tail_stats(g),
            "marginal_l1": round(marginal_l1(g * std + mean, real_raw), 4)}
        print(f"  ddim {steps}: std={sampler[f'ddim_{steps}']['stats']['std']:.3f} "
              f"L1={sampler[f'ddim_{steps}']['marginal_l1']:.4f}")
    torch.manual_seed(1)
    ga = diff.p_sample_loop(model, (args.n_ancestral, L, 64, 64), device).cpu()
    sampler["ancestral_1000"] = {
        "stats": tail_stats(ga),
        "marginal_l1": round(marginal_l1(ga * std + mean, real_raw), 4)}
    print(f"  ancestral: std={sampler['ancestral_1000']['stats']['std']:.3f} "
          f"L1={sampler['ancestral_1000']['marginal_l1']:.4f}")
    report["B_sampler"] = sampler

    # ---- C. training progression ---------------------------------------
    prog = {}
    for name in ["epoch_0010.pt", "epoch_0030.pt", "epoch_0050.pt",
                 "epoch_0070.pt", "best.pt"]:
        p = Path(args.ckpt_dir) / name
        if not p.exists():
            continue
        m, c, ep, st = load_model(p, device)
        d = GaussianDiffusion(c["diffusion"]["timesteps"])
        torch.manual_seed(1)
        g = d.ddim_sample(m, (args.n, L, 64, 64), device, steps=50).cpu()
        prog[name] = {"epoch": ep, "step": st, "stats": tail_stats(g),
                      "marginal_l1": round(marginal_l1(g * std + mean, real_raw), 4)}
        print(f"  {name}: epoch={ep} std={prog[name]['stats']['std']:.3f} "
              f"p99={prog[name]['stats']['p99']:.3f} "
              f"L1={prog[name]['marginal_l1']:.4f}")
        del m
    report["C_progression"] = prog

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
