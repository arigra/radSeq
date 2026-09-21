"""Per-step DDIM trace: where does the generated distribution get decided?

Hypothesis under test: at the first DDIM steps alpha_bar is ~0 (2.4e-9 at
t=999), so x0 = (x - sqrt(1-ab) eps_hat) / sqrt(ab) turns tiny eps errors into
huge x0 values, the +/-4 clamp saturates, and the final sample inherits that.

Records per step: unclamped x0 std, fraction of x0 pixels outside the clamp,
clamped x0 mean/std, and correlation of the clamped x0 with the final sample.
Reference: one-step x0 from noised REAL validation data at the same timesteps.
"""
import argparse
import json
from pathlib import Path

import torch

from src.dataset import RadarSequenceDataset
from src.diffusion import DEFAULT_X0_CLAMP, GaussianDiffusion
from src.sample import select_checkpoint_state
from src.train import build_model


def corr(a, b):
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def trace(ckpt_path, n, steps, seed, device, mean_x0_below=0.0):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ckpt, weights="raw"))
    model.eval()
    diff = GaussianDiffusion(
        cfg["diffusion"]["timesteps"],
        x0_clamp=cfg["diffusion"].get("x0_clamp", DEFAULT_X0_CLAMP),
        parameterization=cfg["diffusion"].get("parameterization", "eps"))
    lo, hi = diff.x0_clamp
    shape = (n, cfg["data"]["seq_len"], 64, 64)
    ts = torch.linspace(diff.T - 1, 0, steps, device=device).long()

    val = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
    real = torch.stack([val[i]["x"] for i in range(n)]).to(device)

    torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    rows, x0s = [], []
    for i in range(steps):
        t = ts[i].repeat(n)
        eps, x0_raw = diff.to_eps_x0(model(x, t), x, t)
        x0 = diff.clamp_x0(x0_raw)
        ab_t = diff._ab(t, x)
        if float(ab_t.flatten()[0]) < mean_x0_below:
            # SNR ~ 0: the posterior-mean x0 is the data mean (0 after
            # normalisation); eps then follows exactly from x = sqrt(ab) x0 + ...
            x0 = torch.zeros_like(x)
            eps = x / (1 - ab_t).sqrt()
        x0s.append(x0.cpu())
        # restoration reference: same timestep, real data, fresh noise
        g = torch.Generator(device=device).manual_seed(seed + i)
        xt_real = diff.q_sample(real, t, torch.randn(shape, device=device, generator=g))
        _, x0_real = diff.to_eps_x0(model(xt_real, t), xt_real, t)
        rows.append({
            "step": i, "t": int(ts[i]), "alpha_bar": float(diff.alphas_bar[ts[i]]),
            "eps_hat_std": float(eps.std()),
            "x0_raw_std": float(x0_raw.std()),
            "x0_frac_clamped": float(((x0_raw < lo) | (x0_raw > hi)).float().mean()),
            "x0_mean": float(x0.mean()), "x0_std": float(x0.std()),
            "x_std": float(x.std()),
            "real_x0_frac_clamped": float(((x0_real < lo) | (x0_real > hi)).float().mean()),
            "real_x0_std": float(diff.clamp_x0(x0_real).std()),
            "real_x0_corr_with_clean": corr(diff.clamp_x0(x0_real), real),
        })
        if i == steps - 1:
            x = x0
        else:
            ab_next = diff._ab(ts[i + 1].repeat(n), x)
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    for row, x0 in zip(rows, x0s):
        row["x0_corr_with_final"] = corr(x0, x.cpu())
    return {"checkpoint": str(ckpt_path), "step": ckpt.get("step"),
            "lambda_smooth": cfg["train"].get("lambda_smooth"),
            "final_mean": float(x.mean()), "final_std": float(x.std()),
            "real_std": float(real.std()), "trace": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mean-x0-below", type=float, default=0.0,
                    help="use x0 = data mean at steps with alpha_bar below this")
    ap.add_argument("--out", default="experiments/dit_64/results/diag_ddim_trace.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [trace(Path(c), args.n, args.steps, args.seed, device,
                     args.mean_x0_below) for c in args.ckpt]
    for r in results:
        print(f"\n{r['checkpoint']} step={r['step']} lambda_smooth={r['lambda_smooth']} "
              f"final mean={r['final_mean']:.3f} std={r['final_std']:.3f} (real {r['real_std']:.3f})")
        print(" step    t     abar  x0raw_std  clamped  x0_std  corr_final | real: clamped  x0_std  corr_clean")
        for row in r["trace"]:
            if row["step"] < 8 or row["step"] % 7 == 0 or row["step"] == args.steps - 1:
                print(f" {row['step']:4d} {row['t']:4d} {row['alpha_bar']:8.2e} {row['x0_raw_std']:10.2f}"
                      f" {row['x0_frac_clamped']:8.3f} {row['x0_std']:7.3f} {row['x0_corr_with_final']:10.3f}"
                      f" | {row['real_x0_frac_clamped']:13.3f} {row['real_x0_std']:7.3f} {row['real_x0_corr_with_clean']:10.3f}")
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
