"""Evidence gathering for the Phase-1 marginal_l1 gap (0.436 vs a 0.065 floor).

Answers, with numbers rather than argument:
  1. What are the real vs generated intensity distributions, exactly?
  2. Does the DDIM x0 clamp (+-4 sigma, diffusion.py:42) truncate the real range?
  3. Is marginal_l1's shared min/max range being widened by outliers?
  4. What IS the real-vs-real floor at the protocol actually used (16 gen vs 64 real)?
     The committed samples/reference_metrics.yaml has no recorded recipe.

Writes a JSON report; prints a summary. Read-only w.r.t. the repo.
"""
import argparse
import json
from pathlib import Path

import torch

from src.eval.metrics import marginal_l1

PCTS = [0.0, 0.1, 1.0, 25.0, 50.0, 75.0, 99.0, 99.9, 100.0]


def load_raw_val(cache_dir, limit=None):
    """Raw (un-normalized, dB) val sequences straight from the shards."""
    seqs = []
    for f in sorted(Path(cache_dir).glob("val_*.pt")):
        for item in torch.load(f):
            seqs.append(item["x"])
            if limit is not None and len(seqs) >= limit:
                return torch.stack(seqs)
    return torch.stack(seqs)


def describe(x, name):
    flat = x.flatten().float()
    q = torch.tensor([p / 100.0 for p in PCTS])
    # torch.quantile caps input size; subsample deterministically if needed.
    sub = flat if flat.numel() <= 16_000_000 else flat[:: flat.numel() // 16_000_000 + 1]
    vals = torch.quantile(sub, q).tolist()
    return {
        "name": name,
        "n_values": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "percentiles": {str(p): round(v, 4) for p, v in zip(PCTS, vals)},
    }


def robust_marginal_l1(x_gen, x_real, bins=64, lo_pct=0.1, hi_pct=99.9):
    """marginal_l1 over a percentile range of the REAL data, with both sides
    clipped into it. Removes the shared-min/max widening effect so a genuine
    shape mismatch can be told apart from a few extreme outliers."""
    r = x_real.flatten().float()
    sub = r if r.numel() <= 16_000_000 else r[:: r.numel() // 16_000_000 + 1]
    lo, hi = torch.quantile(sub, torch.tensor([lo_pct / 100, hi_pct / 100])).tolist()
    g = x_gen.flatten().float().clamp(lo, hi)
    rr = r.clamp(lo, hi)
    hg = torch.histc(g, bins=bins, min=lo, max=hi)
    hr = torch.histc(rr, bins=bins, min=lo, max=hi)
    return float((hg / hg.sum() - hr / hr.sum()).abs().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/phase1_wandb/best.pt")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--n-real", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--floor-draws", type=int, default=8)
    ap.add_argument("--out", default="samples/diag_marginal_l1.json")
    args = ap.parse_args()

    report = {"config": vars(args)}
    stats = torch.load(Path(args.cache) / "stats.pt")
    mean, std = stats["mean"], stats["std"]
    report["norm_stats"] = {"mean": float(mean), "std": float(std)}

    # ---- boundary 1: what the real data actually looks like -------------
    need = args.n_real + args.floor_draws * args.n_gen + args.n_gen
    real_all = load_raw_val(args.cache, limit=need)
    print(f"loaded {real_all.shape[0]} real val sequences {tuple(real_all.shape)}")
    x_real = real_all[: args.n_real]
    report["real"] = describe(x_real, "real_dB")

    # In normalized space, where does real data sit relative to the +-4 clamp?
    real_norm = (x_real - mean) / std
    report["real_normalized"] = describe(real_norm, "real_normalized")
    frac_out = float((real_norm.abs() > 4).float().mean())
    report["real_frac_beyond_4sigma"] = frac_out
    report["clamp_bounds_dB"] = {"lo": float(mean - 4 * std), "hi": float(mean + 4 * std)}

    # ---- boundary 2: the real-vs-real floor at THIS protocol ------------
    # Disjoint draws, same sizes as the gen comparison, so the floor is
    # measured the way the generated number is measured.
    floors = []
    for d in range(args.floor_draws):
        start = args.n_real + d * args.n_gen
        held = real_all[start:start + args.n_gen]
        if held.shape[0] < args.n_gen:
            break
        floors.append(marginal_l1(held, x_real))
    report["real_vs_real_floor"] = {
        "draws": floors,
        "mean": sum(floors) / len(floors) if floors else None,
        "min": min(floors) if floors else None,
        "max": max(floors) if floors else None,
    }
    print(f"real-vs-real floor ({args.n_gen} vs {args.n_real}): "
          f"{report['real_vs_real_floor']['mean']:.4f} "
          f"(range {min(floors):.4f}-{max(floors):.4f})")

    # ---- boundary 3: generated data ------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.diffusion import GaussianDiffusion
    from src.train import build_model

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diff = GaussianDiffusion(cfg["diffusion"]["timesteps"])
    L = cfg["data"]["seq_len"]
    print(f"sampling {args.n_gen} sequences, {args.steps} DDIM steps on {device}...")
    x_gen_norm = diff.ddim_sample(model, (args.n_gen, L, 64, 64), device,
                                  steps=args.steps).cpu()
    x_gen = x_gen_norm * std + mean

    report["gen_normalized"] = describe(x_gen_norm, "gen_normalized")
    report["gen"] = describe(x_gen, "gen_dB")
    # How much generated mass sits exactly at the clamp boundary?
    at_clamp = float((x_gen_norm.abs() >= 3.999).float().mean())
    report["gen_frac_at_clamp"] = at_clamp

    # ---- boundary 4: the metric itself ---------------------------------
    shared = marginal_l1(x_gen, x_real)
    robust = robust_marginal_l1(x_gen, x_real)
    robust_floor = [robust_marginal_l1(real_all[args.n_real + d * args.n_gen:
                                                args.n_real + (d + 1) * args.n_gen],
                                       x_real)
                    for d in range(len(floors))]
    report["marginal_l1"] = {
        "gen_vs_real_shared_range": shared,
        "gen_vs_real_robust_range": robust,
        "floor_robust_mean": sum(robust_floor) / len(robust_floor) if robust_floor else None,
    }

    # Histogram shapes, for eyeballing where the mass actually differs.
    lo = min(x_gen.min().item(), x_real.min().item())
    hi = max(x_gen.max().item(), x_real.max().item())
    hg = torch.histc(x_gen.flatten(), bins=64, min=lo, max=hi)
    hr = torch.histc(x_real.flatten(), bins=64, min=lo, max=hi)
    edges = torch.linspace(lo, hi, 65)
    report["histograms"] = {
        "bin_edges": [round(v, 3) for v in edges.tolist()],
        "gen_frac": [round(v, 6) for v in (hg / hg.sum()).tolist()],
        "real_frac": [round(v, 6) for v in (hr / hr.sum()).tolist()],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "histograms"}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
