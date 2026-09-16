"""At which noise level do the extra blips and broken target tracks appear?

Noise real validation sequences to timestep t, denoise back to 0 with DDIM
(EMA weights), and score with the evaluator. Persistence staying at the real
level from small t but dropping from large t locates the failure in the
high-noise (global/temporal) stage; a drop already at small t locates it in the
low-noise refinement stage, which the shifted schedule trains least.
"""
import argparse
import json
from pathlib import Path

import torch

from evaluate_ablations import load_raw_validation
from src.dataset import denormalize
from src.diffusion import DEFAULT_X0_CLAMP, GaussianDiffusion
from src.eval.metrics import evaluate_sequences
from src.sample import select_checkpoint_state
from src.train import build_model


@torch.no_grad()
def ddim_from(diff, model, x, t_start, steps):
    ts = torch.linspace(t_start, 0, steps, device=x.device).long()
    for i in range(steps):
        t = ts[i].repeat(len(x))
        eps, x0 = diff.to_eps_x0(model(x, t), x, t)
        eps, x0 = diff.terminal_step(x, t, eps, diff.clamp_x0(x0))
        if i == steps - 1:
            return x0
        ab = diff._ab(ts[i + 1].repeat(len(x)), x)
        x = ab.sqrt() * x0 + (1 - ab).sqrt() * eps


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/e3_standard_bs32/last.pt")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--timesteps", default="0,50,150,300,500,700,900,999")
    ap.add_argument("--out", default="samples/diag_persistence_by_t.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg, dc = ck["config"], ck["config"]["diffusion"]
    diff = GaussianDiffusion(dc["timesteps"], x0_clamp=dc.get("x0_clamp", DEFAULT_X0_CLAMP),
                             parameterization=dc.get("parameterization", "eps"),
                             terminal_x0=dc.get("terminal_x0", "model"),
                             schedule_shift=dc.get("schedule_shift", 1.0))
    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ck, weights="ema"))
    model.eval()
    cache = Path(cfg["data"]["cache_dir"])
    stats = torch.load(cache / "stats.pt", map_location="cpu")
    n = args.n
    real_all = load_raw_validation(cache, 4 * n)
    src_raw, ref = real_all[:3 * n], real_all[3 * n:]           # denoise 96, compare to 32 others
    src = ((src_raw - stats["mean"]) / stats["std"]).to(device)
    rows = {"real_input_vs_ref": evaluate_sequences(src_raw, ref)}
    print("real input", {k: round(float(v), 3) for k, v in rows["real_input_vs_ref"].items()}, flush=True)
    for t0 in (int(s) for s in args.timesteps.split(",")):
        outs = []
        for s in range(0, len(src), n):
            b = src[s:s + n]
            g = torch.Generator(device=device).manual_seed(100 + s)
            if t0 == 0:
                outs.append(b.cpu()); continue
            t = torch.full((len(b),), t0, device=device, dtype=torch.long)
            xt = diff.q_sample(b, t, torch.randn(b.shape, device=device, generator=g))
            steps = max(2, round(50 * t0 / 999))
            outs.append(ddim_from(diff, model, xt, t0, steps).float().cpu())
        x = denormalize(torch.cat(outs), stats)
        m = evaluate_sequences(x, ref)
        m["alpha_bar"] = float(diff.alphas_bar[min(t0, 999)])
        rows[f"t{t0}"] = m
        print(f"t={t0:4d} abar={m['alpha_bar']:.3f}",
              {k: round(float(m[k]), 3) for k in ("persistence", "mean_tracks_per_seq",
                                                  "n_target_tracks_per_seq", "marginal_l1",
                                                  "velocity_consistency")}, flush=True)
    Path(args.out).write_text(json.dumps({"checkpoint": args.ckpt, "weights": "ema", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
