"""Score a checkpoint on the full (locked) data; append the verdict to a note.

Decision rule (Ari, 2026-09-14; docs/notes/2026-09-14-hard-standard-recipe.md),
against held-out real data: normalised std within 0.1 of real, marginal L1
<= 0.15, target tracks/seq within 15% of real, persistence within 25% of real.
Metrics are src/eval/metrics.evaluate_sequences, as in every earlier arm.
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

KEYS = ("mean", "std", "marginal_l1", "n_target_tracks_per_seq", "persistence",
        "mean_tracks_per_seq", "velocity_consistency")


def checks(m, real):
    return {
        "std": abs(m["std"] - real["std"]) <= 0.1,
        "marginal_l1": m["marginal_l1"] <= 0.15,
        "target_tracks": abs(m["n_target_tracks_per_seq"] - real["n_target_tracks_per_seq"])
                         <= 0.15 * real["n_target_tracks_per_seq"],
        "persistence": abs(m["persistence"] - real["persistence"]) <= 0.25 * real["persistence"],
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--steps", default="50,250")
    ap.add_argument("--out", default="samples/e3_standard_scores.json")
    ap.add_argument("--note")
    ap.add_argument("--png")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg, dc = ck["config"], ck["config"]["diffusion"]
    diff = GaussianDiffusion(
        dc["timesteps"], x0_clamp=dc.get("x0_clamp", DEFAULT_X0_CLAMP),
        parameterization=dc.get("parameterization", "eps"),
        terminal_x0=dc.get("terminal_x0", "model"),
        schedule_shift=dc.get("schedule_shift", 1.0))
    cache = Path(cfg["data"]["cache_dir"])
    stats = torch.load(cache / "stats.pt", map_location="cpu")
    n = args.n
    real_all = load_raw_validation(cache, 2 * n)
    real_a, real_b = real_all[:n], real_all[n:]
    real_norm = (real_b - stats["mean"]) / stats["std"]
    real = dict(evaluate_sequences(real_a, real_b),
                mean=float(real_norm.mean()), std=float(real_norm.std()))
    report = {"checkpoint": args.ckpt, "step": ck.get("step"), "config": cfg,
              "real_vs_real": real, "arms": {}}
    seeds = [int(s) for s in args.seeds.split(",")]
    shape = (n, cfg["data"]["seq_len"], 64, 64)
    shown = {"real": real_b[:1]}
    weights = ["raw"] + (["ema"] if ck.get("ema_model") is not None else [])
    for w in weights:
        model = build_model(cfg, device)
        model.load_state_dict(select_checkpoint_state(ck, weights=w))
        model.eval()
        for steps in (int(s) for s in args.steps.split(",")):
            per_seed = {}
            for seed in seeds:
                torch.manual_seed(seed)
                z = diff.ddim_sample(model, shape, device, steps=steps).float().cpu()
                x = denormalize(z, stats)
                m = evaluate_sequences(x, real_b)
                m.update(mean=float(z.mean()), std=float(z.std()))
                per_seed[seed] = m
                if seed == seeds[0]:
                    shown[f"{w} ddim{steps}"] = x[:1]
            avg = {k: sum(float(p[k]) for p in per_seed.values()) / len(per_seed) for k in KEYS}
            c = checks(avg, real)
            name = f"{w}_ddim{steps}"
            report["arms"][name] = {"per_seed": per_seed, "mean_over_seeds": avg,
                                    "checks": c, "passes": all(c.values())}
            print(name, {k: round(v, 3) for k, v in avg.items()}, c, flush=True)
        del model
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(shown), 4, figsize=(10, 2.5 * len(shown)))
        lo, hi = float(real_b.quantile(0.01)), float(real_b.max())
        for i, (label, x) in enumerate(shown.items()):
            for j, f in enumerate((0, 5, 10, 15)):
                ax[i, j].imshow(x[0, f].numpy(), vmin=lo, vmax=hi, aspect="auto")
                ax[i, j].set_title(f"{label} f{f}", fontsize=8)
                ax[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(args.png, dpi=70)

    if args.note:
        ok = lambda b: "ok" if b else "FAIL"
        lines = ["", f"## Result (`{args.out}`)", "",
                 f"Checkpoint `{args.ckpt}`, step {ck.get('step')}, {n} sequences x seeds {args.seeds}.",
                 "", "| arm | std | marginal L1 | target tracks/seq | persistence | verdict |",
                 "|---|---:|---:|---:|---:|---|",
                 f"| real | {real['std']:.3f} | {real['marginal_l1']:.3f} (floor) |"
                 f" {real['n_target_tracks_per_seq']:.2f} | {real['persistence']:.3f} | |"]
        for name, a in report["arms"].items():
            m, c = a["mean_over_seeds"], a["checks"]
            lines.append(f"| {name} | {m['std']:.3f} {ok(c['std'])} | {m['marginal_l1']:.3f} {ok(c['marginal_l1'])} |"
                         f" {m['n_target_tracks_per_seq']:.2f} {ok(c['target_tracks'])} |"
                         f" {m['persistence']:.3f} {ok(c['persistence'])} | {'PASSES' if a['passes'] else 'fails'} |")
        any_pass = any(a["passes"] for a in report["arms"].values())
        lines += ["", "**Verdict (pre-set rule): "
                  + ("the DiT WORKS on the full data.**" if any_pass else
                     "the DiT does NOT yet pass on the full data; see which checks fail.**")]
        with open(args.note, "a") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
