"""Score an E0 checkpoint under its own diffusion config; append the verdict to a note.

Uses the E0 metrics of experiments/dit_64/score_e0.py and the decision rule fixed in
docs/notes/2026-09-14-e0-dit-standard-setup.md: std within 0.1 of real,
marginal L1 <= 0.2, peak prominence >= 15 dB, >= 90% of sequences on a line.
"""
import argparse
import json
from pathlib import Path

import torch

from score_e0 import e0_metrics
from src.dataset import RadarSequenceDataset, denormalize
from src.diffusion import DEFAULT_X0_CLAMP, GaussianDiffusion
from src.sample import select_checkpoint_state
from src.train import build_model

KEYS = ("mean", "std", "peak_db_median", "peak_prominence_db_median",
        "frac_seq_on_line", "marginal_l1")


def works(m, real_std):
    return (abs(m["std"] - real_std) <= 0.1 and m["marginal_l1"] <= 0.2
            and m["peak_prominence_db_median"] >= 15 and m["frac_seq_on_line"] >= 0.9)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--steps", default="50,250")
    ap.add_argument("--out", default="experiments/dit_64/results/e0_standard_scores.json")
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
    val = RadarSequenceDataset(cfg["data"]["cache_dir"], "val")
    st, n = val.stats, args.n
    real = torch.stack([val.items[i]["x"] for i in range(2 * n)]).float()
    ra, rb = e0_metrics(real[:n], st), e0_metrics(real[n:], st)
    real_ref = {k: v for k, v in rb.items() if k != "_hist"}
    report = {"checkpoint": args.ckpt, "step": ck.get("step"), "config": cfg,
              "real": real_ref,
              "real_vs_real_marginal_l1": float((ra["_hist"] - rb["_hist"]).abs().sum()),
              "arms": {}}
    seeds = [int(s) for s in args.seeds.split(",")]
    shape = (n, cfg["data"]["seq_len"], 64, 64)
    shown = {"real": real[n:n + 1]}
    weights = ["raw"] + (["ema"] if ck.get("ema_model") is not None else [])
    for w in weights:
        model = build_model(cfg, device)
        model.load_state_dict(select_checkpoint_state(ck, weights=w))
        model.eval()
        for steps in (int(s) for s in args.steps.split(",")):
            per_seed = {}
            for seed in seeds:
                torch.manual_seed(seed)
                x = denormalize(diff.ddim_sample(model, shape, device, steps=steps).float().cpu(), st)
                m = e0_metrics(x, st)
                m["marginal_l1"] = float((m.pop("_hist") - rb["_hist"]).abs().sum())
                per_seed[seed] = m
                if seed == seeds[0]:
                    shown[f"{w} ddim{steps}"] = x[:1]
            avg = {k: sum(p[k] for p in per_seed.values()) / len(per_seed) for k in KEYS}
            name = f"{w}_ddim{steps}"
            report["arms"][name] = {"per_seed": per_seed, "mean_over_seeds": avg,
                                    "works": works(avg, real_ref["std"])}
            print(name, {k: round(v, 3) for k, v in avg.items()},
                  "WORKS" if report["arms"][name]["works"] else "fails", flush=True)
        del model
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frames = (0, 5, 10, 15)
        fig, ax = plt.subplots(len(shown), 4, figsize=(10, 2.5 * len(shown)))
        for i, (label, x) in enumerate(shown.items()):
            for j, f in enumerate(frames):
                ax[i, j].imshow(x[0, f].numpy(), vmin=-40, vmax=110, aspect="auto")
                ax[i, j].set_title(f"{label} f{f}", fontsize=8)
                ax[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(args.png, dpi=70)

    if args.note:
        any_works = any(a["works"] for a in report["arms"].values())
        lines = ["", f"## Standard recipe result (`{args.out}`)", "",
                 f"Checkpoint `{args.ckpt}`, step {ck.get('step')}, {n} sequences x seeds {args.seeds}."
                 " Recipe: `experiments/dit_64/configs/e0_standard.yaml` (v-pred, schedule shift 4, no clamp,"
                 " non-overlapping 8x8 patches, factorized attention, dim 384 x 12, EMA 0.9999,"
                 " no smoothness).",
                 "", "| arm | mean | std | peak dB | prominence dB | on line | marginal L1 | verdict |",
                 "|---|---:|---:|---:|---:|---:|---:|---|",
                 f"| real | {real_ref['mean']:.2f} | {real_ref['std']:.2f} | {real_ref['peak_db_median']:.0f} |"
                 f" {real_ref['peak_prominence_db_median']:.1f} | {real_ref['frac_seq_on_line']:.0%} |"
                 f" {report['real_vs_real_marginal_l1']:.3f} (floor) | |"]
        for name, a in report["arms"].items():
            m = a["mean_over_seeds"]
            lines.append(f"| {name} | {m['mean']:.2f} | {m['std']:.2f} | {m['peak_db_median']:.0f} |"
                         f" {m['peak_prominence_db_median']:.1f} | {m['frac_seq_on_line']:.0%} |"
                         f" {m['marginal_l1']:.3f} | {'WORKS' if a['works'] else 'fails'} |")
        lines += ["", "**Verdict (pre-set rule): "
                  + ("the standard recipe WORKS on E0; earlier failures were setup issues.**"
                     if any_works else
                     "the standard recipe does NOT work on E0; look for a radar-specific cause.**")]
        with open(args.note, "a") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
