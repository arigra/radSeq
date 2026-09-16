"""Score a trajectory-conditioned DiT against the pre-set rule; append the verdict.

Rule (docs/notes/2026-09-17-trajectory-conditioning.md): passes if hit rate >=
simulator hit rate - 0.05, unrequested lasting tracks/seq <= simulator's, and
Ari's four checks pass; the unconditional model on the same requests must hit
at least 0.20 less. Guidance is swept on val 0..3n-1 with seeds 1-3; the
verdict uses val 3n..9n-1 with seeds 4-9.
"""
import argparse
import json
from pathlib import Path

import torch

from score_hard_standard import checks
from src.dataset import RadarSequenceDataset
from src.eval.adherence import trajectory_adherence
from src.eval.metrics import evaluate_sequences
from src.sample import generate, generate_trajectory_conditioned, resolve_cache_dir


def labels_and_maps(items, offset, n):
    chunk = items[offset:offset + n]
    labels = {key: torch.stack([it[key] for it in chunk]) for key in ("traj", "n_targets", "cls")}
    return labels, torch.stack([it["x"] for it in chunk]).float()


def run_requests(ckpt, items, plan, n, device, guidance):
    """plan: list of (val offset, seed); returns concatenated labels, real maps, generated maps."""
    labels, real, gen = [], [], []
    for offset, seed in plan:
        lab, x = labels_and_maps(items, offset, n)
        labels.append(lab)
        real.append(x)
        gen.append(generate_trajectory_conditioned(ckpt, lab, device, steps=30,
                                                   guidance=guidance, weights="ema", seed=seed))
    labels = {key: torch.cat([lab[key] for lab in labels]) for key in labels[0]}
    return labels, torch.cat(real), torch.cat(gen)


def score(gen, real, labels, real_ref, real_metrics, stats):
    adh = trajectory_adherence(gen, labels["traj"], labels["n_targets"])
    sim = trajectory_adherence(real, labels["traj"], labels["n_targets"])
    m = evaluate_sequences(gen, real_ref)
    m["std"] = float(((gen - stats["mean"]) / stats["std"]).std())
    rule = {"hit_rate": adh["hit_rate"] >= sim["hit_rate"] - 0.05,
            "unrequested_tracks": adh["unrequested_lasting_tracks_per_seq"]
                                  <= sim["unrequested_lasting_tracks_per_seq"],
            **checks(m, real_metrics)}
    return {"adherence": adh, "simulator": sim, "metrics": m, "rule": rule,
            "passes": all(rule.values())}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--uncond-ckpt", required=True)
    ap.add_argument("--guidance", default="1,2,3")
    ap.add_argument("--n", type=int, default=32, help="sequences per seed")
    ap.add_argument("--out", default="samples/cond_traj_scores.json")
    ap.add_argument("--note")
    ap.add_argument("--png")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = args.n
    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["config"]
    cache = resolve_cache_dir(cfg["data"]["cache_dir"])
    items = RadarSequenceDataset(cache, "val").items
    stats = torch.load(cache / "stats.pt", map_location="cpu")
    _, ref_a = labels_and_maps(items, 0, 32)
    _, ref_b = labels_and_maps(items, 32, 32)
    real_metrics = evaluate_sequences(ref_a, ref_b)
    real_metrics["std"] = float(((ref_b - stats["mean"]) / stats["std"]).std())
    report = {"checkpoint": args.ckpt, "n_per_seed": n, "real_vs_real": real_metrics, "sweep": {}}

    sweep_plan = [((s - 1) * n, s) for s in (1, 2, 3)]
    best = None
    for w in (float(v) for v in args.guidance.split(",")):
        labels, real, gen = run_requests(args.ckpt, items, sweep_plan, n, device, w)
        result = score(gen, real, labels, ref_b, real_metrics, stats)
        report["sweep"][str(w)] = result
        print(f"w={w}", json.dumps({k: result[k] for k in ("adherence", "simulator", "rule", "passes")}),
              flush=True)
        key = (result["passes"], result["adherence"]["hit_rate"])
        if best is None or key > best[0]:
            best = (key, w)
    w = best[1]

    final_plan = [(3 * n + (s - 4) * n, s) for s in range(4, 10)]
    labels, real, gen = run_requests(args.ckpt, items, final_plan, n, device, w)
    final = score(gen, real, labels, ref_b, real_metrics, stats)
    null_gen = torch.cat([generate(args.uncond_ckpt, n, device, steps=30, weights="ema", seed=s)
                          for _, s in final_plan])
    null = trajectory_adherence(null_gen, labels["traj"], labels["n_targets"])
    final.update(guidance=w, null_unconditional=null,
                 null_check_ok=null["hit_rate"] <= final["adherence"]["hit_rate"] - 0.20)
    report["final"] = final
    print("FINAL", json.dumps({k: final[k] for k in ("guidance", "adherence", "simulator", "rule",
                                                     "passes", "null_unconditional", "null_check_ok")}),
          flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.viz import show_rows
        show_rows([{"x": real, **labels}, {"x": gen, **labels}],
                  ["simulator (requested scene)", f"conditional DiT, w={w}"],
                  title="Requested targets (red circles): simulator vs conditional DiT")
        plt.savefig(args.png, dpi=70)

    if args.note:
        ok = lambda b: "ok" if b else "FAIL"
        s, a, m, rr, rule = (final["simulator"], final["adherence"], final["metrics"],
                             real_metrics, final["rule"])
        sweep = ", ".join(f"w={k}: hit {v['adherence']['hit_rate']:.3f}, passes {v['passes']}"
                          for k, v in report["sweep"].items())
        lines = [
            "", f"## Result (`{args.out}`)", "",
            f"Checkpoint `{args.ckpt}`. Guidance w = {w}, chosen on val 0-{3 * n - 1} / seeds 1-3 "
            f"({sweep}). Verdict on val {3 * n}-{9 * n - 1}, seeds 4-9, {6 * n} sequences.",
            "", "| check | simulator / real | conditional DiT | |", "|---|---:|---:|---|",
            f"| hit rate | {s['hit_rate']:.3f} | {a['hit_rate']:.3f} | {ok(rule['hit_rate'])} |",
            f"| unrequested lasting tracks / seq | {s['unrequested_lasting_tracks_per_seq']:.2f} | "
            f"{a['unrequested_lasting_tracks_per_seq']:.2f} | {ok(rule['unrequested_tracks'])} |",
            f"| std | {rr['std']:.3f} | {m['std']:.3f} | {ok(rule['std'])} |",
            f"| marginal L1 | {rr['marginal_l1']:.3f} | {m['marginal_l1']:.3f} | {ok(rule['marginal_l1'])} |",
            f"| target tracks / seq | {rr['n_target_tracks_per_seq']:.2f} | "
            f"{m['n_target_tracks_per_seq']:.2f} | {ok(rule['target_tracks'])} |",
            f"| persistence | {rr['persistence']:.3f} | {m['persistence']:.3f} | {ok(rule['persistence'])} |",
            "", f"Null check (unconditional e3_long, same requests and seeds): hit rate "
            f"{final['null_unconditional']['hit_rate']:.3f} -> "
            + ("ok (conditional hit rate is at least 0.20 higher)" if final["null_check_ok"] else
               "not met: the conditional model does not hit at least 0.20 more than the unconditional one"),
            "", "**Verdict (pre-set rule): "
            + ("the conditional DiT follows requested trajectories.**"
               if final["passes"] and final["null_check_ok"] else
               "the conditional DiT does NOT yet pass; see the failing checks.**"),
        ]
        with open(args.note, "a") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
