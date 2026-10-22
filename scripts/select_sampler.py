"""Choose the sampler for a fine-tuned scene DiT by how real its maps look.

Pre-registered in docs/notes/2026-10-22-step6-fixed-protocol.md. Each candidate
(DDIM steps, eta, weights) renders the label sets of the first 16 real VAL
sequences; its maps are compared with those 16 real maps on speckle,
frame-to-frame change, brightest cells (above the median) and strong clutter
peaks off the labels. The candidate with the lowest mean relative error wins;
it is "accepted" if every statistic is within 25% of real. Val only: the test
recordings never shape the generator.
"""
import argparse
import json
from pathlib import Path

import torch

from src.eval.scene_realism import mismatch, realism, with_median
from src.sample import load_trajectory_conditioned, sample_trajectory_conditioned
from src.scene_data import SceneSequenceDataset

CANDIDATES = [   # (steps, eta, weights); the first is the step 4 sampler
    (30, 0.0, "ema"), (100, 1.0, "ema"), (250, 1.0, "ema"),
    (100, 1.0, "raw"), (250, 1.0, "raw")]
ACCEPT = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="real cache; its val split is used")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    report = json.loads(out.read_text()) if out.exists() else {"candidates": {}}

    val = SceneSequenceDataset(a.cache, "val", normalise=False)
    items = [{"x": val.x[i], "traj": val.traj[i], "present": val.present[i]}
             for i in range(a.n)]
    real = with_median(realism(items), items)
    report["real_val"] = real
    labels = {"traj": val.traj[:a.n], "present": val.present[:a.n]}
    models = {}
    for steps, eta, weights in CANDIDATES:
        key = f"ddim{steps}_eta{eta:g}_{weights}"
        if key in report["candidates"]:
            continue
        if weights not in models:
            models[weights] = load_trajectory_conditioned(a.ckpt, device, weights=weights)
        model, cfg = models[weights]
        maps = torch.cat([sample_trajectory_conditioned(
            model, cfg, {k: v[s:s + 4] for k, v in labels.items()}, device,
            steps=steps, guidance=1.0, seed=77 + s, eta=eta) for s in range(0, a.n, 4)])
        gen_items = [{"x": maps[i].numpy(), "traj": labels["traj"][i],
                      "present": labels["present"][i]} for i in range(a.n)]
        gen = with_median(realism(gen_items), gen_items)
        rel, score = mismatch(gen, real)
        report["candidates"][key] = {"steps": steps, "eta": eta, "weights": weights,
                                     "stats": gen, "relative_error": rel, "score": score}
        print(key, {k: round(v, 2) for k, v in rel.items()}, f"score {score:.3f}", flush=True)
        out.write_text(json.dumps(report, indent=2))
    best_key = min(report["candidates"], key=lambda k: report["candidates"][k]["score"])
    best = report["candidates"][best_key]
    report["chosen"] = {"key": best_key, "steps": best["steps"], "eta": best["eta"],
                        "weights": best["weights"],
                        "accepted": all(v <= ACCEPT for v in best["relative_error"].values())}
    out.write_text(json.dumps(report, indent=2))
    print("chosen", report["chosen"], flush=True)


if __name__ == "__main__":
    main()
