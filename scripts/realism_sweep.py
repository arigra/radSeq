"""Step 8: choose the most realistic fine-tuned generator, then gate it on data
the choice never saw.

docs/notes/2026-10-24-step8-realistic-finetuning.md. Full fine-tuning on 50
real sequences smooths away clutter as it goes (clutter-peak error 15% at
~500 steps, 34% at 800, 68% at 9,600), and validation loss does not see it, so
the snapshot is chosen by realism instead. To keep the gate an independent
check, real val is split: candidates (snapshot x EMA/raw weights) are scored on
val sequences [0, n), the winner is gated on [n, 2n). Sampler fixed: DDIM 250,
eta 1 (best in every sweep so far). References (e.g. the pretrained model
before any fine-tuning) are scored and reported but never chosen.
Resumable: scored candidates are kept in the output.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.scene_realism import mismatch, realism, with_median
from src.sample import load_trajectory_conditioned, sample_trajectory_conditioned
from src.scene_data import SceneSequenceDataset

ACCEPT = 0.25


def items_of(ds, idx):
    return [{"x": ds.x[i], "traj": ds.traj[i], "present": ds.present[i]} for i in idx]


def generated_stats(path, weights, ds, idx, device, steps, eta, **peaks_kw):
    model, cfg = load_trajectory_conditioned(path, device, weights=weights)
    labels = {"traj": ds.traj[idx], "present": ds.present[idx]}
    maps = torch.cat([sample_trajectory_conditioned(
        model, cfg, {k: v[s:s + 4] for k, v in labels.items()}, device,
        steps=steps, guidance=1.0, seed=77 + s, eta=eta) for s in range(0, len(idx), 4)])
    del model
    torch.cuda.empty_cache()
    gen = [{"x": maps[j].numpy(), "traj": labels["traj"][j], "present": labels["present"][j]}
           for j in range(len(idx))]
    return with_median(realism(gen, **peaks_kw), gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="label=path, a candidate")
    ap.add_argument("--reference", action="append", default=[], help="label=path, reported only")
    ap.add_argument("--cache", required=True, help="real cache; its val split is used")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--weights", default="ema,raw", help="which weights to score")
    ap.add_argument("--ddma-slot", type=int, default=16, help="0: no DDMA copies (CARRADA)")
    ap.add_argument("--peak-db", type=float, default=15.0)
    ap.add_argument("--spread", action="store_true",
                    help="halves from evenly spaced val sequences, alternating, instead of the first 2n")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    rep = json.loads(out.read_text()) if out.exists() else {"candidates": {}, "references": {}}
    val = SceneSequenceDataset(a.cache, "val", normalise=False)
    if a.spread:
        idx = np.linspace(0, len(val) - 1, 2 * a.n).astype(int).tolist()
        sel_idx, gate_idx = idx[0::2], idx[1::2]
    else:
        sel_idx, gate_idx = list(range(a.n)), list(range(a.n, 2 * a.n))
    kw = {"ddma_slot": a.ddma_slot or None, "peak_db": a.peak_db}
    real_sel = with_median(realism(items_of(val, sel_idx), **kw), items_of(val, sel_idx))
    real_gate = with_median(realism(items_of(val, gate_idx), **kw), items_of(val, gate_idx))
    rep.update(real_select=real_sel, real_gate=real_gate, steps=a.steps, eta=a.eta,
               select_val_idx=sel_idx, gate_val_idx=gate_idx, peaks=kw)

    def save():
        out.with_suffix(".tmp").write_text(json.dumps(rep, indent=2))
        out.with_suffix(".tmp").replace(out)

    for group, specs in (("references", a.reference), ("candidates", a.ckpt)):
        for spec in specs:
            label, path = spec.split("=", 1)
            for weights in a.weights.split(","):
                key = f"{label}_{weights}"
                if key in rep[group]:
                    continue
                gen = generated_stats(path, weights, val, sel_idx, device, a.steps, a.eta, **kw)
                rel, score = mismatch(gen, real_sel)
                rep[group][key] = {"ckpt": path, "weights": weights, "stats": gen,
                                   "relative_error": rel, "score": score}
                print(group, key, {k: round(v, 2) for k, v in rel.items()}, f"{score:.3f}", flush=True)
                save()
    best_key = min(rep["candidates"], key=lambda k: rep["candidates"][k]["score"])
    best = rep["candidates"][best_key]
    if rep.get("chosen", {}).get("key") != best_key:
        gen = generated_stats(best["ckpt"], best["weights"], val, gate_idx, device, a.steps, a.eta, **kw)
        rel, score = mismatch(gen, real_gate)
        rep["chosen"] = {"key": best_key, "ckpt": best["ckpt"], "weights": best["weights"],
                         "steps": a.steps, "eta": a.eta, "select_score": best["score"],
                         "gate_stats": gen, "gate_relative_error": rel, "gate_score": score,
                         "accepted": all(v <= ACCEPT for v in rel.values())}
        save()
    print("chosen", {k: v for k, v in rep["chosen"].items() if k != "gate_stats"}, flush=True)


if __name__ == "__main__":
    main()
