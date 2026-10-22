"""Step 6 verdicts: AP differences between arms, with bootstrap confidence
intervals over the test recordings.

Pre-registered in docs/notes/2026-10-22-step6-fixed-protocol.md. Seeds measure
only training noise; the 121 test sequences come from 13 recordings, and which
recordings happen to be in the test set is a second, larger source of
uncertainty. Each bootstrap draw resamples the 13 recordings with replacement,
recomputes every arm's AP (mean over its seeds) on the drawn frames, and takes
the paired difference. "X > Y" needs the 95% interval of X - Y above zero.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

COMPARISONS = [   # (x, y, role)
    ("dit_mix", "sim_mix", "primary"), ("dit_mix", "real", "primary"),
    ("sim_mix", "real", "primary"),
    ("dit_pt", "sim_pt", "secondary"), ("dit_pt", "real", "secondary"),
    ("sim_pt", "real", "secondary"),
    ("dit_only", "sim_only", "secondary"), ("dit_mix", "ditrl_mix", "secondary"),
]
N_BOOT = 1000


def frame_scores(dets, tgts, n_doppler=256, radius=2.0):
    """Per frame: (scores, is_tp) of its detections and its number of true
    vehicles, matched exactly as src.detector_scene.average_precision_wrapped."""
    out = []
    for frame_dets, tgt in zip(dets, tgts):
        used = torch.zeros(len(tgt), dtype=torch.bool)
        s, tp = [], []
        for score, r, d in sorted(frame_dets, key=lambda t: -t[0]):
            hit = -1
            if len(tgt):
                dr = tgt[:, 0] - r
                dd = torch.remainder(tgt[:, 1] - d + n_doppler / 2, n_doppler) - n_doppler / 2
                dist = torch.sqrt(dr ** 2 + dd ** 2)
                dist[used] = float("inf")
                best = int(torch.argmin(dist))
                if float(dist[best]) <= radius:
                    hit = best
            if hit >= 0:
                used[hit] = True
            s.append(score)
            tp.append(hit >= 0)
        out.append((np.array(s, np.float64), np.array(tp, bool), len(tgt)))
    return out


def ap_from(frames):
    """All-point interpolated AP over a list of frame_scores entries."""
    n_true = sum(f[2] for f in frames)
    if n_true == 0:
        return float("nan")
    s = np.concatenate([f[0] for f in frames])
    tp = np.concatenate([f[1] for f in frames])
    # stable sort on -score keeps detection order on ties, as list.sort does
    order = np.argsort(-s, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp)
    recall = ctp / n_true
    precision = ctp / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * envelope))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="radial_detector_study.py output")
    ap.add_argument("--dets-dir", default="samples/step6_dets")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text())
    variant, cache = report["variant"], Path(report["real_cache"]).name

    per_arm = {}
    recs = None
    for name, arm in report["arms"].items():
        tag = ("seq_" if report.get("detector") == "sequence" else "") + \
              ("" if name == "real" else f"{variant}_")
        runs = []
        for seed in sorted(arm["runs"], key=int):
            d = torch.load(Path(a.dets_dir) / f"{tag}{cache}_{name}_s{seed}.pt")
            if recs is None:
                recs = d["recs"]
            runs.append(frame_scores(d["dets"], d["tgts"]))
        per_arm[name] = runs
    recs = np.array(recs)
    recordings = sorted(set(recs))
    frames_of = {r: np.nonzero(recs == r)[0] for r in recordings}

    def arm_ap(name, idx):
        return float(np.mean([ap_from([run[i] for i in idx]) for run in per_arm[name]]))

    full = np.arange(len(recs))
    point = {n: arm_ap(n, full) for n in per_arm}
    rng = np.random.default_rng(0)
    boot = {n: [] for n in per_arm}
    for _ in range(N_BOOT):
        drawn = rng.choice(recordings, size=len(recordings), replace=True)
        idx = np.concatenate([frames_of[r] for r in drawn])
        for n in per_arm:
            boot[n].append(arm_ap(n, idx))
    boot = {n: np.array(v) for n, v in boot.items()}

    verdicts = []
    for x, y, role in COMPARISONS:
        if x not in per_arm or y not in per_arm:
            continue
        diff = boot[x] - boot[y]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        verdict = (f"{x} > {y}" if lo > 0 else f"{x} < {y}" if hi < 0
                   else "not resolved (95% interval spans 0)")
        verdicts.append({"comparison": f"{x} - {y}", "role": role,
                         "difference": point[x] - point[y], "ci95": [float(lo), float(hi)],
                         "verdict": verdict})
    result = {"report": a.report, "n_recordings": len(recordings), "n_boot": N_BOOT,
              "test_ap": point,
              "test_ap_ci95": {n: [float(v) for v in np.percentile(boot[n], [2.5, 97.5])]
                               for n in per_arm},
              "seeds": {n: len(v) for n, v in per_arm.items()}, "comparisons": verdicts}
    Path(a.out).write_text(json.dumps(result, indent=2))
    for v in verdicts:
        print(f"{v['role']:9s} {v['comparison']:22s} {v['difference']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['verdict']}")


if __name__ == "__main__":
    main()
