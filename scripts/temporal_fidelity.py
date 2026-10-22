"""Do generated RADIal sequences behave like real ones over time? (step 7)

Compares the real training sequences of a cache with the DiT's renderings of
the same label sets (first part of a gen_scene_synthetic output), so the labels
are identical and only the maps differ: frame-to-frame change of the whole map
and the frame-to-frame correlation of each vehicle's window, plus the four
single-frame realism statistics. Reported, not a gate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.scene_realism import realism, temporal
from src.scene_data import SceneSequenceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--generated", required=True, help="dir with part_0.pt of the real labels")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = SceneSequenceDataset(a.real, "train", normalise=False)
    gen = torch.load(Path(a.generated) / "part_0.pt")
    n = min(len(ds), len(gen["x"]))
    real_items = [{"x": np.asarray(ds.x[i]), "traj": ds.traj[i], "present": ds.present[i]}
                  for i in range(n)]
    gen_items = [{"x": gen["x"][i].float().numpy(), "traj": gen["traj"][i],
                  "present": gen["present"][i]} for i in range(n)]
    for r, g in zip(real_items, gen_items):
        assert torch.equal(torch.as_tensor(r["traj"]), torch.as_tensor(g["traj"]))
    out = {"n": n, "real": {**temporal(real_items), **realism(real_items)},
           "generated": {**temporal(gen_items), **realism(gen_items)}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
