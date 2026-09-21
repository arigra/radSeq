import json

import torch

from src.scene_data import MAX_VEHICLES, SceneSequenceDataset, pack_labels


def test_labels_keep_vehicle_identity_when_one_leaves_the_view():
    lab = lambda i, r: {"id": i, "range_bin": r, "doppler_bin": 1.0}
    frames = [[lab(0, 10.0), lab(1, 50.0)],   # both visible
              [lab(1, 51.0)]]                 # vehicle 0 has left
    traj, present = pack_labels(frames, 2)
    assert present[0].tolist() == [True, False]
    assert present[1].tolist() == [True, True]
    assert traj[1, :, 0].tolist() == [50.0, 51.0]   # vehicle 1 stayed in slot 1


def test_cache_builds_resumes_and_loads(tmp_path, monkeypatch):
    import subprocess, sys
    out = tmp_path / "cache"
    cmd = [sys.executable, "scripts/build_scene_cache.py", "--variant", "engineer",
           "--n-train", "4", "--n-val", "2", "--seq-len", "2", "--out", str(out)]
    subprocess.run(cmd, check=True, env={"PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": "",
                                         "PATH": "/usr/bin:/bin"})
    first = SceneSequenceDataset(out, "train")[3]["x"].clone()
    # simulate an interruption: forget progress after 2 sequences and rebuild
    (out / "train_progress.json").write_text(json.dumps({"done": 2, "n": 4}))
    subprocess.run(cmd, check=True, env={"PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": "",
                                         "PATH": "/usr/bin:/bin"})
    ds = SceneSequenceDataset(out, "train")
    assert len(ds) == 4 and ds[0]["x"].shape == (2, 512, 256)
    diff = float((ds[3]["x"] - first).abs().max())
    assert diff == 0.0, diff                    # per-sequence seeds: reproducible
    assert ds.traj.shape == (4, MAX_VEHICLES, 2, 2)    # one label per map
