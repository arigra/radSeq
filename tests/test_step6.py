import random

import numpy as np
import torch

from src.detector_scene import average_precision_wrapped


def test_bootstrap_ap_reproduces_the_reference_ap():
    """step6_score scores frames once and recomputes AP on resampled frames;
    on the full set it must equal the AP every earlier step reported."""
    from scripts.step6_score import ap_from, frame_scores
    rng = random.Random(0)
    dets, tgts = [], []
    for _ in range(60):
        tg = torch.tensor([[rng.uniform(0, 512), rng.uniform(0, 256)]
                           for _ in range(rng.randint(0, 3))]).reshape(-1, 2)
        fd = [(rng.random(), float(t[0]) + rng.uniform(-3, 3), (float(t[1]) + rng.uniform(-3, 3)) % 256)
              for t in tg] + [(rng.random(), rng.uniform(0, 512), rng.uniform(0, 256))
                              for _ in range(rng.randint(0, 4))]
        fd.append((0.5, 10.0, 10.0))          # ties across frames
        dets.append(fd)
        tgts.append(tg)
    ref = average_precision_wrapped(dets, tgts, 256, radius=2.0)
    assert abs(ap_from(frame_scores(dets, tgts)) - ref) < 1e-12


def test_off_label_peaks_ignore_the_labelled_vehicle_and_its_ddma_copies():
    from src.eval.scene_realism import off_label_peaks
    x = np.zeros((1, 64, 256), np.float32)
    for k in range(0, 256, 16):                # a vehicle and its DDMA copies
        x[0, 30, (40 + k) % 256] = 30.0
    x[0, 50, 100 + 3] = 30.0                   # off-slot clutter peak
    traj = torch.tensor([[[30.0, 40.0]]])
    present = torch.ones(1, 1, dtype=torch.bool)
    assert off_label_peaks(x, traj, present) == 1


def test_speckle_and_frame_change_are_zero_on_a_flat_map():
    from src.eval.scene_realism import frame_change_db, speckle_db
    x = np.full((3, 16, 16), 97.0, np.float32)
    assert speckle_db(x) == 0.0 and frame_change_db(x) == 0.0


def test_vehicle_persistence_is_one_for_a_static_vehicle_and_lower_when_it_flickers():
    from src.eval.scene_realism import vehicle_persistence
    rng = np.random.default_rng(0)
    blob = rng.normal(size=(9, 5)).astype(np.float32)
    x = np.zeros((3, 64, 32), np.float32)
    for t in range(3):
        x[t, 26:35, 8:13] = blob
    traj = torch.tensor([[[30.0, 10.0]] * 3])
    present = torch.ones(1, 3, dtype=torch.bool)
    assert abs(vehicle_persistence(x, traj, present) - 1.0) < 1e-6
    x[1, 26:35, 8:13] = rng.normal(size=(9, 5))
    assert vehicle_persistence(x, traj, present) < 0.5
