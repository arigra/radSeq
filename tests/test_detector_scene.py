"""Vehicle detector for RADIal-grid sequences (512x256 power maps)."""
import torch

from src.detector_scene import RadialVehicleDetector, scene_heatmap_targets


def test_targets_mark_present_vehicles_only():
    traj = torch.zeros(1, 2, 2, 2)
    traj[0, 0] = torch.tensor([[100.0, 40.0], [102.0, 41.0]])
    traj[0, 1] = torch.tensor([[300.0, 80.0], [300.0, 80.0]])
    present = torch.tensor([[[True, True], [True, False]]])
    h = scene_heatmap_targets(traj, present, n_range=512, n_doppler=256)
    assert h.shape == (1, 2, 512, 256)
    assert float(h[0, 0, 100, 40]) == 1.0
    assert float(h[0, 1, 102, 41]) == 1.0
    assert float(h[0, 0, 300, 80]) == 1.0
    assert float(h[0, 1, 300, 80]) == 0.0        # absent in frame 1


def test_targets_wrap_around_the_doppler_axis():
    traj = torch.zeros(1, 1, 1, 2)
    traj[0, 0, 0] = torch.tensor([100.0, 255.0])
    h = scene_heatmap_targets(traj, torch.ones(1, 1, 1, dtype=torch.bool),
                              n_range=512, n_doppler=256)
    assert float(h[0, 0, 100, 0]) > 0.5          # bin 0 is one step from 255


def test_detector_returns_one_heatmap_per_frame():
    torch.manual_seed(0)
    model = RadialVehicleDetector(width=8)
    out = model(torch.randn(2, 4, 512, 256))
    assert out.shape == (2, 4, 512, 256) and torch.isfinite(out).all()


def test_detector_is_per_frame_so_frames_do_not_leak():
    """Each frame is detected on its own; changing frame 1 must not change
    frame 0's output, or a detector could cheat by reading other frames."""
    torch.manual_seed(0)
    model = RadialVehicleDetector(width=8).eval()
    x = torch.randn(1, 2, 512, 256)
    y = x.clone()
    y[:, 1] += 5.0
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)


def test_average_precision_wraps_in_doppler():
    """Many RADIal vehicles sit at the Doppler axis edge. A detection at bin 255
    is two bins from a target at bin 1, not 254 -- without wrapping every such
    vehicle counts as a miss and every arm is scored unfairly."""
    from src.detector import average_precision
    from src.detector_scene import average_precision_wrapped
    dets = [[(0.9, 100.0, 255.0)]]
    tgts = [torch.tensor([[100.0, 1.0]])]
    assert average_precision(dets, tgts, radius=2.0) == 0.0          # unwrapped: miss
    assert average_precision_wrapped(dets, tgts, 256, radius=2.0) == 1.0


def test_wrapped_average_precision_still_rejects_far_detections():
    from src.detector_scene import average_precision_wrapped
    dets = [[(0.9, 100.0, 128.0)]]
    tgts = [torch.tensor([[100.0, 1.0]])]
    assert average_precision_wrapped(dets, tgts, 256, radius=2.0) == 0.0


def test_training_batches_are_frames_not_whole_sequences():
    """The detector is per-frame, and a batch of whole 512x256 sequences does
    not fit in GPU memory (it ran out overnight). Batches sample one frame at a
    time, and the target must be that frame's vehicles."""
    from scripts.radial_detector_arms import batch_from
    pool = [{"x": torch.zeros(8, 512, 256, dtype=torch.float16),
             "traj": torch.zeros(8, 8, 2), "present": torch.zeros(8, 8, dtype=torch.bool),
             "stats": {"mean": 0.0, "std": 1.0}}
            for _ in range(3)]
    pool[0]["traj"][0, 5] = torch.tensor([100.0, 40.0])
    pool[0]["present"][0, 5] = True
    x, heat = batch_from(pool, 4, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert x.shape == (4, 1, 512, 256)
    assert heat.shape == (4, 1, 512, 256)


def test_each_source_is_normalised_by_its_own_statistics():
    """Real maps sit near 97 dB and simulated ones near 12 dB. Normalising the
    simulated frames by the real statistics put them ~15 std off the real data,
    which made the real+sim arm meaningless."""
    from scripts.radial_detector_arms import batch_from
    def item(level, mean, std):
        return {"x": torch.full((8, 512, 256), level, dtype=torch.float16),
                "traj": torch.zeros(8, 8, 2), "present": torch.zeros(8, 8, dtype=torch.bool),
                "stats": {"mean": mean, "std": std}}
    pool = [item(97.0, 97.0, 5.6), item(12.5, 12.5, 5.5)]
    x, _ = batch_from(pool, 16, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert float(x.abs().max()) < 0.05


def test_every_vehicle_gets_exactly_one_positive_cell_even_off_grid():
    """The focal loss counts a cell as a vehicle only if its target is exactly
    1.0. RADIal ranges are fractional (e.g. 157.69 bins), so an unrounded
    Gaussian never reaches 1.0 -- 596 vehicles produced 1 positive cell, the
    detector learned to predict nothing, and every arm scored AP 0."""
    traj = torch.zeros(1, 2, 1, 2)
    traj[0, 0, 0] = torch.tensor([157.69, 40.0])
    traj[0, 1, 0] = torch.tensor([300.31, 254.6])
    h = scene_heatmap_targets(traj, torch.ones(1, 2, 1, dtype=torch.bool),
                              n_range=512, n_doppler=256)
    assert int(h.eq(1).sum()) == 2
    assert float(h[0, 0, 158, 40]) == 1.0
    assert float(h[0, 0, 300, 255]) == 1.0


def test_real_subsets_are_nested():
    """Step 5's learning curve compares sizes; each smaller subset must sit
    inside every larger one, so a size difference is only more data."""
    from scripts.make_radial_subset import subset_indices
    sizes = [25, 50, 100, 200, 387]
    sets = [set(subset_indices(387, n)) for n in sizes]
    assert [len(s) for s in sets] == sizes
    assert all(a <= b for a, b in zip(sets, sets[1:]))


def test_a_peak_across_the_doppler_seam_is_detected_once():
    """RADIal's Doppler axis wraps. With plain max-pool padding, a vehicle
    straddling bins 255 and 0 gave two local maxima, one of them a false alarm."""
    from src.detector import detect
    prob = torch.zeros(1, 32, 256)
    prob[0, 10, 255], prob[0, 10, 0], prob[0, 10, 1] = 0.8, 0.9, 0.3
    assert len(detect(prob, circular_doppler=True)[0]) == 1
    assert len(detect(prob)[0]) == 2          # the old behaviour, still the default


def test_sequence_detector_reads_all_frames_and_predicts_each():
    from src.detector_scene import RadialSequenceDetector
    det = RadialSequenceDetector(width=4, seq_len=8)
    x = torch.zeros(2, 8, 64, 32)
    assert det(x).shape == (2, 8, 64, 32)
    x2 = x.clone()
    x2[:, 0, 30, 10] = 5.0                     # a change in frame 0 reaches frame 7's output
    assert not torch.allclose(det(x)[:, 7], det(x2)[:, 7])


def test_batches_take_their_grid_from_the_data():
    """CARRADA maps are 256 x 64; targets built on RADIal's 512 x 256 grid
    would not match the detector output."""
    from scripts.radial_detector_arms import batch_from
    pool = [{"x": torch.zeros(8, 256, 64, dtype=torch.float16),
             "traj": torch.zeros(4, 8, 2), "present": torch.zeros(4, 8, dtype=torch.bool),
             "stats": {"mean": 0.0, "std": 1.0}}]
    pool[0]["traj"][0, :] = torch.tensor([100.0, 40.0])
    pool[0]["present"][0, :] = True
    x, heat = batch_from(pool, 4, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert x.shape == heat.shape == (4, 1, 256, 64)
    assert int(heat.eq(1).sum()) == 4
