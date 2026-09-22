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
