import torch

from src.detector import (HeatmapDetector, average_precision, detect, focal_loss,
                          heatmap_targets)


def _labels():
    traj = torch.zeros(1, 5, 2, 2)
    traj[0, 0, :, 0], traj[0, 0, :, 1] = 20.4, 41.6        # rounds to (20, 42)
    traj[0, 1, :, 0], traj[0, 1, :, 1] = 50.0, 10.0        # padding (n_targets = 1)
    return traj, torch.tensor([1])


def test_heatmap_peaks_at_rounded_target_bin_and_ignores_padding():
    traj, n = _labels()
    h = heatmap_targets(traj, n)
    assert h.shape == (1, 2, 64, 64)
    assert float(h[0, 0, 20, 42]) == 1.0
    assert float(h[0, 0, 50, 10]) < 1e-6


def test_detector_output_shape_and_loss_is_finite():
    torch.manual_seed(0)
    model = HeatmapDetector()
    x = torch.randn(3, 64, 64)
    logits = model(x)
    assert logits.shape == (3, 64, 64)
    target = torch.zeros(3, 64, 64)
    target[:, 10, 10] = 1.0
    assert torch.isfinite(focal_loss(logits, target))


def test_perfect_heatmap_scores_ap_one_and_shifted_scores_zero():
    traj, n = _labels()
    perfect = heatmap_targets(traj, n)[0]                  # (2, 64, 64) as probabilities
    targets = [torch.round(traj[0, :1, l]) for l in range(2)]
    dets = detect(perfect)
    assert average_precision(dets, targets, radius=2.0) == 1.0
    shifted = torch.roll(perfect, shifts=10, dims=-1)
    assert average_precision(detect(shifted), targets, radius=2.0) == 0.0


def test_average_precision_penalises_false_positives_ranked_first():
    targets = [torch.tensor([[10.0, 10.0]])]
    dets = [[(0.9, 40.0, 40.0), (0.5, 10.0, 10.0)]]         # (score, range, doppler) per frame
    assert abs(average_precision(dets, targets, radius=2.0) - 0.5) < 1e-6


def test_class_heatmap_routes_each_target_to_its_class_channel():
    from src.detector import class_heatmap_targets
    traj, n = _labels()
    n = torch.tensor([2])
    cls = torch.tensor([[2, 0, 0, 0, 0]])                 # target 0 extended, target 1 steady
    h = class_heatmap_targets(traj, n, cls)
    assert h.shape == (1, 3, 2, 64, 64)
    assert float(h[0, 2, 0, 20, 42]) == 1.0 and float(h[0, 0, 0, 20, 42]) < 1e-6
    assert float(h[0, 0, 1, 50, 10]) == 1.0 and float(h[0, 2, 1, 50, 10]) < 1e-6


def test_sequence_class_detector_shape_and_loss():
    from src.detector import SequenceClassDetector
    torch.manual_seed(0)
    model = SequenceClassDetector()
    logits = model(torch.randn(2, 16, 64, 64))
    assert logits.shape == (2, 3, 16, 64, 64)
    target = torch.zeros(2, 3, 16, 64, 64)
    target[:, 1, :, 30, 30] = 1.0
    assert torch.isfinite(focal_loss(logits, target))
