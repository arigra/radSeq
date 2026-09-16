import torch

from src.trajectory_condition import COND_CHANNELS, drop_condition, render_condition


def _one_target(r, d, cls_id, n_frames=2):
    traj = torch.zeros(1, 5, n_frames, 2)
    traj[0, 0, :, 0], traj[0, 0, :, 1] = r, d
    return traj, torch.tensor([1]), torch.tensor([[cls_id, 0, 0, 0, 0]])


def test_blob_peaks_at_requested_bin_in_its_class_channel():
    traj, n, cls = _one_target(20.0, 41.0, cls_id=2)
    c = render_condition(traj, n, cls)
    assert c.shape == (1, 2, COND_CHANNELS, 64, 64)
    peak = int(c[0, 0, 2].argmax())
    assert (peak // 64, peak % 64) == (20, 41)
    assert float(c[0, 0, 2, 20, 41]) == 1.0
    assert float(c[0, 0, 0].max()) == 0.0 and float(c[0, 0, 1].max()) == 0.0


def test_padded_targets_are_ignored():
    traj, n, cls = _one_target(20.0, 41.0, cls_id=0)
    traj[0, 1, :, 0], traj[0, 1, :, 1] = 50.0, 10.0      # slot 1 is padding (n_targets = 1)
    assert float(render_condition(traj, n, cls)[0, 0, 0, 50, 10]) < 1e-6


def test_presence_channel_is_ones():
    traj, n, cls = _one_target(5.0, 5.0, cls_id=1)
    assert torch.equal(render_condition(traj, n, cls)[:, :, 3], torch.ones(1, 2, 64, 64))


def test_drop_condition_zeroes_whole_sequences():
    torch.manual_seed(0)
    cond = torch.ones(200, 2, COND_CHANNELS, 4, 4)
    per_seq = drop_condition(cond, 0.5).flatten(1)
    assert torch.equal(per_seq.min(1).values, per_seq.max(1).values)   # all or nothing
    assert 60 < int((per_seq.max(1).values == 0).sum()) < 140
    assert torch.equal(drop_condition(cond, 0.0), cond)
