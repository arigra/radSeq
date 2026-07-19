import torch
from src.losses import (soft_positions, doppler_centroids,
                        traj_consistency_loss, doppler_consistency_loss,
                        traj_loss_from_batch)


def _blob_sequence(traj_bins):
    """(M, L, 2) integer trajectory -> (1, L, 64, 64) map with blobs."""
    M, L, _ = traj_bins.shape
    rr, cc = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    frames = []
    for l in range(L):
        f = torch.zeros(64, 64)
        for m in range(M):
            r, c = traj_bins[m, l]
            f = f + 10 * torch.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 1.5)
        frames.append(f)
    return torch.stack(frames)[None]


def test_soft_positions_recover_blobs():
    ell = torch.arange(8, dtype=torch.float)
    traj = torch.stack([10 + ell, 20 + 0.5 * ell], dim=-1)[None]     # (1, 8, 2)
    x = _blob_sequence(traj)
    pos = soft_positions(x, traj[None], half=2, temp=0.5)            # (1, 1, 8, 2)
    assert (pos[0, 0] - traj[0]).abs().max() < 0.5


def test_traj_loss_zero_for_constant_velocity():
    ell = torch.arange(8, dtype=torch.float)
    traj = torch.stack([10 + ell, 20 + 0.5 * ell], dim=-1)[None][None]
    x = _blob_sequence(traj[0])
    pos = soft_positions(x, traj, half=2, temp=0.5)
    mask = torch.ones(1, 1)
    assert traj_consistency_loss(pos, mask).item() < 0.05


def test_gradients_flow():
    ell = torch.arange(8, dtype=torch.float)
    traj = torch.stack([10 + ell, 20 + 0.5 * ell], dim=-1)[None][None]
    x = _blob_sequence(traj[0]).requires_grad_(True)
    pos = soft_positions(x, traj, half=2, temp=0.5)
    cent = doppler_centroids(x, traj, half=2)
    mask = torch.ones(1, 1)
    (traj_consistency_loss(pos, mask)
     + doppler_consistency_loss(cent, mask)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_traj_loss_from_batch_masks_padding():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64, 64)
    batch = {
        "traj": torch.rand(2, 5, 8, 2) * 50 + 5,
        "n_targets": torch.tensor([1, 2]),
    }
    tr = {"lambda_traj": 0.01, "lambda_doppler": 0.01}
    loss = traj_loss_from_batch(x, batch, tr, torch.device("cpu"))
    assert torch.isfinite(loss) and loss.item() >= 0
