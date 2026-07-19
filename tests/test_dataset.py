import torch
from src.dataset import generate_cache, RadarSequenceDataset, denormalize


def test_cache_and_load(tmp_path):
    generate_cache(str(tmp_path), n_train=6, n_val=3, seq_len=16,
                   seed=7, shard_size=4)
    ds = RadarSequenceDataset(str(tmp_path), "train")
    assert len(ds) == 6
    item = ds[0]
    assert item["x"].shape == (16, 64, 64)
    assert item["traj"].shape == (5, 16, 2)
    assert item["v0"].shape == (5,) and item["cls"].shape == (5,)
    # normalized: roughly zero-mean unit-std over the split
    xs = torch.stack([ds[i]["x"] for i in range(len(ds))])
    assert xs.mean().abs() < 0.3 and (xs.std() - 1).abs() < 0.3


def test_val_uses_train_stats(tmp_path):
    generate_cache(str(tmp_path), n_train=6, n_val=3, seq_len=16,
                   seed=7, shard_size=4)
    tr = RadarSequenceDataset(str(tmp_path), "train")
    va = RadarSequenceDataset(str(tmp_path), "val")
    assert tr.stats == va.stats


def test_denormalize_roundtrip(tmp_path):
    generate_cache(str(tmp_path), n_train=2, n_val=1, seq_len=16,
                   seed=7, shard_size=4)
    ds = RadarSequenceDataset(str(tmp_path), "train")
    item = ds[0]
    raw = denormalize(item["x"], ds.stats)
    renorm = (raw - ds.stats["mean"]) / ds.stats["std"]
    assert torch.allclose(renorm, item["x"], atol=1e-5)


def test_collate_batches(tmp_path):
    generate_cache(str(tmp_path), n_train=4, n_val=1, seq_len=16,
                   seed=7, shard_size=4)
    ds = RadarSequenceDataset(str(tmp_path), "train")
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch["x"].shape == (2, 16, 64, 64)
    assert batch["traj"].shape == (2, 5, 16, 2)
