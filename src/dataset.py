"""Pre-generate, cache, and load temporal radar sequences.

Shards: <split>_<idx:04d>.pt, each a list of per-sequence dicts from
TemporalRadarSimulator.gen_sequence() with target fields padded to
MAX_TARGETS. stats.pt holds train-split mean/std of the log-magnitude
maps; both splits are normalized with it.
"""
import math
from pathlib import Path

import torch
import yaml

from src.simulator import TemporalRadarSimulator

MAX_TARGETS = 5


def _pad(item):
    m = item["n_targets"]
    out = dict(item)
    out["traj"] = torch.zeros(MAX_TARGETS, item["traj"].shape[1], 2)
    out["traj"][:m] = item["traj"]
    for k in ("v0", "acc"):
        out[k] = torch.zeros(MAX_TARGETS)
        out[k][:m] = item[k]
    out["cls"] = torch.zeros(MAX_TARGETS, dtype=torch.long)
    out["cls"][:m] = item["cls"]
    out["n_targets"] = torch.tensor(m)
    return out


def generate_cache(cache_dir, n_train, n_val, seq_len=16, seed=1234,
                   shard_size=1000, frame_interval=0.5):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    sim = TemporalRadarSimulator(seq_len=seq_len, frame_interval=frame_interval)

    def write_split(split, n):
        n_shards = math.ceil(n / shard_size)
        for si in range(n_shards):
            count = min(shard_size, n - si * shard_size)
            shard = [_pad(sim.gen_sequence()) for _ in range(count)]
            torch.save(shard, cache / f"{split}_{si:04d}.pt")

    write_split("train", n_train)
    # stats from the train shards only
    total, total_sq, count = 0.0, 0.0, 0
    for f in sorted(cache.glob("train_*.pt")):
        for item in torch.load(f):
            x = item["x"]
            total += x.sum().item()
            total_sq += (x ** 2).sum().item()
            count += x.numel()
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean ** 2, 1e-12))
    torch.save({"mean": mean, "std": std}, cache / "stats.pt")

    write_split("val", n_val)
    with open(cache / "manifest.yaml", "w") as fh:
        yaml.safe_dump({"n_train": n_train, "n_val": n_val, "seq_len": seq_len,
                        "seed": seed, "shard_size": shard_size,
                        "frame_interval": frame_interval,
                        "mean": mean, "std": std}, fh)


def denormalize(x, stats):
    return x * stats["std"] + stats["mean"]


class RadarSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, split):
        cache = Path(cache_dir)
        self.stats = torch.load(cache / "stats.pt")
        self.items = []
        for f in sorted(cache.glob(f"{split}_*.pt")):
            self.items.extend(torch.load(f))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        item["x"] = (item["x"] - self.stats["mean"]) / self.stats["std"]
        return item
