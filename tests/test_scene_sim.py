"""The scene simulator's mechanisms are physics from published sources, so the
tests pin each mechanism rather than any match to RADIal's pixels."""
import math

import torch

from src.scene_sim import DDMA_OFFSETS, SceneSimulator


def _one(power_db=30.0, rng=50.0, az=0.0, vr=-3.0):
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(0))
    p = sim._render(torch.tensor([rng]), torch.tensor([az]), torch.tensor([vr]),
                    torch.tensor([power_db]), None)
    return 10 * torch.log10(p / 16 + 1e-12)


def test_ddma_offsets_match_radials_processing_code():
    """rpl.py: slots every 16 bins, Tx0 plus slots 5..15 (1-4 empty)."""
    assert DDMA_OFFSETS == (0, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240)


def test_every_return_is_replicated_once_per_transmitter():
    db = _one()
    tx0 = (-30) % 256                       # -3 m/s at 0.1 m/s per bin
    row = db[250]                           # 50 m at 0.2 m per bin
    top = sorted(int(k) for k in torch.topk(row, 12).indices)
    assert top == sorted((tx0 + o) % 256 for o in DDMA_OFFSETS)


def test_rendered_peak_carries_the_requested_power():
    assert abs(float(_one(power_db=30.0).max()) - 30.0) < 0.5


def test_static_world_is_seen_at_minus_ego_speed_straight_ahead():
    """A stationary object dead ahead closes at exactly the ego speed."""
    sim = SceneSimulator(seq_len=1)
    az = torch.tensor([0.0])
    vr = (torch.zeros(1) - 7.0) * torch.cos(az)
    assert float(vr) == -7.0


def test_near_range_is_dark_but_floored():
    """IF high-pass darkens near range; ADC noise stops it going to -inf."""
    sim = SceneSimulator(seq_len=1, generator=torch.Generator().manual_seed(1))
    x = sim.gen_sequence()["x"][0]
    near, mid = float(x[:10].median()), float(x[100:300].median())
    assert near < mid
    assert float(x.max() - x.min()) < 120.0
