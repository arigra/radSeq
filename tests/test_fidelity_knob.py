"""The pilot needs a continuous, physically meaningful simulator-mismatch knob.

Target brightness is the right one: it is the dominant gap measured against
RADIal (+39 dB of target prominence for the specification-driven simulator),
and it maps onto a single dB offset.
"""
import torch

from src.simulator import TemporalRadarSimulator


def test_gain_offset_brightens_targets_without_touching_the_background():
    def run(offset):
        torch.manual_seed(0)
        sim = TemporalRadarSimulator(seq_len=2, gain_offset_db=offset)
        return sim.gen_sequence(r0=torch.tensor([90.0]), v0=torch.tensor([2.0]),
                                a=torch.tensor([0.0]), cls=torch.tensor([0]))["x"]

    base, bright = run(0.0), run(12.0)
    ri = int(90.0 / 3.0)
    peak_gain = float(bright[:, ri].max() - base[:, ri].max())
    assert 9.0 < peak_gain < 15.0, peak_gain
    # a band far from the target is background only, and must not move
    far = slice(0, 10)
    assert abs(float(bright[:, far].median() - base[:, far].median())) < 1.0


def test_zero_offset_is_the_unmodified_simulator():
    torch.manual_seed(3)
    a = TemporalRadarSimulator(seq_len=2, gain_offset_db=0.0).gen_sequence(n_targets=2)["x"]
    torch.manual_seed(3)
    b = TemporalRadarSimulator(seq_len=2).gen_sequence(n_targets=2)["x"]
    assert torch.equal(a, b)
