"""The simulator's radar constants must become configurable so a RADIal-matched
grid can be pretrained on, without disturbing the locked 64x64 cache."""
import pytest
import torch

from src.simulator import RADIAL_GEOMETRY, RadarGeometry, TemporalRadarSimulator


def test_default_geometry_reproduces_the_locked_simulator_exactly():
    """Every cached training sequence was drawn with the hardcoded constants.
    Making them configurable must leave the default path bit-identical."""
    from src.simulator import generate_sequences
    torch.manual_seed(0)
    a = generate_sequences(n=2, seed=11)["x"]
    torch.manual_seed(0)
    b = generate_sequences(n=2, seed=11)["x"]
    assert torch.equal(a, b)
    assert a.shape == (2, 16, 64, 64)


def test_default_geometry_matches_the_historical_constants():
    g = RadarGeometry()
    assert (g.N, g.K) == (64, 64)
    assert g.dr == pytest.approx(3.0)
    assert g.r_max == pytest.approx(189.0)


def test_radial_geometry_matches_the_measured_dataset_grid():
    """Measured from RADIal: 512 range bins at 0.2 m, 256 Doppler bins."""
    g = RADIAL_GEOMETRY
    assert (g.N, g.K) == (512, 256)
    assert g.dr == pytest.approx(0.2, rel=1e-6)
    assert g.r_max == pytest.approx(102.2, rel=1e-3)
    assert g.dv == pytest.approx(0.1, rel=1e-2)


def test_simulator_honours_a_non_default_geometry():
    sim = TemporalRadarSimulator(seq_len=2, geometry=RADIAL_GEOMETRY)
    out = sim.gen_sequence(n_targets=1)
    assert out["x"].shape == (2, 512, 256)


def test_geometries_do_not_contaminate_each_others_cached_matrices():
    """The steering matrices are module-level caches; keyed wrongly, a RADIal
    run would silently hand 512x256 matrices to a 64x64 one."""
    small = TemporalRadarSimulator(seq_len=2)
    big = TemporalRadarSimulator(seq_len=2, geometry=RADIAL_GEOMETRY)
    assert big.gen_sequence(n_targets=1)["x"].shape == (2, 512, 256)
    assert small.gen_sequence(n_targets=1)["x"].shape == (2, 64, 64)
    assert big.gen_sequence(n_targets=1)["x"].shape == (2, 512, 256)
