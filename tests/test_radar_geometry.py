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


def test_clutter_doppler_spread_is_fittable():
    """RADIal's background is flat in Doppler (ego motion smears static
    clutter); the shipped sigma_f=0.05 makes a narrow ridge on a 256-bin
    Doppler axis, so it has to be a fit parameter rather than a literal."""
    narrow = TemporalRadarSimulator(seq_len=2, sigma_f=0.01, noise=False)
    wide = TemporalRadarSimulator(seq_len=2, sigma_f=0.5, noise=False)
    torch.manual_seed(0)
    a = narrow.gen_sequence(n_targets=1)["x"]
    torch.manual_seed(0)
    b = wide.gen_sequence(n_targets=1)["x"]
    spread = lambda x: float(x.median(dim=0).values.median(dim=0).values.max()
                             - x.median(dim=0).values.median(dim=0).values.min())
    assert spread(b) < spread(a), (spread(a), spread(b))


def test_non_coherent_looks_suppress_speckle():
    """RADIal maps sum power over receive channels, which narrows the dB
    histogram. A single-look simulator spans ~120 dB where RADIal spans ~76,
    and no clutter setting closes that -- only integration does."""
    one = TemporalRadarSimulator(seq_len=2, geometry=RADIAL_GEOMETRY, n_looks=1)
    many = TemporalRadarSimulator(seq_len=2, geometry=RADIAL_GEOMETRY, n_looks=8)
    torch.manual_seed(0)
    a = one.gen_sequence(n_targets=1)["x"]
    torch.manual_seed(0)
    b = many.gen_sequence(n_targets=1)["x"]
    assert (b.max() - b.min()) < (a.max() - a.min()) - 20.0


def test_range_gain_profile_shapes_the_background():
    """The receiver's range response (blind zone, peak, far roll-off) is fitted
    from RADIal rather than assumed, so the simulator must accept it."""
    gain = torch.zeros(RADIAL_GEOMETRY.N)
    gain[:64] = -30.0
    sim = TemporalRadarSimulator(seq_len=2, geometry=RADIAL_GEOMETRY,
                                 range_gain_db=gain)
    x = sim.gen_sequence(n_targets=1)["x"]
    near = float(x[:, :64].median())
    far = float(x[:, 64:].median())
    assert near < far - 20.0, (near, far)


def test_default_geometry_ignores_looks_and_gain_by_default():
    sim = TemporalRadarSimulator(seq_len=2)
    assert sim.n_looks == 1 and sim.range_gain_db is None


def test_single_look_path_stays_bit_identical_to_the_shipped_simulator():
    """Pinned against sequences drawn before the geometry/multi-look rework.

    Every cached training sequence and every trained checkpoint came from this
    exact expression, so a silent drift here would invalidate them while all
    the behavioural tests still passed.
    """
    from pathlib import Path
    from src.simulator import generate_sequences
    ref = torch.load(Path(__file__).parent / "data" / "simulator_reference.pt")
    got = generate_sequences(n=1, seed=4321)
    assert torch.equal(got["x"][:, :2], ref["x"])
    assert torch.equal(got["traj"], ref["traj"])
