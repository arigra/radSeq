import pytest
import torch
from src.simulator import TemporalRadarSimulator, create_rd_map


def test_output_shapes_and_keys():
    torch.manual_seed(0)
    sim = TemporalRadarSimulator(seq_len=16)
    out = sim.gen_sequence()
    M = out["n_targets"]
    assert out["x"].shape == (16, 64, 64) and out["x"].dtype == torch.float32
    assert out["traj"].shape == (M, 16, 2)
    assert out["v0"].shape == (M,) and out["acc"].shape == (M,)
    assert out["cls"].shape == (M,) and out["env"].shape == (3,)
    assert 1 <= M <= 5


def test_trajectory_within_grid():
    torch.manual_seed(1)
    sim = TemporalRadarSimulator(seq_len=16)
    for _ in range(10):
        out = sim.gen_sequence()
        assert (out["traj"][..., 0] >= 0).all() and (out["traj"][..., 0] <= 63).all()
        assert (out["traj"][..., 1] >= 0).all() and (out["traj"][..., 1] <= 63).all()


def test_peak_follows_trajectory():
    """Single steady point target (class forced to 0), no clutter/noise floor
    dominance: the RD peak must land within 1.5 bins of the continuous analytic
    trajectory (integer peak vs off-grid ground truth can differ by up to ~1.5
    bins at bin boundaries)."""
    torch.manual_seed(2)
    sim = TemporalRadarSimulator(seq_len=16, max_targets=1, scnr=20.0, force_class=0)
    out = sim.gen_sequence()
    assert out["n_targets"] == 1
    for l in range(16):
        frame = out["x"][l]
        idx = frame.flatten().argmax()
        r_pk, v_pk = (idx // 64).item(), (idx % 64).item()
        r_gt, v_gt = out["traj"][0, l]
        assert abs(r_pk - r_gt) <= 1.5, f"frame {l}: range {r_pk} vs {r_gt}"
        assert abs(v_pk - v_gt) <= 1.5, f"frame {l}: doppler {v_pk} vs {v_gt}"


def test_motion_magnitude():
    """Targets move, but no more than ~2 bins/frame in range."""
    torch.manual_seed(3)
    sim = TemporalRadarSimulator(seq_len=16)
    out = sim.gen_sequence()
    dr = (out["traj"][:, 1:, 0] - out["traj"][:, :-1, 0]).abs()
    assert dr.max() <= 2.0


def test_extended_target_flanks_vs_point():
    """Class 2 vs class 0, same on-grid position: on-grid adjacent range
    bins are exact nulls for a point target (bin-orthogonal steering), so
    flank energy at r±1 bins cleanly separates the range-extended class."""
    torch.manual_seed(4)
    sim = TemporalRadarSimulator(seq_len=1)
    r = torch.tensor([90.0])                      # on-grid: range bin 30
    v = torch.tensor([sim.V[40].item()])          # on-grid: doppler bin 40
    g = torch.tensor([10.0])

    def rd_db(cls_id):
        iq = sim._frame_targets(r, v, g, torch.tensor([cls_id]))
        rd = create_rd_map(iq)
        return 20 * torch.log10(rd.abs() + 1e-6)

    m0, m2 = rd_db(0), rd_db(2)
    flank0 = torch.maximum(m0[29, 40], m0[31, 40])
    flank2 = torch.maximum(m2[29, 40], m2[31, 40])
    assert m0[30, 40] - flank0 > 20, "point target should null adjacent bins"
    assert m2[30, 40] - flank2 < 8, "extended target flanks should be ~3 dB down"


def test_explicit_kinematics_places_target_exactly():
    """gen_sequence(r0=..., v0=..., a=...) must honor the requested
    constant-acceleration trajectory instead of sampling one, while the
    rest of the pipeline (clutter, noise, labels) works as usual."""
    torch.manual_seed(5)
    sim = TemporalRadarSimulator(seq_len=16, scnr=20.0, force_class=0)
    r0 = torch.tensor([90.0])
    v0 = torch.tensor([-3.0])
    a = torch.tensor([0.0])
    out = sim.gen_sequence(r0=r0, v0=v0, a=a)

    assert out["n_targets"] == 1
    assert torch.allclose(out["v0"], v0) and torch.allclose(out["acc"], a)
    ell = torch.arange(16, dtype=torch.float) * sim.Tf
    rb_expected = (r0 + v0 * ell - sim.r_min) / sim.dr
    vb_expected = (v0.expand(16) - sim.v_min) / sim.dv
    assert torch.allclose(out["traj"][0, :, 0], rb_expected, atol=1e-4)
    assert torch.allclose(out["traj"][0, :, 1], vb_expected, atol=1e-4)
    for l in range(16):
        idx = out["x"][l].flatten().argmax()
        r_pk, v_pk = (idx // 64).item(), (idx % 64).item()
        assert abs(r_pk - out["traj"][0, l, 0]) <= 1.5, f"frame {l}"
        assert abs(v_pk - out["traj"][0, l, 1]) <= 1.5, f"frame {l}"


def test_explicit_kinematics_multi_target_and_validation():
    """Multiple explicit targets are all honored; a trajectory that walks
    off the grid raises ValueError instead of silently clipping."""
    torch.manual_seed(6)
    sim = TemporalRadarSimulator(seq_len=16)
    out = sim.gen_sequence(r0=torch.tensor([60.0, 120.0]),
                           v0=torch.tensor([2.0, -2.0]),
                           a=torch.tensor([0.1, -0.1]))
    assert out["n_targets"] == 2
    assert out["traj"].shape == (2, 16, 2)

    with pytest.raises(ValueError):
        sim.gen_sequence(r0=torch.tensor([185.0]),
                         v0=torch.tensor([7.0]),
                         a=torch.tensor([0.0]))
