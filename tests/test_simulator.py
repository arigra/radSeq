import torch
from src.simulator import TemporalRadarSimulator


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
