import pytest
from src import radial


def test_split_is_by_recording_and_disjoint():
    recs = {f"REC{i:03d}": list(range(100)) for i in range(200)}
    s = radial.split_recordings(recs)
    all_names = s["train"] + s["val"] + s["test"]
    assert len(all_names) == len(set(all_names)) == 200
    assert 0.1 < len(s["test"]) / 200 < 0.3


def test_split_is_deterministic():
    recs = {f"REC{i:03d}": list(range(40)) for i in range(60)}
    assert radial.split_recordings(recs) == radial.split_recordings(recs)


def test_split_balances_sequences_not_recording_counts():
    """Recordings differ ~20x in length. Hashing each independently put 6 of
    219 sequences in val -- too few to early-stop on. The split must weight
    recordings by the sequences they contribute."""
    recs = {"LONG": list(range(320))}
    recs.update({f"S{i:02d}": list(range(16)) for i in range(20)})
    s = radial.split_recordings(recs, seq_len=16)
    counts = {k: len(radial.sequences({n: recs[n] for n in v}, seq_len=16))
              for k, v in s.items()}
    total = sum(counts.values())
    assert 0.1 <= counts["test"] / total <= 0.35, counts
    assert counts["val"] >= 1, counts


def test_sequences_are_consecutive_and_non_overlapping_by_default():
    recs = {"A": [0, 1, 2, 3, 4, 5, 9, 10, 11]}
    out = radial.sequences(recs, seq_len=3)
    assert [s for _, s in out] == [[0, 1, 2], [3, 4, 5], [9, 10, 11]]
    for _, s in out:
        assert s == list(range(s[0], s[0] + 3))


def test_sequences_skip_runs_that_are_too_short():
    assert radial.sequences({"A": [0, 1, 2, 7]}, seq_len=4) == []


def test_overlapping_windows_are_opt_in():
    out = radial.sequences({"A": list(range(6))}, seq_len=4, stride=1)
    assert [s[0] for _, s in out] == [0, 1, 2]


def test_fitted_simulator_matches_the_radial_grid():
    import torch
    sim, cfg = radial.fitted_simulator(seq_len=2)
    assert (sim.N, sim.K) == (512, 256)
    assert sim.n_looks == cfg["n_looks"] and sim.range_gain_db is not None
    x = sim.gen_sequence(n_targets=1,
                         gain_db=radial.target_gain_draw(1, cfg))["x"]
    assert x.shape == (2, 512, 256) and torch.isfinite(x).all()
