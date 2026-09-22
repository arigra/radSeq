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



# ---- linking vehicles across the frames of a sequence -----------------------

def test_vehicles_are_linked_across_frames_into_stable_slots():
    """RADIal labels each frame on its own, with no vehicle identity. The DiT's
    condition needs one slot per vehicle over the whole sequence, so labels are
    linked frame to frame by proximity."""
    frames = [[(200.0, 10.0), (400.0, 40.0)],      # two vehicles
              [(202.0, 11.0), (403.0, 41.0)],      # both moved slightly
              [(205.0, 12.0)]]                     # the far one left the view
    traj, present = radial.link_vehicles(frames)
    assert present[0].tolist() == [True, True, True]
    assert present[1].tolist() == [True, True, False]
    assert traj[0, :, 0].tolist() == [200.0, 202.0, 205.0]
    assert traj[1, 1, 0] == 403.0


def test_a_distant_detection_starts_a_new_track():
    frames = [[(100.0, 10.0)], [(400.0, 200.0)]]
    traj, present = radial.link_vehicles(frames)
    assert present[0].tolist() == [True, False]
    assert present[1].tolist() == [False, True]


def test_doppler_gate_wraps_around_the_axis():
    """Doppler is circular: bin 255 and bin 1 are two bins apart, not 254."""
    frames = [[(200.0, 255.0)], [(200.0, 1.0)]]
    traj, present = radial.link_vehicles(frames)
    assert present[0].tolist() == [True, True]      # one vehicle, not two
