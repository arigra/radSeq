from src.eval.adherence import trajectory_adherence
from src.simulator import generate_sequences


def test_simulator_sequences_follow_their_own_labels():
    d = generate_sequences(n=8, seed=0)
    own = trajectory_adherence(d["x"], d["traj"], d["n_targets"])
    shifted = trajectory_adherence(d["x"], d["traj"] + 10.0, d["n_targets"])
    assert own["hit_rate"] > 0.8
    assert shifted["hit_rate"] < 0.2


def test_lasting_tracks_away_from_requests_count_as_unrequested():
    d = generate_sequences(n=8, n_targets=1, target_class="steady", snr_db=20,
                           clutter=False, noise=True, seed=0)
    own = trajectory_adherence(d["x"], d["traj"], d["n_targets"])
    far = trajectory_adherence(d["x"], d["traj"] + 20.0, d["n_targets"])
    assert own["unrequested_lasting_tracks_per_seq"] <= 0.5
    assert far["unrequested_lasting_tracks_per_seq"] >= 1.0
