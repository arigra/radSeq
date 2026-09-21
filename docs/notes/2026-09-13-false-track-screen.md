# One-experiment screen: false tracks (2026-09-13)

Question: does matching a per-cell clutter false-alarm rate also match the
number of false tracks? No, in this simulator.

`archive/codex_research/research_false_tracks.py` draws independent target-free clutter
sequences, calibrates a local-power threshold to nominal Pfa=0.01 on eight
training sequences per condition, then evaluates 32 new sequences per
condition. A false path is three consecutive detections with at most one
range and Doppler cell of movement per frame. The frame-shuffled control
preserves every observed frame and its cell detections but breaks temporal
relationships across sequences. Results: `archive/codex_research/results/research_false_tracks.json`.

| Texture nu | Rho | Observed cell Pfa | Three-frame paths | Frame-shuffled paths |
|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 0.00843 | 2130 | 45 |
| 0.1 | 0.5 | 0.00845 | 2325 | 51 |
| 0.1 | 0.9 | 0.00877 | 3480 | 69 |
| 1.3 | 0.1 | 0.01008 | 938 | 40 |
| 1.3 | 0.5 | 0.00987 | 1127 | 46 |
| 1.3 | 0.9 | 0.00958 | 2200 | 53 |

The observation is robust across both texture settings: raising rho increases
false paths substantially while cell Pfa changes little. Even rho=0.1 exceeds
the shuffled control because clutter texture and velocity remain fixed for
each sequence. Thus cell-tail calibration alone cannot validate a sequence
generator for tracker training.

This is a diagnostic, not yet a generative method result. The path counter is
a deliberately simple proxy tracker; it has no data association, velocity
model, or track-confirmation logic. The test uses only the simulator and no
measured radar. The next candidate should explicitly model or match track
statistics, and be judged against a baseline that preserves temporal clutter.
