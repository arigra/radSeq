"""Append the step 2 / step 3 sections to notebooks/radSeq.ipynb.

The notebook is Ari's reading surface: it has to make sense after days away,
so the new cells carry the findings and their caveats, not pipeline plumbing
(that lives in src/ and scripts/). Cells are executed before being written so
the notebook can be read without running anything.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "radSeq.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(r"""
## Step 2 — does generated data actually help?

A generator is only worth having if what it produces is *useful*. So far the
checks above only ask whether generated sequences look like simulator
sequences. This step asks the question that matters: **does adding generated
data make a detector better when real data is scarce?**

Setup, all on the simulator so the ground truth is exact:

- the generator saw only **2,000** sequences, roughly the scale of a real dataset;
- it then rendered **8,000** labelled sequences from the same trajectory labels;
- a detector must find each target *and say which class it is* (steady,
  Swerling-1, extended). The position-only task was already saturated —
  brightness alone scored AP 0.94 — so it could not show a difference either way.

The rules below were written down **before** the run
(`docs/notes/2026-09-17-step2-synthetic-augmentation.md`): the test counts as
informative only if 20,000 real sequences beat 2,000 by more than the seed
spread, and synthetic data counts as helping only if it clears the same bar.
"""),
    code(r"""
import json
from pathlib import Path

report = json.loads((ROOT / "samples" / "detector_class_augmentation.json").read_text())

print(f"{'training pool':<34}{'sequences':>10}{'mean mAP':>10}{'seed range':>12}")
for name, arm in report["arms"].items():
    print(f"{name:<34}{arm['pool_sequences']:>10}{arm['mean']:>10.3f}{arm['range']:>12.3f}")

rule = report["rule"]
print(f"\nheadroom (20k real - 2k real): {rule['headroom']:+.3f}  bar {rule['headroom_bar']:.3f}"
      f"  -> informative: {rule['informative']}")
print(f"gain (2k real + synthetic - 2k real): {rule['gain']:+.3f}  bar {rule['gain_bar']:.3f}"
      f"  -> helps: {rule['helps']}")
"""),
    md(r"""
**Synthetic data helped.** Adding it gained +0.040 mAP, about twice the bar,
and closed roughly 40% of the gap between 2,000 real sequences and the
20,000-real ceiling. The gain came from the two hard classes (steady and
Swerling-1, both ~0.51 → ~0.57); the extended class was already near 0.97 and
had nothing left to give.

The striking arm is `synth_only`: a detector that **never saw a real sequence**
(0.689) beat one trained on the 2,000 real ones (0.665). That is hard to
explain by copying alone.

### But the generator does copy

The same run checked whether the generator reproduces its training sequences.
It compares how close its output gets to a sequence it trained on, versus one
it never saw.
"""),
    code(r"""
mem = json.loads((ROOT / "samples" / "memorization_n2000.json").read_text())
for key in ("median_rms_training_labels", "median_rms_heldout_labels",
            "median_rms_unrelated_real_pairs", "ratio_training_over_heldout"):
    print(f"{key:<36}{mem[key]:.3f}")
print(f"\nmemorising (ratio < 0.8): {mem['memorising']}")
"""),
    md(r"""
Ratio **0.715**, below the 0.8 cutoff: with only 2,000 sequences the generator
is partly memorising. Under the pre-set rule this flags the positive result —
some of the gain may come from near-copies rather than genuinely new data.

So step 2 ends with a real but **caveated** positive, and one clear lesson for
the real-data work: *scarcity drives the generator to memorise*.

---

## Step 3 — real radar (RADIal)

Everything above is simulated. The question that decides whether any of this
matters is whether it survives contact with real radar.

[RADIal](https://github.com/valeoai/RADIal) is 91 recordings from a 77 GHz
imaging radar with 8,252 labelled frames. Each frame is a
(512 range × 256 Doppler × 16 receive antenna) complex tensor.

The first thing to do is simply **look at a real map next to a simulated one**.
"""),
    code(r"""
import numpy as np
from src import radial
from src.radar_physics import RADIAL_SPEC
from src.simulator import spec_simulator
from src.viz import show_real_vs_sim

by_frame = radial.read_labels()
sample = sorted(by_frame)[0]
real = radial.power_map(sample)

sim = spec_simulator(RADIAL_SPEC, seq_len=1)
torch.manual_seed(0)
sim_map = sim.gen_sequence(n_targets=2)["x"][0].numpy()

show_real_vs_sim(real, sim_map, targets=radial.targets(sample, by_frame),
                 title="one labelled RADIal frame (red = labelled vehicles) vs the simulator")
"""),
    md(r"""
### The dataset's own labels are mislabelled

Before anything could be compared, the grid had to be worked out from the data,
because two columns in `labels.csv` do not contain what their names say:

- **`radar_D_mps` is an integer Doppler *bin index*, not metres per second.**
- **`radar_P_db` holds linear power (~1.4e7), not dB.**

Taking either at face value would have silently corrupted everything
downstream. The range axis *is* what it claims: exactly 0.2 m per bin.

The check below confirms the grid is right — each labelled vehicle should sit
on a local peak at the predicted cell.
"""),
    code(r"""
peaks = []
for s in sorted(by_frame)[:40]:
    p = radial.power_map(s)
    for rb, db in radial.targets(s, by_frame):
        ri, di = int(round(rb)), int(round(db))
        if 2 <= ri < p.shape[0] - 2 and 2 <= di < p.shape[1] - 2:
            peaks.append(p[ri-2:ri+3, di-2:di+3].max() - np.median(p[ri-2:ri+3, :]))

peaks = np.array(peaks)
print(f"{len(peaks)} labelled vehicles")
print(f"peak above its range ring:  p10 {np.percentile(peaks,10):.1f} dB   "
      f"median {np.median(peaks):.1f} dB   p90 {np.percentile(peaks,90):.1f} dB")
"""),
    md(r"""
A median of ~24 dB above the surrounding range ring: the vehicles really are
where the labels say, so the grid is correct.

### Real data is far scarcer than the rehearsal assumed

Our sequences are 16 frames long, and only frames that are *consecutive* can
form one. That turns out to be the binding constraint on the whole study.

Recordings are split **whole**, never by frame: neighbouring frames of one
drive are nearly identical, so a frame-level split would measure memorisation
rather than transfer. The split is also balanced by the number of sequences
each recording contributes — recordings differ ~20× in length, and assigning
them independently left validation with 6 sequences out of 219.
"""),
    code(r"""
split = json.loads((ROOT / "data" / "radial" / "split.json").read_text())
recs = radial.recordings()

print(f"{'split':<8}{'recordings':>12}{'frames':>9}{'16-frame seq':>14}{'8-frame seq':>13}")
total16 = 0
for name in ("train", "val", "test"):
    sub = {k: recs[k] for k in split[name]}
    n16 = len(radial.sequences(sub, seq_len=16))
    n8 = len(radial.sequences(sub, seq_len=8))
    total16 += n16
    print(f"{name:<8}{len(sub):>12}{sum(len(v) for v in sub.values()):>9}{n16:>14}{n8:>13}")
print(f"\nindependent 16-frame sequences in all of RADIal: {total16}")
"""),
    md(r"""
**219 independent 16-frame sequences exist in the entire dataset** — 153 for
training. The step 2 rehearsal used 2,000 and already caught the generator
memorising. This is far scarcer, which is why the plan moved to 8-frame
sequences (387 training sequences) and why pretraining on simulated data is the
main defence rather than a nicety.

---

## Building the simulator from specifications, not by fitting it

A tempting shortcut is to tune the simulator until its maps match RADIal's.
**That is the wrong thing to do**, for two reasons:

1. It quietly spends real data on the simulator. The "simulator only" baseline
   would then secretly contain real information, and in a real deployment you
   would not have the data to fit with.
2. It turns the sim-to-real gap into a *minimised* quantity instead of a
   *measured* one — and that gap is exactly what the generator is supposed to
   close.

So every parameter comes from a published source: RADIal's own processing code
(`rpl.py`) and Table 5 of the CVPR 2022 paper. Nothing is fitted to the maps.
Two of those published numbers corrected earlier guesses — the radar runs at
**5 fps** (not the 0.5 s frame interval we had), and the map integrates
**16 receive channels**, which is a documented property, not a free parameter.
"""),
    code(r"""
s = RADIAL_SPEC
print(f"{'published':<26}{'value':>12}")
for label, value in (("carrier", f"{s.fc_hz/1e9:.0f} GHz"),
                     ("range resolution", f"{s.range_res_m} m"),
                     ("velocity resolution", f"{s.velocity_res_mps} m/s"),
                     ("frame rate", f"{s.frame_rate_hz} fps"),
                     ("range x Doppler bins", f"{s.n_range} x {s.n_doppler}"),
                     ("Tx / Rx antennas", f"{s.n_tx} / {s.n_rx}"),
                     ("FFT window", s.window)):
    print(f"{label:<26}{value:>12}")

print(f"\n{'derived':<26}{'value':>12}")
for label, value in (("bandwidth", f"{s.bandwidth_hz/1e6:.0f} MHz"),
                     ("chirp interval", f"{s.chirp_interval_s*1e6:.1f} us"),
                     ("frame interval", f"{s.frame_interval_s} s"),
                     ("max range", f"{s.max_range_m:.1f} m"),
                     ("integrated looks", f"{s.n_looks}")):
    print(f"{label:<26}{value:>12}")
"""),
    md(r"""
### The measured gap — and a bug it exposed

With nothing fitted, the residual against **held-out** recordings is the honest
sim-to-real gap. It is large:
"""),
    code(r"""
gap = json.loads((ROOT / "samples" / "sim2real_gap.json").read_text())
print(f"{'statistic':<30}{'real':>9}{'sim':>9}{'gap':>9}")
for key, value in gap["gap"].items():
    print(f"{key:<30}{gap['real'][key]:>9.2f}{gap['sim'][key]:>9.2f}{value:>+9.2f}")
"""),
    md(r"""
Part of this is the genuine gap we want to measure. **Most of it is a modelling
bug**, and the two must not be confused:

- The range-profile gap of +56 dB is exactly `30·log10(102.2/0.2)` — a surface
  clutter law applied from 0.2 m outward, where the antenna does not illuminate
  the ground at all. The target-prominence gap largely follows from it.
- A real engineer building from a datasheet would produce something
  *uncalibrated*, not something whose targets are 39 dB too bright. Leaving it
  broken would make the low-fidelity baseline a strawman.

So this is a bug to fix (antenna elevation geometry), **not** a fidelity
setting to tune against the data. The honest gap is whatever remains after the
physics is right.

---

## The actual experiment: simulator fidelity as the variable

This reframes the study. Instead of building one simulator and hoping it is
good enough, **fidelity becomes the independent variable**. For each simulator,
a DiT is pretrained on it and fine-tuned on the small real set, and we compare:

| arm | training pool |
|---|---|
| `A` | real only |
| `B` | real + **raw simulator** data |
| `D` | real + data from a **DiT pretrained on that simulator, fine-tuned on real** |

`B` and `D` are matched in pool size, so the comparison isolates *where* the
extra sequences came from, not how many.

**The result is an interaction, not a single number.** `B` should degrade as
the simulator gets worse. If `D` stays flat, the claim is:

> **Generative fine-tuning substitutes for simulator fidelity** — you do not
> need a carefully calibrated simulator if you have a generative model and a
> little real data.

That is more useful, and more surprising, than "synthetic data helps". It is
also falsifiable: **if `D` tracks `B`, the mechanism is dead.**

### Rehearsing it sim-to-sim first

RADIal offers only 153–387 real training sequences, so the arms may not
separate at all. Before spending days of GPU time, the same experiment runs on
the 64×64 grid where the recipe is already proven — and there the mismatch can
be dialled **continuously**, giving a curve instead of a few scattered points.

One simulator configuration is designated "reality"; the others are wrong by a
controlled dB offset in target brightness, the dominant mismatch measured
above.
"""),
    code(r"""
from src.viz import show_fidelity_curve

path = ROOT / "samples" / "fidelity_pilot.json"
if not path.exists():
    print("pilot still running - no results yet (scripts/run_fidelity_pilot.sh)")
else:
    pilot = json.loads(path.read_text())
    done = sorted(pilot["deltas"], key=float)
    print(f"{'mismatch dB':>12}{'A real':>9}{'B raw sim':>11}{'D dit':>9}{'D - B':>8}")
    arms = {"real": [], "real_sim": [], "real_synth": []}
    for key in done:
        a = pilot["deltas"][key]["detector"]["arms"]
        for name in arms:
            arms[name].append(a[name]["mean"])
        print(f"{key:>12}{a['real']['mean']:>9.3f}{a['real_sim']['mean']:>11.3f}"
              f"{a['real_synth']['mean']:>9.3f}"
              f"{a['real_synth']['mean'] - a['real_sim']['mean']:>+8.3f}")
    if len(done) > 1:
        show_fidelity_curve([float(d) for d in done], arms)
"""),
    md(r"""
### Where this stands

- **Done:** a DiT that generates radar sequences passing all four checks; a
  trajectory-conditioned version that puts targets where asked; evidence that
  its output helps a detector when data is scarce (flagged for memorisation).
- **Done:** RADIal decoded, its grid verified, recordings split, and a
  simulator built from published specifications with its gap measured.
- **Running:** the sim-to-sim fidelity pilot, to measure the effect size.
- **Next:** fix the simulator's clutter geometry, then run the two-variant
  comparison on real RADIal data.

The claim to aim at is the interaction — not that synthetic data helps, but
that **a generator makes simulator fidelity matter less**.
"""),
]


def main():
    nb = json.loads(NB.read_text())
    marker = "## Step 2 — does generated data actually help?"
    keep = [c for c in nb["cells"]
            if marker not in "".join(c["source"])]
    # drop anything previously appended after the marker, so re-running is safe
    if len(keep) != len(nb["cells"]):
        idx = next(i for i, c in enumerate(nb["cells"])
                   if marker in "".join(c["source"]))
        keep = nb["cells"][:idx]
    nb["cells"] = keep + copy.deepcopy(CELLS)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"notebook now has {len(nb['cells'])} cells "
          f"({len(CELLS)} appended)")


if __name__ == "__main__":
    main()
