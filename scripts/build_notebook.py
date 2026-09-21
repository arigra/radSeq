"""Rebuild notebooks/radSeq.ipynb as one readable story.

The notebook grew by appending, so it read as a history of the project rather
than an explanation of it, and its only real-vs-simulated comparison was near
the end and used a simulator that has since been replaced. This rebuilds it:

  1  what real radar looks like          (everything is judged against it)
  2  simulators vs real                  Comparison 1
  3  the generator                       Comparison 2 + the four checks
  4  does generated data help?           Comparison 3
  5  the fidelity study
  6  where this stands
  A  appendix: how the 64x64 simulator builds a map   (Ari's walkthrough)
  B  appendix: the DiT code                           (kept on request)

Cells written by Ari are reused verbatim by index from the committed notebook
(read from git, so re-running the builder is idempotent). The superseded
"learned sequence generators" section is dropped; it remains in git history.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "radSeq.ipynb"
SOURCE_COMMIT = "7c26d80"   # last notebook in the old, appended layout


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def old_cells():
    raw = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:notebooks/radSeq.ipynb"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return json.loads(raw)


def reuse(cells, i, replace=None):
    c = json.loads(json.dumps(cells[i]))
    c["outputs"] = [] if c["cell_type"] == "code" else c.get("outputs", None)
    if c["cell_type"] == "markdown":
        c.pop("outputs", None)
    else:
        c["execution_count"] = None
    if replace:
        src = "".join(c["source"])
        for old, new in replace:
            assert old in src, (i, old)
            src = src.replace(old, new)
        c["source"] = src.splitlines(keepends=True)
    return c


def build():
    nb = old_cells()
    old = nb["cells"]
    # old cell 1 (the project introduction) moves to the README, not the notebook
    cells = [reuse(old, 0)]

    cells += [md(r"""
### At a glance

| | status |
|---|---|
| DiT generates radar sequences (simulated data) | **works** — passes all four checks |
| DiT puts targets where it is asked | **works** — trajectory-conditioned |
| generated data helps a detector when data is scarce | **yes, +0.040 mAP** — flagged: the generator partly memorises |
| simulator that looks like real RADIal | **rebuilt from physics** — scene composition still open |
| does a generator make simulator fidelity matter less? | **pilot running** (sim-to-sim rehearsal) |

The sections follow the argument, and every comparison has its own heading:
**1** real radar → **2** simulators vs real (*Comparison 1*) → **3** the generator
(*Comparison 2*) → **4** does generated data help (*Comparison 3*) → **5** the
fidelity study → **6** where this stands. The appendices hold the 64×64
simulator walkthrough and the DiT code.
"""),
              md("## 0. Setup"),
              code(r"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

from src import radial
from src.simulator import generate_sequences
from src.viz import (show_clutter_correlation, show_cuts, show_brightness,
                     show_normalization, show_range_profile, show_rows,
                     show_sequence, show_rd_rows)
"""),

              # ------------------------------------------------ 1. real radar
              md(r"""
## 1. What real radar looks like (RADIal)

Everything in this project is ultimately judged against real radar, so start
there. [RADIal](https://github.com/valeoai/RADIal) is 91 recordings from a
77 GHz imaging radar: 512 range bins (0.2 m each, up to 102 m) × 256 Doppler
bins, from 16 receive antennas, 5 frames per second. Red circles are the
labelled vehicles.
"""),
              code(r"""
by_frame = radial.read_labels()
frames = sorted(by_frame)
real_ids = (frames[0], frames[400], frames[2000])
real_maps = [radial.power_map(s) for s in real_ids]
show_rd_rows({"real RADIal": real_maps},
             marks={"real RADIal": [radial.targets(s, by_frame) for s in real_ids]},
             col_titles=[f"frame {s}" for s in real_ids])
"""),
              md(r"""
What you are looking at — each of these turns out to matter for the simulator:

- **Every return is repeated ~12 times across Doppler, 16 bins apart.** The radar's
  12 transmitters fire at once, each tagged with its own Doppler phase code, so
  every object appears once per transmitter. This is the dominant pattern.
- **Dark vertical bands** are the 4 empty transmitter slots in that code.
- **The textured band** (roughly 10–60 m) is the static world — road edge,
  guardrails, parked cars — seen from a moving car, so it sits at Doppler
  −v·cos(angle).
- **Near range and far range are dark**: the receiver's filters.

**The dataset's labels are partly mislabelled.** `radar_D_mps` holds a Doppler
*bin index*, not m/s, and `radar_P_db` holds linear power, not dB. Range is what
it claims (0.2 m per bin). The check: each labelled vehicle should sit on a local
peak at its predicted cell.
"""),
              code(r"""
peaks = []
for s in frames[:40]:
    p = radial.power_map(s)
    for rb, db in radial.targets(s, by_frame):
        ri, di = int(round(rb)), int(round(db))
        if 2 <= ri < p.shape[0] - 2 and 2 <= di < p.shape[1] - 2:
            peaks.append(p[ri-2:ri+3, di-2:di+3].max() - np.median(p[ri-2:ri+3, :]))
print(f"{len(peaks)} labelled vehicles: peak above their range ring, "
      f"median {np.median(peaks):.1f} dB (p10 {np.percentile(peaks,10):.1f}, p90 {np.percentile(peaks,90):.1f})")
"""),
              md(r"""
### Real data is scarce

Sequences need *consecutive* labelled frames, and recordings are split **whole**
(neighbouring frames of one drive are near-duplicates, so a frame-level split
would measure memorisation). The split is balanced by how many sequences each
recording contributes and is pinned in `data/radial/split.json`.
"""),
              code(r"""
split = json.loads((ROOT / "data" / "radial" / "split.json").read_text())
recs = radial.recordings()
print(f"{'split':<7}{'recordings':>11}{'frames':>8}{'16-frame seq':>14}{'8-frame seq':>13}")
for name in ("train", "val", "test"):
    sub = {k: recs[k] for k in split[name]}
    print(f"{name:<7}{len(sub):>11}{sum(len(v) for v in sub.values()):>8}"
          f"{len(radial.sequences(sub, seq_len=16)):>14}{len(radial.sequences(sub, seq_len=8)):>13}")
"""),
              md(r"""
Only **219 independent 16-frame sequences** exist in all of RADIal (153 for
training) — far fewer than the 2,000 the simulated rehearsal used. That is why
the plan moved to 8-frame sequences and why pretraining on simulated data is the
main defence against memorisation.

---
"""),

              # ------------------------------------------------ 2. simulators
              md(r"""
## 2. Simulators vs real

### 2.1 The simulator the DiT was developed on

A 64×64 grid (3 m range bins), a few point targets over a statistical clutter
model. It was built to develop the generator, not to look like RADIal. Its
building blocks are walked through in **Appendix A**.
"""),
              reuse(old, 32, replace=[('title="8. a random scene from the training distribution"',
                                       'title="a random scene from the 64x64 training simulator"')]),
              md(r"""
### 2.2 Comparison 1 — real radar vs the simulators, one frame

Three rows, same axes:

- **real RADIal**
- **scene simulator** (`src/scene_sim.py`) — rebuilt from radar physics and
  scene geometry: transmitter replication from the authors' own processing code,
  a static world seen from a moving car, 16-antenna averaging, receiver filters,
  antenna patterns. **Nothing is tuned to the recordings.**
- **first spec simulator** — the earlier attempt: the old point-target simulator
  moved onto RADIal's grid. It is kept here to show what was missing.
"""),
              code(r"""
from src.radar_physics import RADIAL_SPEC
from src.scene_sim import SceneSimulator
from src.simulator import spec_simulator

scene = [SceneSimulator(seq_len=1, device=device,
                        generator=torch.Generator().manual_seed(i)).gen_sequence()["x"][0]
         for i in range(3)]
first = []
for i in range(3):
    torch.manual_seed(i)
    first.append(spec_simulator(RADIAL_SPEC, seq_len=1).gen_sequence(n_targets=3)["x"][0])

show_rd_rows({"real RADIal": real_maps, "scene simulator": scene,
              "first spec simulator": first},
             title="Comparison 1: one frame from each source")
"""),
              md(r"""
### 2.3 Comparison 1 — over time

The point of the project is *sequences*, so compare how a scene evolves: an
8-frame real sequence (0.2 s apart) against a simulated one.
"""),
              code(r"""
val = {k: recs[k] for k in split["val"]}
name, run = radial.sequences(val, seq_len=8)[0]
real_seq = radial.load_sequence(run)
sim_seq = SceneSimulator(seq_len=8, device=device,
                         generator=torch.Generator().manual_seed(5)).gen_sequence()["x"]
cols = (0, 2, 4, 7)
show_rd_rows({"real RADIal": [real_seq[t] for t in cols],
              "scene simulator": [sim_seq[t] for t in cols]},
             col_titles=[f"t = {t * 0.2:.1f} s" for t in cols],
             title="Comparison 1: an 8-frame sequence")
"""),
              md(r"""
### 2.4 Comparison 1 — in numbers

Background statistics over many frames (real: 60 frames across the dataset;
simulators: 30 frames each).
"""),
              code(r"""
real_many = np.stack([radial.power_map(s) for s in frames[::60][:60]])
scene_many = np.stack([SceneSimulator(seq_len=1, device=device,
                       generator=torch.Generator().manual_seed(100 + i)).gen_sequence()["x"][0].numpy()
                       for i in range(30)])
first_many = []
for i in range(30):
    torch.manual_seed(100 + i)
    first_many.append(spec_simulator(RADIAL_SPEC, seq_len=1).gen_sequence(n_targets=3)["x"][0].numpy())
table = {"real": radial.background_stats(real_many),
         "scene sim": radial.background_stats(scene_many),
         "first spec sim": radial.background_stats(np.stack(first_many))}
print(f"{'dB':<28}" + "".join(f"{k:>16}" for k in table))
for stat in table["real"]:
    print(f"{stat:<28}" + "".join(f"{table[k][stat]:>16.1f}" for k in table))
"""),
              md(r"""
The scene simulator now matches the real background on range structure and
spread, where the first attempt was off by tens of dB (its targets were also
~39 dB too prominent).

**What is still different, and why it is left open.** Real texture is a
*herringbone* of short slanted streaks; the scene simulator draws long clean
arcs. The cause is **scene composition** — how many guardrails, walls, parked
cars and bushes a road has. Choosing that by comparing with RADIal frames would
be fitting by eye, so the plan is two explicit variants:

- **engineer's simulator** — composition from general knowledge of roads, never
  tuned against RADIal. The honest baseline.
- **fitted simulator** — composition tuned by eye to RADIal, labelled as such:
  the upper bound on what calibration can buy.

---
"""),

              # ------------------------------------------------ 3. generator
              md("## 3. The generator (DiT)"),
              reuse(old, 47, replace=[
                  ("Everything above draws its maps with the simulator. From here on the maps come from the **DiT itself**",
                   "The maps in this section come from the **DiT itself**"),
                  ("The DiT is **unconditional**",
                   "It was developed on the 64×64 simulator of section 2.1. The code is written out in **Appendix B**.\n\nThe DiT shown here is **unconditional**")]),
              code(r"""
from src.dataset import RadarSequenceDataset, held_out_batch
from src.sample import generate

stats = torch.load(ROOT / "data" / "cache" / "stats.pt")
"""),
              md("### Comparison 2 — simulator vs DiT, easy regime: one moving target, no clutter or noise"),
              reuse(old, 66, replace=[("from src.sample import generate\n\n", "")]),
              md("### Comparison 2 — simulator vs DiT, full regime: several targets, classes, clutter and noise"),
              reuse(old, 68),
              reuse(old, 69),
              reuse(old, 70),
              reuse(old, 71, replace=[("As in the section above, these checks",
                                       "These checks")]),
              md("---"),

              # ------------------------------------------------ 4. does it help
              md(r"""
## 4. Does generated data help a detector?

A generator only matters if its output is *useful*. Setup (all simulated, so
the ground truth is exact): the generator saw only **2,000** sequences and
rendered **8,000** labelled ones; a detector must find each target *and* name
its class. Rules written before the run: the test is informative only if 20,000
real sequences beat 2,000 by more than the seed spread, and synthetic data
helps only if it clears the same bar.

### Comparison 3 — detector trained on different data
"""),
              code(r"""
report = json.loads((ROOT / "samples" / "detector_class_augmentation.json").read_text())
names = {"real_n": "2,000 real", "real_n_synth": "2,000 real + 8,000 generated",
         "synth_only": "8,000 generated only", "real_full": "20,000 real (ceiling)"}
print(f"{'trained on':<32}{'mean mAP':>10}{'seed range':>12}")
for key, arm in report["arms"].items():
    print(f"{names[key]:<32}{arm['mean']:>10.3f}{arm['range']:>12.3f}")
rule = report["rule"]
print(f"\ngain from generated data: {rule['gain']:+.3f} (bar {rule['gain_bar']:.3f}) -> helps: {rule['helps']}")

mem = json.loads((ROOT / "samples" / "memorization_n2000.json").read_text())
print(f"memorisation ratio {mem['ratio_training_over_heldout']:.3f} (below 0.8 = copying) -> memorising: {mem['memorising']}")
"""),
              md(r"""
**Generated data helped** (+0.040 mAP, twice the bar, ~40% of the way to the
20,000-real ceiling), and a detector trained on generated data *only* beat one
trained on the 2,000 real sequences. **But** the generator partly memorises
(ratio 0.715), so the result carries that flag — and the lesson for real data
is that scarcity drives memorisation.

---
"""),

              # ------------------------------------------------ 5. fidelity
              md(r"""
## 5. The fidelity study

Instead of building one simulator and hoping it is good enough, **simulator
fidelity is the variable**. For each simulator a DiT is pretrained on it and
fine-tuned on a little real data, and a detector is trained with:

- **A** real only
- **B** real + raw simulator data
- **D** real + DiT data (pretrained on that simulator, fine-tuned on real)

**The result is an interaction:** B should get worse as the simulator gets
worse. If D stays flat, then *a generator makes simulator fidelity matter
less*. If D tracks B, the idea is dead.

**The pilot** rehearses this sim-to-sim on the 64×64 grid: one simulator setting
plays "reality", the others are wrong by a controlled number of dB of target
brightness. (A first run was invalid — the generator's averaged weights were
still mostly random after a short training run — and was discarded; the
corrected run checks that generated targets land where requested before any
detector trains on them.)
"""),
              code(r"""
from src.viz import show_fidelity_curve

path = ROOT / "samples" / "fidelity_pilot.json"
if not path.exists():
    print("pilot still running - no results yet")
else:
    pilot = json.loads(path.read_text())
    done = sorted(pilot["deltas"], key=float)
    arms = {"real": [], "real_sim": [], "real_synth": []}
    print(f"{'mismatch dB':>12}{'A real':>9}{'B raw sim':>11}{'D DiT':>9}{'D - B':>8}{'hit rate':>10}")
    for key in done:
        d = pilot["deltas"][key]
        a = d["detector"]["arms"]
        for n in arms:
            arms[n].append(a[n]["mean"])
        print(f"{key:>12}{a['real']['mean']:>9.3f}{a['real_sim']['mean']:>11.3f}{a['real_synth']['mean']:>9.3f}"
              f"{a['real_synth']['mean'] - a['real_sim']['mean']:>+8.3f}{d.get('synthetic_hit_rate', float('nan')):>10.2f}")
    if len(done) > 1:
        show_fidelity_curve([float(k) for k in done], arms)
"""),
              md(r"""
---

## 6. Where this stands

- **Done:** a DiT that generates radar sequences and passes the four checks; a
  version that places targets on request; evidence its data helps a detector
  when data is scarce (flagged for memorisation).
- **Done:** RADIal decoded and split; a scene simulator from physics that
  matches real background statistics.
- **Running:** the sim-to-sim fidelity pilot.
- **Next:** the two scene-simulator variants (engineer's and fitted), then the
  fidelity comparison on real RADIal with 8-frame sequences.

---
"""),
              ]

    # ------------------------------------------------ appendices
    cells.append(reuse(old, 6, replace=[("## Simulated Dataset",
                                         "## Appendix A — how the 64×64 simulator builds a map")]))
    cells += [reuse(old, i) for i in range(7, 30)]       # walkthrough steps 0..7
    cells.append(md("### 8. The full training distribution\n\nShown in section 2.1."))
    cells.append(reuse(old, 33, replace=[("## Normalization", "### Normalization")]))
    cells += [reuse(old, i) for i in range(34, 39)]
    cells.append(reuse(old, 48, replace=[("### Inside the DiT — the code",
                                          "## Appendix B — the DiT code")]))
    cells += [reuse(old, i) for i in range(49, 65)]
    # cell 64 already imports generate; section 3 imported it too, harmless

    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {NB}: {len(cells)} cells (was {len(old)})")


if __name__ == "__main__":
    build()
