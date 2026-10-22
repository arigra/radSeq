"""Step 5 headroom rule: which real-data size, if any, leaves room for extra data.

Pre-registered in docs/notes/2026-10-22-step5-source-only-and-headroom.md:
headroom(N) = AP(387) - AP(N) for the real-only detector; bar = the largest
seed range among all real-only points. N* is the largest N in 200/100/50/25
whose headroom exceeds the bar. Prints N* or "none" and writes the table.
"""
import json
from pathlib import Path

SIZES = (200, 100, 50, 25)


def main():
    full = json.loads(Path("samples/step4_engineer.json").read_text())["arms"]["real"]
    points = {387: full}
    for n in SIZES:
        points[n] = json.loads(Path(f"samples/step5_curve_n{n}.json").read_text())["arms"]["real"]
    bar = max(p["range"] for p in points.values())
    headroom = {n: full["mean"] - points[n]["mean"] for n in SIZES}
    chosen = next((n for n in SIZES if headroom[n] > bar), None)
    Path("samples/step5_headroom.json").write_text(json.dumps({
        "real_only_ap": {n: {"mean": p["mean"], "range": p["range"]} for n, p in points.items()},
        "bar (largest seed range)": bar,
        "headroom vs 387": headroom,
        "chosen_n": chosen}, indent=2))
    print(chosen if chosen is not None else "none")


if __name__ == "__main__":
    main()
