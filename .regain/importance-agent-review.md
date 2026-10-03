# ReGain agent review

Work through this project as an engineer would before assigning semantic
importance. Read its README, architecture and configuration documents, the
linked source files, their callers, and the notebook. Trace the main inputs,
outputs, decisions, and side effects. If a claim remains uncertain, say so in
the reason and keep its importance conservative. Do not claim perfect project
understanding from syntax alone.

Inspect `.regain/line-importance.json`. The generated first pass describes
syntax and marks uncertain lines with low confidence. For every important or
critical line, decide whether changing it could affect a project result,
whether it only affects presentation, and what downstream behavior depends on
it. Review supporting lines that might actually select input data or control
processing. Give each correction a short, specific reason an engineer can
understand on hover. A conditional used only for logging or plotting is
supporting, even if it uses `if`. Mark a decision gate critical only when its
effect has been traced. Report unresolved questions instead of guessing.
Definition headers inherit the strongest body color as a draft. Review each
function or class as a whole: a helper can stay green even if it contains a
branch, and a public entry point may matter more than any one body line.
The C++ first pass uses a lightweight scanner, not a compiler. Check macros,
templates, overloads, build flags, and call sites before trusting its map.

Write corrections to `.regain/importance-reviews.json` as a JSON array. Each
entry needs `file` (relative to project root) or `cell_sha256`, `sha256`, an
exact `match` unique to one line, `importance` (critical, important, or
supporting), and `why`. If a match repeats, also supply its one-based `line`
number. Use the sha256 shown below so changed code cannot
silently retain an old explanation. For example, the shape is:

```json
[
  {
    "file": "path/to/module.py",
    "sha256": "copy the current file hash from the list below",
    "match": "an exact substring of one source line",
    "importance": "critical",
    "why": "Explain the specific downstream decision this line controls."
  }
]
```

Run the same generator command again to apply reviews. Resolve every
"Review needed" message. Revisit explanations after source changes.

## Sources to understand and review

- src/scene_sim.py — 5 decision/calculation lines to review; sha256 `c9bc7d3202f2beb303234d8928f4a4512e44a92f82216ef034425948f21d6fb0`
- experiments/step2_detector/results/detector_class_augmentation.json — 7 decision/calculation lines to review; sha256 `2527fbceabf593d23733e089a6440c1f56535eb4f48f717cbdc000e31cd9b086`
- experiments/step2_detector/results/memorization_n2000.json — 8 decision/calculation lines to review; sha256 `07a4a6efea1fd0627fec65913c139e7d4c01480f2c61ccd42e8908a8d82df844`
- scripts/memorization_check.py — 13 decision/calculation lines to review; sha256 `b993f5b07eab1e6e10675ec4b4c2fc7dfd990c54b34490db483733be06ff1e9e`
- archive/fidelity_pilot_64/results/fidelity_pilot.json — 32 decision/calculation lines to review; sha256 `f9a1fe31f9b46fdaa6dbf2824689110e5f385ced1d20dc30434d76d29c15eb44`
- samples/scene_sim_fit.json — 35 decision/calculation lines to review; sha256 `c10b3aa91c627d6a0fdc442be30ddfcde5c6d885af794036e0c923e1d35e34c1`
- samples/scene_dit_smoke.json — 19 decision/calculation lines to review; sha256 `e8055b9d8e607c796cba1979368ad52e77ac7a38784b41241d70c7272529c4ff`
- configs/pretrain_engineer.yaml — 27 decision/calculation lines to review; sha256 `253304573239c573cc357bd73e85ef7fbf752df90d88bcd32bf80cfad3fd9673`
- src/train.py — 15 decision/calculation lines to review; sha256 `ee608ce603e22f8dc85af215cf30765b334a04a0643046a093b0487fddf8373c`
