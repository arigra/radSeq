<!-- REGAIN_AGENT_V2 -->
# Does measured adaptation close the gap?

Measured adaptation has not been shown to compensate for simulator mismatch. A generator trained from scratch raised synthetic target-class mean AP from 66.46% to 70.46%, with a copying warning. A separate simulator pilot saved only zero brightness offset, where generated additions scored 48.51% against raw simulation’s 60.55%. Larger-scene recipe checks are saved, but the measured RADIal detector comparison is still missing.

Words used here:

- mean AP: Area under the precision-recall curve, averaged over target classes. Higher is better.
- fine-tuning: Continuing a pretrained generator’s learning on measured training examples.
- seed spread: Highest minus lowest score across runs with different random starting points.
- EMA: An exponential moving average of model weights, used for generation.
- dynamic range: Here, the difference between the 99.9th and 0.1st percentile map values.
- schedule shift: A setting that changes how much noise the generator learns to remove.
- dB: Decibels, a logarithmic scale for comparing radar power.
- checkpoint: Saved model weights and training state that can start or continue a run.

## Compare added data

The test needs to show that adaptation protects detector performance when extra raw simulator data becomes less useful.

For each simulator variant, compare measured training sequences alone, those sequences plus raw simulation, and those sequences plus generation after fine-tuning. The design also proposes generation without fine-tuning to isolate what adaptation contributes. Keep the measured subset, held-out recordings, added sequence count and detector training budget fixed, with three random seeds. The current code exposes engineer settings without measured fitting and fitted settings calibrated on measured training recordings, while the design describes three simulator levels. Success requires a difference in how raw simulation and adapted generation respond to mismatch, which the saved zero-offset pilot cannot establish.

### Define the success pattern

Lines 142-146 propose raw simulation worsening while adapted generation stays useful. This is the intended test, not a saved measured result.

Source: [docs/notes/2026-09-21-radial-sim2real-design.md lines 138-146](../docs/notes/2026-09-21-radial-sim2real-design.md)

```file docs/notes/2026-09-21-radial-sim2real-design.md 138-146 visual=visuals/2026-09-21-radial-sim2real-design-138-146.svg
```

### Separate simulator settings

Line 397 exposes engineer and fitted variants. Lines 391-394 load fitted settings separately, keeping measured calibration out of the engineer variant.

Source: [src/scene_sim.py lines 383-397](../src/scene_sim.py)

```file src/scene_sim.py 383-397 visual=visuals/scene_sim-383-397.svg
```

## Bound the rehearsal

The synthetic gain passes its saved comparison rule, but neither pretrained adaptation nor transferable variation is established.

The 64×64 rehearsal trained the generator from scratch on 2,000 simulator sequences, rather than adapting pretrained weights to recordings. Adding 8,000 generated sequences raised mean AP from 66.46% to 70.46%, with detectors trained for 3,000 steps and evaluated on 512 held-out simulator sequences. A detection must have the correct target class and lie within two map bins of its labelled position, and the 4.00-point gain exceeds the larger seed spread of the compared runs, 1.82 points. The rule also checks that using all 20,000 simulator training sequences improves the baseline beyond run variation before calling the augmentation test informative. Separately, 128 requests from training labels and 128 from held-out labels produced median normalized map distances of 0.795 and 1.112 to their matching examples, triggering the copying warning.

> **Experimental-design limitation:** The equal-size simulator-only control was not tested: 10,000 simulator sequences versus 2,000 simulator sequences plus 8,000 generated sequences. The observed gain shows that generated augmentation helped relative to using only 2,000 simulator sequences; it does not show that generated data is better than additional simulator data. The 20,000-sequence simulator arm does not answer that question because its dataset is larger. To assess this, compare the two 10,000-sequence datasets with the same detector, training steps, seeds, and evaluation set. The copying warning also limits conclusions about new variation.

### Pass the gain rule

Line 122 saves a gain of 0.0400, or 4.00 percentage points, against line 123’s 1.82-point bar. Line 121 records that the full-data comparison leaves enough room for improvement.

Source: [experiments/step2_detector/results/detector_class_augmentation.json lines 118-125](../experiments/step2_detector/results/detector_class_augmentation.json)

```file experiments/step2_detector/results/detector_class_augmentation.json 118-125
```

### Flag closer training matches

Line 8’s distance ratio is 0.715, below the check’s 0.8 threshold. Generated sequences match training examples more closely than held-out examples.

Source: [experiments/step2_detector/results/memorization_n2000.json lines 2-9](../experiments/step2_detector/results/memorization_n2000.json)

```file experiments/step2_detector/results/memorization_n2000.json 2-9
```

### Compare matching examples

Lines 50-51 compare generation with examples carrying the requested labels. Line 60 applies the ratio threshold, which flags a copying concern without proving exact duplication.

Source: [scripts/memorization_check.py lines 47-60](../scripts/memorization_check.py)

```file scripts/memorization_check.py 47-60
```

## Zero offset loses

At the pilot’s only saved brightness offset, adding raw simulation beats adding generated sequences.

The design planned offsets of 0, 4, 8 and 16 dB between simulator target brightness and a simulated reference world, but the result contains only 0 dB. It records 200 reference training sequences and 6,000 simulator sequences for generator pretraining, with 800 added sequences in each detector comparison. After 2,500 detector training steps, mean AP across three seeds on 400 reference test sequences is 46.20% with reference data alone, 60.55% with raw simulation and 48.51% with generation. The generated comparison’s 12.47-point seed spread is also larger than raw simulation’s 3.46-point spread. The archive index identifies this as the corrected completed point after an invalid EMA run, and the missing nonzero offsets leave the proposed mismatch trend untested.

### Save zero offset

Lines 6-8 mark only the 0 dB point complete. Lines 13-14 record equal additions of 800 sequences, and lines 15-16 fix the detector budget and test count.

Source: [archive/fidelity_pilot_64/results/fidelity_pilot.json lines 2-16](../archive/fidelity_pilot_64/results/fidelity_pilot.json)

```file archive/fidelity_pilot_64/results/fidelity_pilot.json 2-16
```

### Favor raw simulation

Lines 69 and 96 give mean AP of 60.55% with raw simulation and 48.51% with generation. Lines 70 and 97 give their respective seed spreads.

Source: [archive/fidelity_pilot_64/results/fidelity_pilot.json lines 69-98](../archive/fidelity_pilot_64/results/fidelity_pilot.json)

```file archive/fidelity_pilot_64/results/fidelity_pilot.json 69-98
```

## Check scene readiness

The saved larger-scene checks justify a training recipe, but they compare generation with simulation rather than measured detector performance.

Calibration uses measured training recordings only, and the fitted simulator’s vehicles remain 20.03 dB above surrounding range background against measured vehicles’ 23.98 dB. The current engineer value is 14.84 dB, so the older design note’s account of overly bright targets belongs to an earlier simulator. The recipe evaluation uses eight generated sequences and eight separate held-out engineer-simulator sequences, each containing eight 512×256 frames. After 20 epochs, schedule shift 16 produces a 50.00 dB dynamic range against the simulator’s 57.20 dB, closer than shift 4’s 42.06 dB but still short by 7.20 dB. Both pretraining configurations select shift 16 and 80 epochs, which specify a budget rather than prove that training finished.

### Retain a contrast gap

Line 9 gives measured vehicle prominence of 23.98 dB. Lines 18 and 40 give engineer and fitted values of 14.84 and 20.03 dB, leaving fitted vehicles 3.95 dB below the measured value.

Source: [samples/scene_sim_fit.json lines 2-40](../samples/scene_sim_fit.json)

```file samples/scene_sim_fit.json 2-40 visual=visuals/scene_sim_fit-2-40.svg
```

### Recover brightness spread

Lines 91-95 identify shift 16 after 20 epochs. Lines 97 and 104 give generated and simulator dynamic ranges of 50.00 and 57.20 dB, measured between the same percentiles.

Source: [samples/scene_dit_smoke.json lines 88-109](../samples/scene_dit_smoke.json)

```file samples/scene_dit_smoke.json 88-109 visual=visuals/scene_dit_smoke-88-109.svg
```

### Choose the training budget

Lines 8-10 set eight frames at 512×256. Lines 24 and 32 select schedule shift 16 and 80 epochs, not a completed training result.

Source: [configs/pretrain_engineer.yaml lines 5-32](../configs/pretrain_engineer.yaml)

```file configs/pretrain_engineer.yaml 5-32 visual=visuals/pretrain_engineer-5-32.svg
```

## Finish measured adaptation

The launcher stops at simulator pretraining, while measured fine-tuning and detector evaluation still need concrete configurations and saved outputs.

The launcher trains engineer then fitted variants, continues from last.pt when present, and writes STAGE_DONE after each training command succeeds. This checkout has neither local checkpoint nor log directories, and its current configurations and scripts provide no measured adaptation run or measured detector result. The training loop can start from pretrained weights through train.init_from with a fresh optimizer, whereas resume restores optimizer state and training counters. Scene inputs request per-frame vehicle positions and presence rather than the rehearsal’s target classes, so the measured detector task and matching rule need to be specified. Short adaptation runs also need an explicit raw or EMA weight choice because the averaging estimate can retain substantial starting weights, and its warning uses the configured full budget rather than actual completed updates.

### End at simulator training

Line 18 continues an existing checkpoint, and line 22 writes the completion marker after success. The launcher invokes no measured adaptation or detector evaluation.

Source: [scripts/run_pretrain.sh lines 12-24](../scripts/run_pretrain.sh)

```file scripts/run_pretrain.sh 12-24
```

### Load pretrained weights

Line 245 starts from init_from only without a resume state. Lines 250-254 select EMA weights by default or raw weights explicitly before line 261 creates the new average.

Source: [src/train.py lines 245-261](../src/train.py)

```file src/train.py 245-261
```

### Estimate averaging carryover

Line 306 estimates starting-weight retention using configured epochs times loader length. Line 307 warns above 10%, so this warning does not verify the checkpoint’s actual completed updates.

Source: [src/train.py lines 306-311](../src/train.py)

```file src/train.py 306-311
```

## Open questions

- Which cluster checkpoints, completed epoch counts and STAGE_DONE markers establish that both simulator pretraining runs finished?
- Which measured vehicle detection task, position matching rule and score will replace the synthetic target-class task?
- Which measured budgets, eight-frame windows and weight choices will configure adaptation and the proposed generator control without adaptation?
- Will the study retain engineer and fitted variants, or implement the design note’s third simulator level?
- Will the design note be updated to distinguish the older overly bright simulator from the current engineer simulator’s fainter vehicles?
- Does adapted generation beat matched raw simulation as mismatch increases, with uncertainty estimated across held-out recordings?
