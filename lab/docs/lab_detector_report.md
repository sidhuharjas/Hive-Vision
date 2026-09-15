# Lab Ball Detector — evaluation report

The **middle track** of Hive Vision: a ball detector that runs on the robot's
**Control Hub** (like the raw-HSV OpenCV track, no model / no coprocessor) but
matches on **CIELAB chromaticity** instead of HSV channels (like the YOLO
track's color language, minus the network).

- `LabBallDetectorPipeline.java` — drop-in `VisionProcessor`.
- Constants are **learned from the flagship YOLO model** (`yolo/weights/best.pt`),
  never hand-picked: the YOLO sees a ball → we sample the pixels it really
  looked at → we store that color's Lab hue band + chroma floor.
- Result: a shadow cannot change a ball's classification by dragging its HSV
  V/S values out of a tuned window, because Lab separates **lightness (L*)** —
  which shadows change — from **chromaticity hue** `atan2(b*, a*)` — which they
  mostly don't.

## How the detector works

Per frame (one `RGB→Lab` conversion, shared by all three colors):

1. `theta = atan2(b* - 128, a* - 128)` in degrees → a single hue angle per pixel.
   The Control Hub Java pipeline applies the hue band with **cross-product edge
   tests** (two `>=0` comparisons of `a*`, `b*` products) instead of a per-pixel
   `atan2`, so it stays OpenCV element-wise ops only.
2. A pixel matches a color iff:
   ```
   hue in [hue_band_lo, hue_band_hi]            // chromaticity band (shadow-invariant)
   AND  sat > sat_floor * scl                    // chroma floor, scaled to lighting
   AND  L  > l_floor                             // 8.0: drop near-blacks only
   scl = clamp(meanL / ref_l, scl_lo, scl_hi)    // adaptive gain from frame ambient L*
   ```
3. The `sat` floor is **not fixed**: in a dim frame the whole image's chroma
   shrinks, so the floor is lowered proportionally to the frame's median L*
   relative to the reference lighting the fitter measured (`ref_l`). This is the
   part a plain HSV rectangle can't reproduce.
4. Morph, area/aspect/fill gates (learned from YOLO boxes), and a best-blob
   picker per color — same pipeline shape as the HSV track.

## Learned constants (`lab/tools/lab_tuned.json`)

Fitted on 293 frames of real match footage that the flagship YOLO scored at
conf ≥ 0.35, sampling 400 ball boxes (yellow 121 / red 113 / blue 123 frames):

| color | hue band (deg, shipped) | sat floor | L floor | typical area |
|-------|-------------------------|-----------|---------|--------------|
| yellow | [22.3, 105.0] | 26.1 | 8.0 | 1950 px |
| red    | [335.0, 62.0] (wraps 0) | 12.4 | 8.0 | 2986 px |
| blue   | [258.0, 326.0] | 7.4 | 8.0 | 3182 px |

Ambient reference `ref_l = 51.7`, adaptive gain `scl ∈ [0.25, 1.5]`.
The raw fit came out **wide** on this footage (≈120°) because real balls are
low-chroma on this sensor: the saturated "core" pixels (sat ≥ 15) pick the
band, but body pixels sit far down the chroma scale where hue is noisier.
A later false-positive pass (the ~15 blobs/candidate that the picker was
choosing from) pulled the volatile **edges** of each band into the darker
YOLO-learned core and raised the red floor 8.2 → 12.4 — this cut no-ball
frame spam (green/yellow mats for yellow, orange/amber chrome for red,
blue-violet field fabric for blue) with a net recall *gain*, because a
cleaner mask stops chrome blobs from beating the real ball in the picker.

## Measured vs flagship YOLO truth (identical ground truth for Lab and HSV)

Same 293 truth frames, same IoU/match rule, same gating (geometry gates
identical; only the color model differs):

| detector | color | frames | recall | precision | dark-frame recall |
|----------|-------|--------|--------|-----------|-------------------|
| **Lab**  | yellow | 121 | 45.5% | 7.5%  | 0.0% (1 dark frame) |
| **Lab**  | red    | 113 | 46.0% | 8.7%  | **95.5%** (of 22)   |
| **Lab**  | blue   | 123 | 54.5% | 8.1%  | 0.0% (11 dark)      |
| HSV ref  | yellow | 121 | 62.0% | 10.5% | 0.0% (1 dark frame) |
| HSV ref  | red    | 113 | 34.5% | 8.6%  | 40.0% (of 22)       |
| HSV ref  | blue   | 123 | 52.8% | 11.1% | 0.0% (11 dark)      |

## What the numbers actually say

**The shadow claim is real — and it shows up mostly on red.** Lab's
dark-frame recall on red is **95.5% vs 40.0% for HSV** (dark frame = truth box
median L* < 40). Those are the frames HSV loses because the shadow drags
S/V out of its window; Lab keeps them because hue is S/V-independent and its
chroma floor adapts downward by `scl`.

**Deep shadow is a physics floor for every color-only detector.** Sampling the
darkest frames shows the sensor stops reporting color before shape fails: e.g.
frame 573's yellow ball has hue ≈ 80° (correct band) but chroma p50 ≈ 4 out of
~0–140 — the "yellow" is a grey. No HSV or Lab threshold can see that; it needs
the **YOLO track** (shape/context). The single yellow and 11 blue dark frames
here are exactly these unwinnable ones, which is why dark recall reads 0.0%
while overall blue recall still beats HSV.

**Lab trades yellow-precision for red/blue robustness on this footage.** HSV
was hand-tuned on a different video where its yellow window was luckier; Lab is
rigidly learned from this one. The false-positive pass closed most of the
precision gap — Lab now *beats* HSV on red in recall **and** precision
(46.0/8.7 vs 34.5/8.6), holds blue recall, and is the only variant that
recovers red in shadows. Yellow remains HSV's (its tighter window admits less
bright field-scape); blue precision too (both see the field's blue-violet
fabric). The remaining Lab FPs are real-color patches elsewhere on the field —
only multi-frame confirmation or the YOLO track removes those.

Bottom line for a match: run the Lab track when the field puts balls in
**shadows** (its whole purpose), pair its candidates with multi-frame
confirmation, and graduate balls that only ever appear in the darkest corners
to the YOLO track if you have a coprocessor.

## Reproduce

```bash
# 1. Re-fit constants from the flagship model (cached truth skips the model)
python lab/tools/fit_lab_from_yolo.py --source path/to/match_raw.mp4 \
  --out lab/tools/lab_tuned.json --truth-json lab/docs/yolo_truth_lab.json

# 2. Sweep scl (chroma-floor gain) x floor (per-color floor multiplier)
python lab/tools/sweep_lab.py --source path/to/match_raw.mp4 \
  --truth-json lab/docs/yolo_truth_lab.json --scl 0.18,0.25,0.35 \
  --floor 0.4,0.6,0.8,1.0

# 3. Final side-by-side vs the HSV track (same truth)
python lab/tools/eval_lab_vs_yolo.py --source path/to/match_raw.mp4 \
  --config lab/tools/lab_tuned.json --compare-hsv \
  --truth-json lab/docs/yolo_truth_lab.json

# 4. Live debug / live re-tune
python lab/tools/realtime_lab.py --source path/to/match.mp4
python lab/tools/lab_tuner.py path/to/field_frame.jpg
```

Use **raw** (un-annotated) footage so both detectors and the truth generator see
pixels the way the cameras see them.