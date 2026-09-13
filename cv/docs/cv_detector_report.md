# CV Detector Report — HSV color detector vs YOLO truth

**Date:** 2026-09-13  ·  **Video:** development match video (866 frames, 1920×1080)
**Config evaluated:** `OpenCV-Ball-Detector/tools/hsv_tuned.json` (final shipped config)

Scoring matches what an OpMode trusts: gated blobs → **best-blob-per-color
(mixed)** → check overlap of that one blob with a YOLO-labeled ball. Every
number below is from `eval_cv_vs_yolo.py` run directly on the final config.

---

## Final config

HSV ranges were **learned from YOLO-confirmed ball pixels** on this video
(`fit_hsv_from_yolo.py`), not hand-tuned:

| color | HSV segments (H/S/V, OpenCV H 0..180) |
|-------|----------------------------------------|
| yellow | `[9, 90, 45] – [33, 255, 238]` |
| red    | `[0, 81, 45] – [15, 255, 232]` and `[170, 81, 45] – [180, 255, 232]` |
| blue   | `[102, 80, 45] – [128, 255, 247]` |

Gates: area `900..12000` px, aspect `0.71..1.4`, fill `0.5..1.3`.
Best-of score = `|area/expect − 1| + 4·(1 − fill) + edge_penalty` where
`expect = {yellow: 3500, red/blue: 4200}` and edge penalty hits blobs that
touch the frame border (robot panels hug the edges).

Mirrored in: `hsv_tuned.json`, `tools/realtime_cv.py`, `hsv_tuner.py`,
`TeamCode/BallDetectorPipeline.java`.

---

## Results vs YOLO ground truth (conf ≥ 0.35, ball-like boxes)

| color | yolo frames | tp | fn | fp | precision | recall |
|-------|-------------|----|----|----|-----------|--------|
| yellow | 415 | 285 | 130 | 430 | 39.9% | 68.7% |
| red    | 151 | 121 |  30 | 344 | 26.0% | 80.1% |
| blue   | 434 | 357 |  77 | 188 | 65.5% | 82.3% |
| **all** | **576** | **763** | **237** | **962** | **44.2%** | **76.3%** |

Definitions (from `eval_cv_vs_yolo.py`):
- **tp** = best blob's center is inside a same-color YOLO box.
- **fn** = YOLO labeled the color, CV's best blob missed it (or fired nothing).
- **fp** = CV produced a best blob of a color that had no matching YOLO ball.
  FP is frame-based, so repeated panel detections across frames count up.

Firing volume (same config, no YOLO): **1.99 blobs/frame, output on 770/866
frames** — vs ~50 raw threshold blobs/frame before gating.

---

## Why balls are missed (per-FN reason, `reports/cv_fn_reasons_final.csv`)

Reason = union-of-YOLO-box crop doesn't color-match (`COLOR_MISS`), smallest
gated component too small (`TOO_SMALL`), fails aspect/fill (`SHAPE_MISS`),
or the pixel region WOULD pass but best-of picked a different blob
(`HIT_BUT_BEST_WRONG`).

| \ | COLOR_MISS | TOO_SMALL | TOO_BIG | SHAPE_MISS | HIT_BUT_BEST_WRONG |
|-----|-----------|-----------|---------|------------|--------------------|
| yellow (130) | 0 | 6 | 1 | 10 | **113 (87%)** |
| red (30)     | 4 | 5 | 0 | 7 | 14 (47%) |
| blue (77)    | 7 | 27 | 0 | **30 (39%)** | 13 (17%) |

Conclusions:
- **Yellow** misses are almost all a pick-quality problem, not a color
  problem — the ball pixels are in range; the scorer took a bigger/squarer
  yellow blob instead. Next lever if it matters: raise `FILL_W`.
- **Blue** misses split between region shape (`SHAPE_MISS`, often motion
  blur or overlap) and small balls (`TOO_SMALL` < 900 px — genuinely tiny
  far-corner balls, below what a raw CV detector can reliably separate
  from speckle).
- **Red** is the noisiest color (FP 344) — red panels/tape share the band;
  precision is inherently lower.

---

## Debug frames

`reports/cv_fn_debug_final/` — 169 annotated frames for the **final shipped
config** (122 `wrongbest_`, 47 `fn_`):
- `fn_<color>_<frame>.jpg`: YOLO truth green; CV produced nothing.
- `wrongbest_<color>_<frame>.jpg`: green = YOLO box, orange = all gated CV
  blobs, white border = blob sitting on the ball that the best-of **didn't**
  pick, blue border = the blob best-of *did* pick.

(`reports/cv_fn_debug/` holds the same annotations from earlier gate
variants.)

---

## Artifact map

- `tools/fit_hsv_from_yolo.py` — mines HSV ranges from YOLO boxes (two-stage
  hue band + S/V floors + red wrap handling).
- `tools/eval_cv_vs_yolo.py` — the scorer above; `--score mixed --fill-w 4
  --edge-w 1.0 --truth-json reports/yolo_truth.json` reproduce the final
  numbers in ~25 s (labels cached, no YOLO pass on repeat runs).
- `tools/hsv_tuned.json` — final ranges (same content as `hsv_from_yolo.json`).
- `tools/realtime_cv.py` / `hsv_tuner.py` / `sample_hsv.py` — viewer, live
  tuner, render-range miner.
- `TeamCode/BallDetectorPipeline.java` — Java port (constants, gates, mixed
  `bestOf()`).
- `reports/yolo_truth.json` — cached YOLO ground truth (576 frames).
- `reports/cv_fn_reasons_final.csv` — the FN reason table source.

---

## Verified `--fill-w` choice

`--fill-w 6` was probed as a candidate improvement: it trades **red recall
80.1 → 78.8** for trivial gains on yellow/blue (69.2/82.7 vs 68.7/82.3; +4
TP total). Rejected. **Final weights: `--fill-w 4 --edge-w 1.0`.**

## Next steps (open items)

1. **Eyeball** `reports/cv_fn_debug_final/` frames to confirm the scoring
   story (the developer tool used here can't render images, so this is a
   human check). Green = YOLO truth, orange = gated CV blobs, blue = picked
   blob, white = blob-on-ball that wasn't picked.
2. **Robot side (done as guidance):** `BallDetectorPipeline.java` `bestOf()`
   javadoc + README now carry the candidate-confirmation pattern — require
   the same ball for N consecutive frames (track `cxNorm/cyNorm` within a
   small radius) before acting, since red/yellow blob precision is only
   ~26-40%.
3. **On-field:** re-tune once on the real field camera with `hsv_tuner.py`
   and re-sync `BallDetectorPipeline.java` constants if lighting differs.