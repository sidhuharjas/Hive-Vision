# Hive Vision — Update #3

**Limelight 3A neural detector is now ship-ready.**

Real-footage testing on the 3A exposed that the 8-bit exports initially in the
repo wouldn't load on the hardware ("needs to be a 8-bit.tflite"). This update
fixes that.

## What changed

- **Real model shipped.** The SSD-MobileNetV2 retrain from the **Limelight
  online trainer** (`weights/best_limelight3a_ssd_mobilenetv2_300x300.tflite`)
  is the one the 3A actually runs: uint8 300×300 input, full-INT8, float32
  `TFLite_Detection_PostProcess` outputs — the exact contract the hardware
  loader requires.
- **Bad int8 removed.** Deleted `weights/best_limelight3a_int8.tflite` (DRQ
  int8-weight / float-activation). The 3A rejected it on load, so it's gone
  from the repo to guarantee nobody ships it by accident.
- **float32 export re-labeled.** `weights/best_limelight3a_float32.tflite` is
  float32 in/out, so the 3A won't load it either — it is now documented as
  **PC / ONNX Runtime only**, not a 3A artifact.
- **Labels:** `yellow_pollen`, `red_nectar`, `blue_nectar` (class ids 0/1/2).

## Detection quality (test frames)

- **Independent model.** SSD-MobileNetV2 is its own architecture — unrelated
  to the local YOLO retrain (`best.pt`/`best.onnx`). Different input size,
  different backbone, no shared weights; do **not** expect it to match the
  YOLO detections. This is the model the 3A runs, full stop.
- Confidences 0.87–0.92 on the multi-ball clip (excludes the filtered
  low-confidence false positives).
- Yellow pollen, red nectar, and blue nectar all detected and classified
  correctly.

See `yolo/README.md` for the artifact table.

## Reading the SSD detector output (decode in FTC code)

The 3A returns 4 arrays per frame:

| Array | Shape | Meaning |
|-------|-------|---------|
| `num_detections` | 1 | how many objects found (0–10) |
| `scores` | 10 | confidence per detection, 0.0–1.0 |
| `class_ids` | 10 | 0=yellow pollen, 1=red nectar, 2=blue nectar |
| `boxes` | 10×4 | `[ymin, xmin, ymax, xmax]`, normalized 0–1 |

Rules: only trust entries `i < num_detections[0]`; skip detections with
`scores[i] < 0.3`; then `class_ids[i]` = which ball, `boxes[i]` = where
(multiply by frame W/H to get pixels).

```python
for i in range(int(num_detections[0])):
    if scores[i] < 0.3:
        continue
    cls = class_ids[i]                    # 0 yellow, 1 red, 2 blue
    ymin, xmin, ymax, xmax = boxes[i]     # 0..1 -> * frame size for pixels
    cx = (xmin + xmax) / 2 * W
    cy = (ymin + ymax) / 2 * H
    print(f"{LABELS[cls]} at ({cx:.0f},{cy:.0f}) conf={scores[i]:.2f}")
```
