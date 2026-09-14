# Hive Vision — Limelight track (ONNX / YOLO)

The model track of Hive Vision: a lightweight YOLOv8n detector trained on
thousands of synthetically rendered frames of the team's CAD ball plus real
match footage. Ships two artifacts for any ONNX-capable device (Limelight,
Jetson, laptop):

| File | What it is | Use on |
|------|------------|--------|
| `weights/best.onnx` | exported ONNX, 960×960 input, opset 12, ~12 MB | Limelight model runner / ONNX Runtime |
| `weights/best.pt` | Ultralytics source checkpoint, ~6 MB | re-training, re-exporting, PC use |
| `weights/best_limelight3a_float32.tflite` | validated float32 TFLite export, 960×960 input, ~12 MB | Limelight 3A testing |
| `weights/labels.txt` | `yellow_pollen`, `red_nectar`, `blue_nectar` | Limelight 3A labels |

## Viewer (PC / dev)

```bash
# webcam (default)
python scripts/detect_video_realtime.py

# video file
python scripts/detect_video_realtime.py --source path/to/video.mp4

# batch-export an annotated MP4
python scripts/export_annotated.py --source path/to/video.mp4 --out annotated.mp4

# test the Limelight 3A TFLite export over a video
python yolo/scripts/detect_video_tflite.py \
  --source path/to/video.mp4 \
  --output demo/tflite_test.mp4 \
  --confidence 0.35
```

Keys: `q` quit · `p` pause · `f` toggle sphere filter (drops non-ball boxes:
aspect outside 0.55–1.8, area > 3.1% of frame) · `s` save current frame.

The TFLite test runner writes an annotated MP4 with class labels, confidence,
and NMS-filtered boxes. The included test clip is
[`demo/tflite_test_v7f.mp4`](../demo/tflite_test_v7f.mp4). It is a model
inspection tool, not a substitute for testing on Limelight 3A hardware. Its
sphere filter removes oversized or non-ball-shaped boxes by default; add
`--no-sphere` to inspect raw model output.

Verify the ONNX itself:

```bash
python -c "from ultralytics import YOLO; YOLO('weights/best.onnx').predict('test.jpg', imgsz=960)"
```

## Re-exporting / tuning for Limelight

```bash
python -m ultralytics.export model=weights/best.pt format=onnx imgsz=960 opset=12
```

- **imgsz matters.** The dev baseline was measured at `960`; smaller values
  (e.g. `640`) lose the tiny far-corner balls — acceptable on constrained
  coprocessors, but measure with this track's own metrics first.
- Re-upload `best.onnx` to the Limelight after any re-tune.
- Limelight 3A users should upload `best_limelight3a_float32.tflite` with
  `labels.txt`; this export was generated from the shipped ONNX model and
  verified on the included TFLite video test runner.
- The two tracks stay in sync through `cv/tools/fit_hsv_from_yolo.py`, which
  mines the Control Hub HSV ranges straight from this model's detections.

## Accuracy context

This model's detections are the *ground truth* every Hive Vision Control Hub
metric is measured against — see `cv/docs/cv_detector_report.md`.