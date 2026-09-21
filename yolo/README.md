# Hive Vision — Limelight 3A (TFLite) + ONNX reference track

The model track of Hive Vision: a lightweight YOLOv8n detector trained on thousands of synthetically rendered frames of the team's CAD ball plus real match footage. It ships an SSD-MobileNetV2 TFLite model for the Limelight 3A and a YOLOv8n ONNX export for **ONNX Runtime hosts** (Jetson, laptop). Note: no Limelight runs ONNX — Limelight neural detectors accept `.tflite` or Hailo `.hef` models only, so the ONNX export never goes on a Limelight of any kind:

| File                                                      | What it is                                                                                                                                                                  | Use on                                                          |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `weights/best.onnx`                                       | exported ONNX, 960×960 input, opset 12, \~12 MB                                                                                                                             | ONNX Runtime only (PC / Jetson / coprocessor) — not a Limelight |
| `weights/best.pt`                                         | Ultralytics source checkpoint, \~6 MB                                                                                                                                       | re-training, re-exporting, PC use                               |
| `weights/best_limelight3a_float32.tflite`                 | validated **float32** TFLite export, 960×960 input, \~12 MB (float32 in/out — **not** full-INT8, so the 3A neural detector will not load it)                                | PC / ONNX Runtime testing only                                  |
| `weights/best_limelight3a_ssd_mobilenetv2_300x300.tflite` | **SSD-MobileNetV2 retrain from the Limelight online trainer** — the model the 3A actually runs. uint8 300×300 input, float32 `TFLite_Detection_PostProcess` outputs, 5.0 MB | Limelight 3A (neural detector)                                  |
| `weights/labels.txt`                                      | `yellow_pollen`, `red_nectar`, `blue_nectar`                                                                                                                                | Limelight 3A labels                                             |

All ONNX artifacts emit the **raw YOLO tensor** `output0` (1×7×18900) — rows are \[cx, cy, w, h, yellow, red, blue] in the model's 960×960 grid, five pre-NMS. Decode + NMS must run on the device or in the FTC pipeline; the model itself does not post-process.

The `ssd_mobilenetv2_300x300` model is different by design: it is the SSD `TFLite_Detection_PostProcess` contract the 3A requires (full-INT8 uint8 input, float32 outputs: num detections, scores `[1,10]`, class ids `[1,10]` 0=yellow 1=red 2=blue, boxes `[1,10,4]` normalized ymin,xmin,ymax,xmax). It was trained on the same synthetic V7f ball corpus and cross-checked against the locally trained equivalent (identical detections on all test frames). This is the artifact to upload to the 3A's Neural Detector.

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

Keys: `q` quit · `p` pause · `f` toggle sphere filter (drops non-ball boxes: aspect outside 0.55–1.8, area > 3.1% of frame) · `s` save current frame.

The TFLite test runner writes an annotated MP4 with class labels, confidence, and NMS-filtered boxes. The included test clip is [`demo/tflite_test_v7f.mp4`](../demo/tflite_test_v7f.mp4). It is a model inspection tool, not a substitute for testing on Limelight 3A hardware. Its sphere filter removes oversized or non-ball-shaped boxes by default; add `--no-sphere` to inspect raw model output.

Verify the ONNX itself:

```bash
python -c "from ultralytics import YOLO; YOLO('weights/best.onnx').predict('test.jpg', imgsz=960)"
```

## Re-exporting / tuning models

```bash
python -m ultralytics.export model=weights/best.pt format=onnx imgsz=960 opset=12
```

* **imgsz matters.** The dev baseline was measured at `960`; smaller values (e.g. `640`) lose the tiny far-corner balls — acceptable on constrained coprocessors, but measure with this track's own metrics first.
* Re-export and re-upload `best.onnx` to your ONNX Runtime host after any re-tune (a Limelight will not load it).
* Limelight 3A users should upload `best_limelight3a_ssd_mobilenetv2_300x300.tflite` with `labels.txt` — the SSD-MobileNetV2 model described above. The YOLO float32/int8 exports below are PC verification artifacts and will not load on the 3A.
* `best_limelight3a_int8.tflite` is _dynamic-range_ quantization (int8 weights, float activations). Full INT8 activations are not possible: the Detect head's per-stride branches keep separate quantization scales, which TFLite forbids on the `CONCAT` (`concatenation` backend error), and a single mixed box+score tensor would collapse the score rows to zero under the box-dominated scale. The dynamic-range variant keeps the exact float32 contract and matches the float32 model's detections (≤0.02% box delta on the 13-frame verification sweep). Rebuild it with `scripts/export_int8_tflite.py` (requires Python 3.12 + TF in `hive-vision/yolo/.venv-tflite`).
* The two tracks stay in sync through `cv/tools/fit_hsv_from_yolo.py`, which mines the Control Hub HSV ranges straight from this model's detections.

## Accuracy context

This model's detections are the _ground truth_ every Hive Vision Control Hub metric is measured against — see `cv/docs/cv_detector_report.md`.
