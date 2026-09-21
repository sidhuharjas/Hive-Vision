# Run YOLO on a PC (ONNX Runtime)

This page gets the YOLOv8n reference model finding balls on a PC — dev laptop,
Jetson, or any other ONNX Runtime host — and exporting annotated videos to skim.
It is dev tooling, not a robot deployment: **no Limelight can run ONNX**.

## What you need

| Thing | What it's for |
|-------|---------------|
| Python 3.10+ | runs the viewer and export scripts |
| `pip install -r requirements.txt` | the model deps (Ultralytics, ONNX Runtime, OpenCV) |
| A webcam or a video file | the detection source |

All commands below run from the repo root (`hive-vision/`).

## Step 1 — Install

```bash
pip install -r requirements.txt
```

## Step 2 — Detect live on a webcam

```bash
python yolo/scripts/detect_video_realtime.py --weights yolo/weights/best.pt
```

...or over a video file:

```bash
python yolo/scripts/detect_video_realtime.py \
  --weights yolo/weights/best.pt --source path/to/match.mp4
```

Viewer keys: `q` quit · `p` pause · `f` toggle sphere filter (drops non-ball
boxes) · `s` save the current frame.

## Step 3 — Export an annotated video

```bash
python yolo/scripts/export_annotated.py \
  --weights yolo/weights/best.pt --source path/to/match.mp4 --out annotated.mp4
```

## Step 4 — Verify the ONNX artifact on its own

```bash
python -c "from ultralytics import YOLO; YOLO('yolo/weights/best.onnx').predict('test.jpg', imgsz=960)"
```

## What to know before you trust it

- **`best.onnx` outputs the raw YOLO tensor, not detections.** It is 960×960
  input, opset 12, and its only output is `output0` (1×7×18900) — rows are
  `[cx, cy, w, h, yellow, red, blue]` in the 960 grid, pre-NMS. The viewer and
  export scripts do the decode + NMS for you; if you consume the ONNX
  directly, you must decode + NMS on your side.
- **Class order**: `yellow_pollen`=0, `red_nectar`=1, `blue_nectar`=2.
- **Never upload ONNX to a Limelight** — Limelight neural detectors accept
  `.tflite`/`.hef` only. The 3A runs
  [SSD-MobileNetV2](limelight_3a_ssd.md) instead.
- **`imgsz` matters**: 960 catches the tiny far-corner balls; 640 saves memory
  but you lose far-corner recall.
- This YOLO model is the **ground truth** the Control Hub tracks are
  tuned/scored against — use it when you're measuring a color detector, not
  the hub detector itself.

Full model details, exports, and TFLite test tooling are in
[yolo/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/yolo/README.md).