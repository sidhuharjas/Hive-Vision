# Limelight 3A (SSD-MobileNetV2)

The on-robot neural track: a TFLite **SSD-MobileNetV2** model trained on the
same synthetic + real corpus as the YOLOv8n reference, running on the
Limelight 3A's own CPU. The decision guide is in
[Which detector track?](README.md); wire it up step by step with the
[set-up walkthrough](limelight_3a_setup.md). Model details and export notes
are in [yolo/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/yolo/README.md).
Short version:

- **Model**: `yolo/weights/best_limelight3a_ssd_mobilenetv2_300x300.tflite`,
  300×300 uint8 input, SSD `TFLite_Detection_PostProcess` outputs (`num`,
  `scores [1,10]`, `class_ids [1,10]`, `boxes [1,10,4]` normalized) — read
  them through the Limelight API used by your FTC integration.
- **Classes**: `yolo/weights/labels.txt` — `yellow_pollen`=0, `red_nectar`=1,
  `blue_nectar`=2. Keep the file order exactly as shipped.
- **Run**: upload to a **Neural Detector** pipeline slot, set **runtime
  engine to CPU** (the 3A has no accelerator — Coral/Hailo models won't run),
  threshold **0.35–0.45**, input resolution **300 × 300**.

Two things to know, stated plainly:

- This is the **only track that understands shape** — use it when balls and
  their shadows are in play and a color-only detector can't be trusted.
- **No Limelight runs ONNX.** This SSD model is what sits *in place of* YOLO;
  `best.onnx` stays on a PC / ONNX Runtime host.
- Per the repo checklist this SSD is verified on PC runners but **not yet on a
  real 3A** — first time on hardware, confirm the pipeline loads and reports
  detections before match day.