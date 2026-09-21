# PC / ONNX Runtime (YOLO)

The reference model + dev tooling. Model details, commands, and export notes
are all in
[yolo/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/yolo/README.md),
and the step-by-step is on [Run YOLO on a PC](onnx_run_pc.md).
Short version:

- **Model**: `yolo/weights/best.onnx`, 960×960 input, raw YOLO tensor output
  (`1×7×18900`, pre-NMS — decode it yourself).
- **Run**: `python yolo/scripts/detect_video_realtime.py --weights yolo/weights/best.pt`.
- **Never** upload this to a Limelight.

Two things to know, stated plainly:

- This is the **ground truth** the Control Hub tracks are tuned/scored against —
  use it when you're measuring a color detector, not the hub detector itself.
- Drop to `imgsz 640` only if your host is memory-bound; you lose far-corner
  recall.