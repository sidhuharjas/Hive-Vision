# Overview

Hive Vision is a real-time FTC ball-detection suite (`yellow_pollen`, `red_nectar`, `blue_nectar`). The repo README is the source of truth for everything technical; this book exists to (a) help a new team pick a track and (b) carry setup gotchas the READMEs don't spell out.

* [Which detector track?](detector_tracks/) — the decision guide
* [Limelight 3A (SSD)](detector_tracks/limelight_3a_setup.md) — upload + gotchas
* [PC / ONNX Runtime (YOLO)](detector_tracks/onnx_pc.md)
* [Control Hub (OpenCV + Lab)](detector_tracks/control_hub.md)
* [Deployment checklist](deployment_checklist.md) — what ships where

Classes are always `yellow_pollen`=0, `red_nectar`=1, `blue_nectar`=2.

## In action

Same kickoff footage through each detector:

![Limelight 3A (SSD)](.gitbook/assets/ssd_mobilenetv2.gif)

![Control Hub (OpenCV)](.gitbook/assets/control_hub_cv_example_1.gif)

![Control Hub (Lab)](.gitbook/assets/cielab_demo.gif)

The Limelight 3A model in the real world:

![Limelight 3A in action](.gitbook/assets/ssd_in_action.gif)
