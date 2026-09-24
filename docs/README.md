# Get started

Hive Vision is a real-time FTC ball-detection suite for **BioBuzz** (`yellow_pollen`
= neutral, `red_nectar` = red alliance, `blue_nectar` = blue alliance). The repo
README is the source of truth for everything technical; this book helps a new
team (a) pick a detector track and (b) get from bench setup to a robot that
picks up balls on its own.

- **Get started** — pick a track, set up the camera, deploy
- **Updates** — what changed between model/docs releases
- **[API](api/README.md)** — the `ftc_ball_chase_lib` classes teams wire into
  their autonomous: `BallTracker`, the chase drivers, and the fluent wrapper

## Which detector track?

* [Which detector track?](detector_tracks/README.md) — the decision guide
* [Limelight 3A (SSD)](detector_tracks/limelight_3a_ssd/README.md) — [set it up](detector_tracks/limelight_3a_ssd/limelight_3a_setup.md)
* [PC / ONNX Runtime (YOLO)](detector_tracks/onnx_pc/README.md) — [run it on a PC](detector_tracks/onnx_pc/onnx_run_pc.md)
* [Control Hub (OpenCV + Lab)](detector_tracks/control_hub/README.md) — [HSV setup](detector_tracks/control_hub/hub_hsv_setup.md) · [Lab setup](detector_tracks/control_hub/hub_lab_setup.md)
* [Collecting training data](training_data.md)
* [Deployment checklist](deployment_checklist.md) — what ships where

Classes are always `yellow_pollen`=0, `red_nectar`=1, `blue_nectar`=2.

## Robot doesn't just detect — it picks

Detection alone doesn't score points. The
[`ftc_ball_chase_lib`](../ftc_ball_chase_lib/README.md) package turns a sighting
into a robot action: `BallTracker` picks one ball per frame, the chase drivers
(`BallChaseController`, `BallChaseFollower`) drive to it, and the wrapper gives
your autonomous a one-line, readable collect API. That integration is the
[**API**](api/README.md) tab of this book.

## In action

Same kickoff footage through each detector:

![Limelight 3A (SSD)](.gitbook/assets/ssd_mobilenetv2.gif)

![Control Hub (OpenCV)](.gitbook/assets/control_hub_cv_example_1.gif)

![Control Hub (Lab)](.gitbook/assets/cielab_demo.gif)

The Limelight 3A model in the real world:

![Limelight 3A in action](<.gitbook/assets/ssd_in_action (1).gif>)