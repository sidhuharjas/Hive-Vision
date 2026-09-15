# Hive Vision — Lab track (Control Hub, middle ground)

A middle track between the two shipped Hive Vision paths:

| track | what | cost |
|-------|------|------|
| YOLO (Limelight) | full CNN, shape + context | coprocessor required |
| **Lab (this)** | CIELAB chromaticity hue + adaptive chroma floor | runs on the Control Hub, no model |
| HSV (Control Hub) | 3 raw HSV boxes, luminance-coupled | cheapest, shadow-fragile |

The Lab detector converts each frame `RGB→Lab` **once**, computes one per-pixel
chromaticity hue `atan2(b*, a*)` and chroma `sqrt(a*² + b*²)`, and thresholds
them with constants that were **learned from the flagship YOLO model** on real
match footage — so it inherits the YOLO's color language while running with
zero model inference on the hub. Because lightness (L*) is a separate channel,
a ball in shadow keeps its hue and simply needs a *lower* chroma bar, which the
detector sets adaptively from the frame's median brightness.

Deep-shadow caveat: when a ball's chroma collapses toward zero (sensor stops
reporting color) no color-only detector can see it — that is the YOLO track's
job. Mid shadows are exactly where this track beats HSV. Measured head-to-head
numbers are in [`docs/lab_detector_report.md`](docs/lab_detector_report.md).

## Files

```
lab/
  TeamCode/LabBallDetectorPipeline.java   drop-in VisionProcessor (Control Hub)
  tools/
    lab_lib.py            shared core (features, mask, blobs, adaptive gain)
    fit_lab_from_yolo.py  learn hue bands/floors/gates from the YOLO truth
    eval_lab_vs_yolo.py   recall/precision + dark-frame recall vs YOLO truth
    sweep_lab.py          quick threshold sweep (scl × floor)
    truth_lib.py          generates + caches the YOLO truth (conf ≥ 0.35)
    lab_tuner.py          live chromaticity tuner (image or webcam)
    detect_video_realtime.py  step-through viewer with YOLO-truth overlay
    realtime_lab.py       live viewer (video or webcam)
    lab_tuned.json        shipped learned constants
  docs/
    yolo_truth_lab.json   cached YOLO truth (293 frames, 400 boxes)
    lab_detector_report.md  evaluation write-up
```

## Using it on the robot

Copy `TeamCode/LabBallDetectorPipeline.java` into `TeamCode/`, then:

```java
LabBallDetectorPipeline pipeline = new LabBallDetectorPipeline();
VisionPortal portal = new VisionPortal.Builder()
        .setCamera(webcam)
        .addProcessor(pipeline)
        .setCameraResolution(new Size(1280, 720))
        .build();
...
LabBallDetectorPipeline.BallBlob ball = pipeline.bestOf(LabBallDetectorPipeline.BallColor.YELLOW);
if (ball != null) { /* steer using ball.cxNorm, ball.cyNorm */ }
```

Confirm a candidate across a few frames before moving — the detector is a
candidate signal, not ground truth (same contract as the HSV track).

Area gates are stored per-1080p and scaled automatically to the live camera
resolution; `aspect`/`fill` are already scale-free.

## Re-tuning

The shipped `lab_tuned.json` was fitted to the dev match video. On your field:

```bash
# live look at what the Control-Hub pipeline will see
python tools/realtime_lab.py --source path/to/match.mp4

# live tuning of the hue band + floors on a real frame/webcam
python tools/lab_tuner.py path/to/frame.jpg

# or re-learn everything from the YOLO model as described in the report
python tools/fit_lab_from_yolo.py --source path/to/match_raw.mp4 --out lab_tuned.json \
  --truth-json docs/yolo_truth_lab.json
```

Copy the printed values into `LabBallDetectorPipeline.java`'s `YELLOW_LO`,
`RED_LO`, `BLUE_LO`, `YELLOW_SAT`, `RED_SAT`, `BLUE_SAT` and the `REF_L`
constants.

## Ground truth / license notes

`docs/yolo_truth_lab.json` is derived by running the **flagship YOLO model**
(`../yolo/weights/best.pt`, Ultralytics, AGPL-3.0 — see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)) over development footage;
the JSON itself is a list of frame-wise ball boxes and carries no model weights.