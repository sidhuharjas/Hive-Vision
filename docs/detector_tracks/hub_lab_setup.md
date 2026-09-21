# Set up the Control Hub (Lab)

The shadow-robust middle track: CIELAB chromaticity hue plus an adaptive
chroma floor, with constants **learned from the YOLO model** on real match
footage. Like the HSV track it runs on the Control Hub with zero model
inference; unlike the HSV track it separates lightness from color, so a ball
in shadow keeps its hue and only needs a lower chroma bar.

## What you need

| Thing | What it's for |
|-------|---------------|
| `lab/TeamCode/LabBallDetectorPipeline.java` | the VisionProcessor drop-in |
| `lab/tools/lab_tuned.json` | shipped learned constants |
| An FTC SDK project with a webcam | `VisionPortal` + EasyOpenCV |

## Step 1 — Copy the pipeline into your project

Copy `lab/TeamCode/LabBallDetectorPipeline.java` into your FTC SDK project's
`TeamCode/` folder.

## Step 2 — Register it in an OpMode

```java
LabBallDetectorPipeline pipeline = new LabBallDetectorPipeline();

VisionPortal portal = new VisionPortal.Builder()
        .setCamera(webcam)
        .setCameraResolution(new Size(1280, 720))
        .addProcessor(pipeline)
        .build();
```

## Step 3 — Read detections each loop

```java
LabBallDetectorPipeline.BallBlob ball = pipeline.bestOf(LabBallDetectorPipeline.BallColor.YELLOW);
if (ball != null) { /* steer using ball.cxNorm, ball.cyNorm */ }
```

Same candidate contract as the HSV track: confirm across a few frames before
moving.

## Step 4 — Resolution

No scaling work to do — area gates are stored per-1080p and are scaled
automatically to the live camera resolution; `aspect`/`fill` are already
scale-free.

## (Optional) Retune on your field

```bash
# live look at what the Control-Hub pipeline will see
python lab/tools/realtime_lab.py --source path/to/match.mp4

# live tuning of the hue band + adaptive chroma floor on a real frame/webcam
python lab/tools/lab_tuner.py path/to/frame.jpg

# or re-learn everything straight from the YOLO model
python lab/tools/fit_lab_from_yolo.py --source path/to/match_raw.mp4 \
  --out lab_tuned.json --truth-json docs/yolo_truth_lab.json
```

Copy the printed values into the `YELLOW_LO`, `RED_LO`, `BLUE_LO`,
`YELLOW_SAT`, `RED_SAT`, `BLUE_SAT` and `REF_L` constants at the top of
`LabBallDetectorPipeline.java`.

## What to know before you trust it

- A ball **deep in shadow** loses its chroma entirely and no color-only
  detector — Lab included — can see it. That case belongs to the
  [Limelight 3A](limelight_3a_setup.md). Mid shadows are exactly where this
  track beats HSV.
- The shipped constants were fitted to the dev match video — re-check on
  your field's white balance.
- Measured recall/precision and the dark-frame head-to-head vs HSV are in
  [lab/docs/lab_detector_report.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/lab/docs/lab_detector_report.md).