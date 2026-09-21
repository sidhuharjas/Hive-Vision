# Set up the Control Hub (OpenCV HSV)

Model-free color detection that runs on the robot's Control Hub itself — no
coprocessor, no TensorFlow, no model. The shipped HSV config
(`cv/tools/hsv_tuned.json`) was learned from the YOLO model's detections, so
both tracks agree on where the balls are.

> **Candidate detector, not a reliable ball detector.** Measured precision on
> the dev video: yellow 40%, red 26%, blue 66%. Treat every detection as a
> _candidate_ — require 3–5 consecutive frames with the same normalized
> position before the robot acts.

## What you need

| Thing | What it's for |
|-------|---------------|
| `cv/TeamCode/BallDetectorPipeline.java` | the VisionProcessor drop-in |
| `cv/tools/hsv_tuned.json` | shipped config (ranges + gates) |
| An FTC SDK project with a webcam | `VisionPortal` + EasyOpenCV |

## Step 1 — Copy the pipeline into your project

Copy `cv/TeamCode/BallDetectorPipeline.java` into your FTC SDK project's
`TeamCode/` folder.

## Step 2 — Register it in an OpMode

```java
import org.firstinspires.ftc.vision.VisionPortal;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Size;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;

BallDetectorPipeline baller = new BallDetectorPipeline();

VisionPortal portal = new VisionPortal.Builder()
        .setCamera(hardwareMap.get(WebcamName.class, "Webcam 1"))
        .setCameraResolution(new Size(1280, 720))
        .addProcessor(baller)
        .enableLiveView(true)
        .build();
```

## Step 3 — Read detections each loop

```java
BallDetectorPipeline.BallBlob red = baller.bestOf(BallDetectorPipeline.BallColor.RED);
if (red != null) {
    telemetry.addData("red", "cx=%.2f cy=%.2f", red.cxNorm, red.cyNorm);
}
```

Steer using `cxNorm` / `cyNorm` — and confirm the candidate across several
frames before acting (the candidate-detector warning above).

## Step 4 — Match your camera resolution

The shipped area constants were tuned at 1920×1080 (≈2 MP); the example OpMode
above runs 1280×720 (≈0.92 MP). Scale the pixel-area thresholds by the
pixel-count ratio:

```
scale = (W_new * H_new) / (1920 * 1080)
```

At 1280×720 that is roughly ×0.444:

| Constant | 1920×1080 | 1280×720 (×0.444) |
|----------|-----------|-------------------|
| `MIN_AREA_PX` | 900 | 400 |
| `MAX_AREA_PX` | 12000 | 5300 |
| `EXPECT_YELLOW_AREA` | 3500 | 1550 |
| `EXPECT_RED_AREA` / `EXPECT_BLUE_AREA` | 4200 | 1865 |

Adjust these in `BallDetectorPipeline.java` before deploying at a different
resolution.

## (Optional) Retune on your field

Hive Vision ships pre-tuned ranges, but lighting changes with every field:

```bash
# preview what the on-hub pipeline will see on your footage
python cv/tools/realtime_cv.py --source path/to/match.mp4

# live-tune H/S/V ranges on a real frame or webcam
python cv/tools/hsv_tuner.py path/to/frame.jpg
```

`1/2/3` pick yellow/red/blue, `r` prints the range, `s` saves a preview. Copy
the tuned values into the static fields at the top of
`BallDetectorPipeline.java`.

## What to know before you trust it

- Keep the field white-balanced so red and blue stay distinct (pure color is a
  luminance-coupled signal).
- Deep shadows break it — a shadowed ball reads as a different color. That's
  the [Lab track's](hub_lab_setup.md) job; a ball so dark even Lab can't see it
  belongs to the [Limelight 3A](limelight_3a_setup.md).
- Full workflow, tuning notes, and measured recall/precision are in
  [cv/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/cv/README.md).