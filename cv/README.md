# Hive Vision — Control Hub track (OpenCV, no model)

The Control Hub track of Hive Vision. Raw-OpenCV alternative to the Limelight
track: no coprocessor, no TensorFlow, no extra hardware — color-threshold
detection runs on the FTC **Control Hub** itself via **VisionPortal** (VisionProcessor).
Its HSV ranges are learned from the Limelight track's model detections, so both
tracks agree on where the balls are.

> **⚠️ Candidate detector, not a reliable ball detector.**
> Precision on the development video: yellow 40%, red 26%, blue 66%.
> Treat every detection as a *candidate* — require N consecutive frames
> (e.g. 3‑5) with the same normalized position before acting.

> **Performance:** “~60 FPS” was an informal estimate on a development laptop.
> Real‑world FPS on a Control Hub (1280×720, EasyOpenCV) has not been
> measured; expect lower.

## What's here

| Path | What it is |
|------|------------|
| `tools/hsv_tuned.json` | **Shipped config** (YOLO‑learned ranges + gates) — what the Java pipeline uses. |
| `tools/fit_hsv_from_yolo.py` | Mines HSV ranges straight from the bundled model's detections on your footage. |
| `tools/eval_cv_vs_yolo.py` | Scores any config vs the model's ground truth (uses the cached `docs/yolo_truth.json`; ~25 s). |
| `tools/sample_hsv.py` | Mines per‑color HSV ranges from a YOLO label set + images (needs your dataset). |
| `tools/hsv_tuner.py` | Live tuner: drag H/S/V trackbars to get a mask that only lights up the ball. |
| `tools/realtime_cv.py` | Viewer for the on‑hub detector over your own footage (same keys as the model viewer). |
| `TeamCode/BallDetectorPipeline.java` | **VisionProcessor** implementation — drop into `TeamCode/` of your FTC SDK project. |

## Workflow

1. **(Optional) Re‑mine HSV ranges from fresh footage** — runs the bundled Limelight model and learns the ball pixels:

       python tools/fit_hsv_from_yolo.py --source path/to/match.mp4

2. **Tune live on your actual camera/frame**:

       python tools/hsv_tuner.py path/to/saved_frame.jpg
       python tools/hsv_tuner.py --camera 0

   Keys: `1/2/3` pick yellow/red/blue, `r` prints the range, `s` saves a preview, `q` quits.

3. **Copy the tuned HSV constants** into the static fields at the top of
   `TeamCode/BallDetectorPipeline.java`, then copy that file into `TeamCode/` of your FTC SDK project.

4. **Register the processor in your OpMode** (VisionPortal API):

```java
import org.firstinspires.ftc.vision.VisionPortal;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Size;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;

// in runOpMode() / init:
BallDetectorPipeline baller = new BallDetectorPipeline();

VisionPortal portal = new VisionPortal.Builder()
        .setCamera(hardwareMap.get(WebcamName.class, "Webcam 1"))
        .setCameraResolution(new Size(1280, 720))
        .addProcessor(baller)               // ← VisionProcessor registration
        .enableLiveView(true)
        .build();

// later in loop:
BallDetectorPipeline.BallBlob red = baller.bestOf(BallDetectorPipeline.BallColor.RED);
if (red != null) {
    telemetry.addData("red", "cx=%.2f cy=%.2f", red.cxNorm, red.cyNorm);
}
```

5. **Consume detections** — call `baller.bestOf(BallColor.COLOR)` each loop.
   Remember the candidate‑detector warning above: confirm over several frames.

## Resolution scaling guidance

The shipped constants (`MIN_AREA_PX`, `MAX_AREA_PX`, `EXPECT_*_AREA`) were
tuned on **1920×1080** footage (≈2 MP). The example OpMode uses **1280×720**
(≈0.92 MP). Scale pixel‑area thresholds by the pixel‑count ratio:

```
scale = (W_new * H_new) / (1920 * 1080)
```

For 1280×720: `scale ≈ 0.444`. Suggested 1280×720 values:

| Constant | 1920×1080 | 1280×720 (×0.444) |
|----------|-----------|-------------------|
| MIN_AREA_PX | 900   | 400 |
| MAX_AREA_PX | 12000 | 5300 |
| EXPECT_YELLOW_AREA | 3500 | 1550 |
| EXPECT_RED_AREA / EXPECT_BLUE_AREA | 4200 | 1865 |

Adjust `MIN_AREA_PX`, `MAX_AREA_PX`, `EXPECT_*_AREA` in the Java file
accordingly before deploying at a different resolution.

## How the defaults were found

`tools/fit_hsv_from_yolo.py` runs the fixed YOLO model on the real match
video and mines the *actual* ball pixels:
- hue = dominant‑population band (two‑stage: ±15 ° window around the peak,
  then circular mean ± p95 angular distance; red wraps → two segments)
- S/V floors clamped to recall‑friendly values (V≈45, S≈60‑90)
- expected real‑ball area per color (yellow 3500 px, red/blue 4200 px at 1080p)

Detection gates at `MIN_AREA_PX 900..12000`, near‑square aspect, high fill,
and picks the **most ball‑y** blob per color (closest to expected area,
roundest, penalised at frame edge). Measured vs YOLO ground truth on the
dev video:

| color | recall | precision |
|-------|--------|-----------|
| yellow | 69% | 40% |
| red   | 80% | 26% |
| blue  | 82% | 66% |

## Tuning notes

- Re‑tune live whenever lighting changes:
  `python tools/hsv_tuner.py path/to/frame.jpg`
- Check FN reasons quickly:
  `python tools/eval_cv_vs_yolo.py --config tools/hsv_tuned.json --fnlog reports/fn.csv`
  Reasons: `COLOR_MISS`, `TOO_SMALL/TOO_BIG`, `SHAPE_MISS`,
  `HIT_BUT_BEST_WRONG` (raise `FILL_W`).
- If your camera/framing differs, remeasure expected ball areas
  (`EXPECT_AREA` in `realtime_cv.py` / `EXPECT_*_AREA` in Java).
- Pure color is a **candidate** signal (26‑40 % precision on red/yellow);
  require N consecutive frames before acting (track `cxNorm/cyNorm`).
- Keep the field white‑balanced so red/blue stay distinct.