# Set up the Limelight 3A

This page gets a Limelight 3A detecting balls and reporting them to your FTC
code. It is written so a team that has never wired a vision camera can work
through it top to bottom at the bench. The detector model and labels already
exist in this repo — you do not train anything.

## What you need

| Thing | What it's for |
|-------|---------------|
| Limelight 3A | the camera; it runs the detector on its own CPU |
| USB-C to USB-A cable | power + data to the Control Hub |
| Control Hub with a free USB 3.0 (blue) port | runs your FTC code |
| Laptop on the robot's Wi-Fi | to open the camera's web UI |
| `best_limelight3a_ssd_mobilenetv2_300x300.tflite` | the detector model (`yolo/weights/`) |
| `labels.txt` | the three class names (`yolo/weights/`) |

The 3A has **no neural-network accelerator**. It runs models on its CPU, and
you must tell the web UI to use the CPU engine — models made for "Coral" or
"Hailo" won't run here.

## Step 1 — Wire it up

1. Plug the USB-C end into the Limelight 3A.
2. Plug the USB-A end into a **blue** (USB 3.0) port on the Control Hub.
3. Power the Control Hub. After about 15–20 seconds the camera's green status
   light settles — that means it has booted.

## Step 2 — Open the web UI

With your laptop on the robot's Wi-Fi:

1. Open a browser and go to `http://limelight.local:5801` — the camera's own
   setup page, not a website.
2. If nothing loads, use the **Limelight Hardware Manager** tool
   (downloads at limelightvision.io), which lists every Limelight it can find;
   double-click yours.
3. Either way you get the setup page with tabs down the left side
   (`Settings`, `Camera`, pipeline tabs).

If you can't reach the UI at all, it is almost always power (green light dark)
or network (laptop not actually on the robot Wi-Fi). See Troubleshooting.

## Step 3 — First-run settings (once)

In the **Settings** tab:

1. Set **Team Number** to your FTC team number.
2. Set a **Hostname** you'll recognize, e.g. `hive-vision-3a`.
3. Save. A static IP is optional but makes events more reliable — the UI keeps
   a fixed address instead of depending on name lookup.

## Step 4 — Load the detector model

1. Pick the pipeline slot you'll use and set **Pipeline Type** to
   **Neural Detector**.
2. **Upload the model**: choose
   `yolo/weights/best_limelight3a_ssd_mobilenetv2_300x300.tflite`.
3. **Upload the labels**: choose `yolo/weights/labels.txt`. The file is one
   class name per line — keep the order exactly as shipped (`yellow_pollen`,
   `red_nectar`, `blue_nectar`); the model was trained in that order.
4. Set **Runtime engine** to **CPU**. Leaving it on "Coral" is the most common
   reason a 3A shows no detections.
5. Set **Detection threshold** to **0.35–0.45**. At 0.25 this model shows
   visible false positives on real footage.
6. Set the **input resolution** to match the model: **300 × 300**.
7. Apply / save. The 3A compiles the model and restarts that pipeline — give
   the stream a moment before checking it.

## Step 5 — Verify in the web UI before writing any code

Look at the camera stream in the web UI. You should see a box and a class name
(`yellow_pollen`, `red_nectar`, `blue_nectar`) drawn over each ball. **If
boxes don't appear here, the problem is the pipeline, not your code** — fix it
at this step.

One honest caveat from the repo's own checklist: this SSD model is verified on
PC runners but not yet on a real 3A. First time on hardware, confirm the
pipeline actually loads and reports detections before match day.

## Step 6 — Read detections in FTC code

The configured device name must match, character for character, the name in
`hardwareMap.get(...)`. In the Driver Station **Configure** app, add a
`Limelight3A` device with a name, then use the same name in code:

```java
import com.qualcomm.hardware.limelightvision.LLResult;
import com.qualcomm.hardware.limelightvision.Limelight3A;

Limelight3A limelight;

@Override
public void init() {
    // "limelight" must equal the device name you configured on the Control Hub
    limelight = hardwareMap.get(Limelight3A.class, "limelight");
    limelight.setPollRateHz(100); // ask the camera for data 100 times/second
    limelight.start();
}

@Override
public void loop() {
    LLResult result = limelight.getLatestResult();
    if (result == null || !result.isValid()) return;

    // getStaleness() is MILLISECONDS in the FTC SDK. 100–120 ms is "fresh".
    // A 100 Hz poll rate does NOT mean every poll gets a new frame — only
    // rely on data that passed the staleness gate.
    if (result.getStaleness() < 120) {
        // getDetectorResults() lists every ball. getTx()/getTy()/getTa()
        // alone only describe the single primary target.
        for (var det : result.getDetectorResults()) {
            telemetry.addData(det.getClassName(), "x=" +
                det.getTargetXDegrees() + " y=" + det.getTargetYDegrees());
        }
    }
}
```

Then filter by your alliance's class in the loop — don't act on every class
the model can see.

## What to know before you trust it

Plain facts, stated up front (most learned the hard way on a robot):

- **Runtime engine must stay on CPU.** The 3A has no neural accelerator. A
  "Coral"-flavored setting produces an empty result with no error.
- **`getStaleness()` is milliseconds in the FTC SDK** (the FRC side of
  Limelight reports microseconds — don't mix the two up). Gate on ~100–120 ms.
- **Polling faster does not invent new frames.** A neural detector runs at
  roughly 10–30 fps; everything between is the same result. Always gate on
  staleness.
- **`getTx/ty/ta` = primary target only.** For all balls, iterate
  `getDetectorResults()`.
- **Confidence can render as 0–1 or 0–100** depending on the surface you're
  looking at. Print it once to telemetry before trusting a constant.
- **labels.txt order must match the training order.** Reordering the label
  file mislabels every class.
- **Threshold is a model property, not a firmware bug.** 0.25 → false
  positives on this footage; 0.35–0.45 is the sweet spot.

## Troubleshooting

- **Can't open the web UI.** Green light on? Laptop on the robot Wi-Fi (not
  your home network)? Try the Hardware Manager; it scans by MAC so it works
  when name lookup fails. Set a static IP in Settings to make this stop
  happening at events.
- **Stream is up but no boxes.** Engine = CPU? Threshold not at 0.35–0.45?
  Input resolution set to 300 × 300? Did the model upload actually finish
  (the UI shows the model name after a successful upload)?
- **Code always says invalid.** Device name mismatch between Configure and
  `hardwareMap.get(...)`, or the code switched to a pipeline slot that doesn't
  hold the detector (`limelight.pipelineSwitch(n)`).
- **Box positions look wrong.** The 3A has two cameras ("eyes"); make sure the
  eye you configured in the web UI is the one mounted where you think it is.

## Official references

- Quick start: <https://docs.limelightvision.io/docs/docs-limelight/getting-started/limelight-3a>
- Neural pipelines (model upload, labels, threshold):
  <https://docs.limelightvision.io/docs/docs-limelight/pipeline-neural/getting-started-with-neural-networks>
- FTC Java guide (init, results, staleness):
  <https://docs.limelightvision.io/docs/docs-limelight/apis/ftc-programming>