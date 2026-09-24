# Control Hub (OpenCV HSV + Lab)

Two model-free hub tracks: `cv/TeamCode/BallDetectorPipeline.java` (HSV) and
`lab/TeamCode/LabBallDetectorPipeline.java` (Lab chromaticity). Setup, tuning
commands, and viewer tools are in
[cv/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/cv/README.md)
and
[lab/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/lab/README.md),
or follow the step-by-step: [Set up the HSV track](hub_hsv_setup.md) ·
[Set up the Lab track](hub_lab_setup.md).
This page covers the decision and what to expect from the outputs.

## Pick

- **Lab** when balls sit in shadows; **HSV** when you want the cheapest path on
  a stable-lighting field.
- A ball deep in shadow is too desaturated for *any* color-only detector — that
  case belongs to the Limelight 3A track.

## What to know

- Expect **7–11% precision** on a real field. The hub outputs are candidates —
  confirm each across several frames before the robot acts.
- Both pipelines were fit/tuned from YOLO truth on published footage; re-check
  recall on *your* field's white balance before relying on them.