# Control Hub (OpenCV + Lab)

Two model-free hub tracks: `cv/TeamCode/BallDetectorPipeline.java` (HSV) and `lab/TeamCode/LabBallDetectorPipeline.java` (Lab chromaticity). Setup, tuning commands, and viewer tools are in [cv/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/cv/README.md) and [lab/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/lab/README.md). This page only adds the decision + the operator gotchas.

## Pick

* **Lab** when balls sit in shadows; **HSV** when you want the cheapest path on a stable-lighting field.
* A ball deep in shadow is too desaturated for _any_ color-only detector — that case belongs to the Limelight 3A track.

## Gotchas

* Expect **7–11% precision** on a real field. The hub outputs are candidates — confirm each across several frames before the robot acts.
* Both pipelines were fit/tuned from YOLO truth on published footage; re-check recall on _your_ field's white balance before relying on them.
