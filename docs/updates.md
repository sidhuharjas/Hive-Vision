# Updates

Hive Vision ships new model weights and docs on a roughly every-other-week
cadence as fresh match footage, field conditions, and hard negatives arrive.
The repo keeps the full notes; this page collects them in one place.

## Current release — SSD v1.1.0 (Limelight 3A)

The Limelight 3A track runs **SSD-MobileNetV2** (full-int8
`TFLite_Detection_PostProcess` contract, no NMS needed on your side). v1.1.0
was retrained with shadow-hardening data and hard negatives (wiring, robot
chassis, field signage) so shadows and off-field objects are no longer
reported as game elements.

What changed → [RELEASE_NOTES_v1.1.0.md](../RELEASE_NOTES_v1.1.0.md).
Training notes → [`neural-net/UPDATE_03.md`](../neural-net/UPDATE_03.md).

## Release history

| Version | Track | What changed |
|---------|-------|--------------|
| v1.1.0 | Limelight 3A SSD | shadow-hardened retrain, hard negatives, absolute release-asset URLs, docs book reorganized (Get started / Updates / API) |
| v1.0.0 | Limelight 3A SSD | first shipped SSD-MobileNetV2 model + FTC-side chase library |

## FTC-side library

The [`ftc_ball_chase_lib`](../ftc_ball_chase_lib/README.md) classes are versioned
with the repo (not the model). Recent changes:

- `BallTracker` gained a color-bound target lock (a `grab(RED)` hunt can't get
  stolen by a nearby blue ball) and a pluggable `DetectionSource` so the whole
  perception core runs on a laptop with scripted detections.
- `BallChaseController` ends its search after `SEARCH_SWEEP_DEG` of
  dead-reckoned rotation, so an empty field hands control back instead of
  spinning.
- Chase drivers now share one set of tuning constants (`CHASE_*` in
  `BallChaseController`), so teleop and auto behave identically.
- Optional `PickupSensor` (`withPickupSensor(...)`) gates pickup-counting on a
  beam break / intake current spike instead of trusting a phantom pickup.
- Fluent wrapper gained cross-wrangler chaining: `then(other)` moves another
  wrangler's steps onto your chain while counting its pickups on the source.
- Game naming corrected to **BioBuzz**; the old working title is gone.

Run the ships-with-it self-tests:

```powershell
.\ftc_ball_chase_lib\compile_check.cmd -runwrapper
.\ftc_ball_chase_lib\compile_check.cmd -runmath
```

## Contributing data

See the repo README: join the [Hive Vision Discord](https://discord.gg/m2yPTprccv)
and share links in the training-data discussion. Label as `yellow` / `red` /
`blue`, include camera + lighting conditions, and prefer footage permissioned
for redistribution.