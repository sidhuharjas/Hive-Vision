# API — `ftc_ball_chase_lib`

The library turns a Limelight 3A ball sighting into a robot action you call
from your own TeleOp or autonomous (`ftc_ball_chase_lib/` in the repo; package
`org.firstinspires.ftc.teamcode`). Two layers:

## `final/` — ship-ready drivers (what your auto integrates)

| Class | Page | What it does |
|-------|------|--------------|
| [`BallTracker`](api/ball_tracker.md) | perception core | raw detections → **one** ball per frame (confidence + staleness gating, color-bound target lock), ground range, staleness |
| [`BallChaseController`](api/chasers.md) | no-odometry chase | pure camera chase state machine for teleop or simple auto — no localization needed |
| [`BallChaseFollower`](api/chasers.md) | Pedro auto hybrid | Pedro plans the approach, the camera finishes; project sightings to field coords via the pose |
| [`BallHunt`](api/chasers.md) | one-liner | the whole collect as a fluent one-liner inside a `LinearOpMode` |

## `wrapper/` — fluent verb API

[`BallWrangler`](api/wrapper.md) (package `org.firstinspires.ftc.teamcode.wrapper`)
lets an autonomous read like prose — `grab(RED).within(6)` — with
`PedroWrangler` / `MecanumWrangler` supplying only the physical motion.

## Who drives which

- **No localizer** → `BallTracker` + `BallChaseController` (teleop or a short
  pickup phase).
- **Pedro Pathing auto** → `BallTracker` + `BallChaseFollower`, or drop
  straight to `BallHunt` / the wrapper for a one-line collect.
- **Sensor gate on pickups** → wrap any chase in `withPickupSensor(...)`.

All tuning constants are `public static` (`MIN_CONF`, `MAX_STALENESS_MS`,
`STOP_DIST`, `CHASE_*`, camera geometry in `BallTracker`, gains in
`BallChaseController`) — hook them to a config system or edit in place. See the
[calibration table](ftc_ball_chase_lib/README.md#calibrate-before-running).