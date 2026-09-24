# Chasers — the drivers you integrate in an autonomous

Three classes in `final/` take a `BallTracker` sighting and drive the robot.
They are the integration surface for your auto/teleop loop.

| Which one | Use when | Localization |
|-----------|----------|--------------|
| `BallChaseController` | a no-odometry chase (teleop assist, or a short pickup phase in auto) | none — camera only |
| `BallChaseFollower` | a Pedro Pathing auto wants balls on the way to score | Pedro pose (SCAN projects to field coords) |
| `BallHunt` | "get 2 red balls, 20 s max" with zero wiring | Pedro poses the approach |

All three share the same perception core, so behavior is consistent whichever
you pick.

---

## `BallChaseController` — no-odometry chase

`SEARCHING -> CHASING -> COASTING -> PICKUP -> SEARCHING ... -> DONE`

| State | Behavior |
|-------|----------|
| `SEARCHING` | rotate (`SEARCH_TURN`); ends early if we dead-reckon `SEARCH_SWEEP_DEG` with nothing in view (empty field → clean `DONE`) |
| `CHASING` | aim (P on `tx`), drive forward (P on distance) once roughly aimed |
| `COASTING` | ball lost close to the intake → keep driving straight `COAST_MS`, don't spin away |
| `PICKUP` | stop, intake `PICKUP_DWELL_MS`, count the pickup |
| `DONE` | `pickups >= maxPickups` (or `start()` never called) |

```java
BallChaseController c = new BallChaseController(tracker, lf, rf, lb, rb, intakeOrNull);
c.setMaxPickups(2);                 // 0 = unlimited
c.start();
while (opModeIsActive() && !c.isDone()) { c.update(); c.addTelemetry(telemetry); }
c.abort();                          // motors back at zero
```

| Method | What it does |
|--------|--------------|
| `BallChaseController(tracker, lf, rf, lb, rb, intakeOrNull)` | build on the four drive motors (intake may be null) |
| `configureMecanumDirections(lf, rf, lb, rb)` | static: set standard symmetric-mecanum directions unless your drivetrain is flipped |
| `setMaxPickups(int n)` | stop after `n` pickups; `0` = unlimited |
| `getPickups()` | pickups counted so far |
| `getState()` | current `State` |
| `isDone()` / `isActive()` | done / running |
| `start()` | begin (resets pickups + tracker lock) |
| `abort()` | stop and zero motors |
| `update()` | **one call per loop** — poll tracker, move motors |
| `addTelemetry(Telemetry t)` | stream state to the driver station |

Tunables (all `public static`): `STOP_DIST`, `AIM_TOL_DEG`, `DRIVE_MIN_TX`,
`DRIVE_KP`, `MIN_FWD`, `MAX_FWD`, `TURN_KP`, `MIN_TURN`, `MAX_TURN`,
`SEARCH_TURN`, `SEARCH_SWEEP_DEG`, `COAST_MAX_DIST`, `COAST_POWER`,
`COAST_MS`, `PICKUP_DWELL_MS`, `INTAKE_POWER`.

---

## `BallChaseFollower` — Pedro plans, camera finishes

`SCAN -> TRAVEL -> TURN -> CHASE -> PICKUP -> DONE`

| State | Behavior |
|-------|----------|
| `SCAN` | stop and settle, average a few Limelight frames, **project each detection to a field position** via the Pedro pose; forget balls that should be visible but aren't |
| `TRAVEL` | Pedro drives to `APPROACH_DIST` short of the nearest remembered ball, facing it |
| `TURN` | face a close ball, or sweep when nothing is known |
| `CHASE` | camera-only final approach (aim `tx`, range `ty`) — localization error no longer matters; coasts if the ball vanishes at the intake |
| `PICKUP` | stop, intake, clear that ball from memory → back to `SCAN` |
| `DONE` | `maxPickups` reached or the time budget ran out |

```java
BallChaseFollower h = new BallChaseFollower(follower, tracker, intake);
h.setAllowedClasses(BallTracker.CLASSES_RED_BLUE);
h.setMaxPickups(3); h.setTimeBudgetSec(20);
h.start();
while (opModeIsActive() && !h.isDone()) {
    follower.update();   // YOU own follower.update() — this class never calls it
    h.update();
    h.addTelemetry(telemetry);
}
h.abort();
```

| Method | What it does |
|--------|--------------|
| `BallChaseFollower(Follower follower, BallTracker tracker, DcMotor intakeOrNull)` | build on a Pedro `Follower` |
| `BallChaseFollower(Follower follower, Limelight3A limelight, DcMotor intakeOrNull)` | convenience — own tracker created from the Limelight |
| `setAllowedClasses(Set/Integer...)` | forwards to the tracker (alliance filter) |
| `setMaxPickups(int n)` | stop after `n` pickups |
| `setTimeBudgetSec(double s)` | hard hunt budget before handing back control |
| `getPickups()` / `getState()` / `isDone()` / `isActive()` | query |
| `start()` / `abort()` / `update()` / `addTelemetry(t)` | lifecycle (call `update()` every loop; never own `follower.update()`) |

Field projection uses `HFOV_DEG`/`VFOV_DEG` plus the camera mount offsets
(`CAM_X_OFFSET`/`CAM_Y_OFFSET`) rotated from the camera frame. Conventions:
Pedro inches, heading radians CCW+, robot +X forward / +Y left.

---

## `BallHunt` — the one-liner

```java
int picked = new BallHunt(follower, limelight, intake)
        .reds().collect(2).within(20)
        .go(this);          // blocking: owns follower.update(), returns balls picked
```

| Verb | What it does |
|------|--------------|
| `.reds()` / `.blues()` / `.yellows()` | restrict to one class |
| `.alliance()` (default) | red + blue |
| `.everything()` | red + blue + neutral yellow |
| `.collect(int n)` | balls per hunt (`n <= 0` = unlimited) |
| `.within(double sec)` | hard time budget (default 15 s) |
| `.go(LinearOpMode)` | **blocking**: runs the hunt, streams telemetry, aborts cleanly, returns balls picked |
| `start()` / `update()` / `abort()` / `isDone()` | loop-driven alternative (you keep your loop; you own `follower.update()`) |
| `got()` | balls picked so far |
| `state()` / `addTelemetry(t)` | current state / telemetry |

`BallHunt(Follower, Limelight3A, DcMotor)` and
`BallHunt(Follower, BallTracker, DcMotor)` constructors.

## BallMath (reference)

`final/BallMath.java` — a pure-Java mirror of the projection math (inches/
degrees, `groundRange`, target range/bearing) with a self-contained `main()`
self-test. Independent of the drives; useful as the math reference.