# API — FTC ball chase library

The [`ftc_ball_chase_lib/`](../ftc_ball_chase_lib/README.md) package turns a
Limelight 3A ball sighting into field coordinates a robot can act on. Two
layers:

- **`final/`** — the ship-ready perception + drive classes (FTC SDK 11.x,
  optionally Pedro Pathing 2.x). Copy them into
  `org.firstinspires.ftc.teamcode`.
- **`wrapper/`** — a fluent, drivetrain-agnostic verb API on top of Pedro or
  mecanum drivetrains (`org.firstinspires.ftc.teamcode.wrapper`).

Full prose documentation (state machines, lock semantics, tuning table):
[`ftc_ball_chase_lib/MODULES.md`](../ftc_ball_chase_lib/MODULES.md). The cheat
sheet is below.

## `BallTracker` — perception core

```java
BallTracker tracker = new BallTracker(Limelight3A limelight);   // or a custom DetectionSource
tracker.setAllowedClasses(BallTracker.CLASS_RED, BallTracker.CLASS_BLUE);
BallTracker.Sighting s = tracker.update();   // once per loop; null = nothing usable
```

- `Sighting` → `classId`, `txDeg`, `tyDeg`, `confidence`, `predicted`,
  `dist` (camera-relative inches via `groundRange`).
- Confidence (`MIN_CONF`) + staleness (`MAX_STALENESS_MS`, ms) gating; a
  color-bound target lock stops two-ball flip-flop; `predicted=true` while
  coasting on a momentarily-lost locked ball.
- Classes: `0 = yellow_pollen` (neutral), `1 = red_nectar`, `2 = blue_nectar`.
- Testable on a laptop via `BallTracker(DetectionSource)` + a scripted source.

## `BallChaseController` — no-odometry chase (teleop / simple auto)

```java
BallChaseController chase = new BallChaseController(tracker, lf, rf, lb, rb, intakeOrNull);
chase.setMaxPickups(2);        // 0 = unlimited
chase.start();
while (opModeIsActive() && !chase.isDone()) { chase.update(); chase.addTelemetry(telemetry); }
chase.abort();                 // hands motors back at zero
```

State machine `SEARCHING -> CHASING -> COASTING -> PICKUP -> ... -> DONE`.
A CCW sweep cap (`SEARCH_SWEEP_DEG`) ends an empty-field hunt instead of
spinning forever.

## `BallChaseFollower` — hybrid collect for auto (Pedro plans, camera finishes)

```java
BallChaseFollower hunt = new BallChaseFollower(follower, tracker, intakeOrNull);
hunt.setMaxPickups(3); hunt.setTimeBudgetSec(20);
hunt.start();
while (opModeIsActive() && !hunt.isDone()) { follower.update(); hunt.update(); }
hunt.abort();
```

States `SCAN -> TRAVEL -> TURN -> CHASE -> PICKUP -> DONE`. YOU own
`follower.update()`.

## `BallHunt` — one-line collect

```java
int picked = new BallHunt(follower, limelight, intake)
        .reds().collect(2).within(20)
        .go(this);        // blocking; owns follower.update(); returns balls picked
```

Verbs: `.reds()` `.blues()` `.yellows()` `.alliance()` (default)
`.everything()` `.collect(n)` `.within(sec)`; loop-driven users call
`start()`/`update()`/`abort()` instead.

## `wrapper/` — fluent verbs

```java
if (hunt.canSee(RED)) hunt.grab(RED).within(6).go(this);
else                  hunt.scan().then(hunt.grabNearest()).go(this);
hunt.grabTwo(RED, BLUE).thenReturnTo(scorePose).go(this);
```

- Finding: `canSee` `count` `findNearest` `find` `findLeftmost` `findRightmost`
  `findBiggest` `findBestScore`
- Grabbing: `grabNearest` `grab(color)` `grab(target)` `grabAll` `grabUpTo`
  `grabTwo` `and`
- Searching: `scan` `scanLeft` `scanRight` `searchAt` `lookFor(color).orGiveUp(sec)`
- Moving: `approach` `alignTo` `backOff(inches)` `nudge`
- Intake: `intakeOn` `intakeOff` `reverse(sec)` `isFull` · Chaining:
  `ifSeen(color).grab(color)` `orElse` `then` `thenReturnTo` `within` `go`

Chains are one-shot and replayable until `clear()`. `then(other)` moves another
wrangler's steps onto your chain (pickups stay counted on `other`).
`withPickupSensor(...)` can gate pickups on a beam-break/current spike.

## Verify on a laptop

```powershell
.\ftc_ball_chase_lib\compile_check.cmd -runwrapper   # compile + behavior tests
.\ftc_ball_chase_lib\compile_check.cmd -runmath      # math round trips
```

Calibration constants (camera mount, gating, gains) are `public static` in
`BallTracker` / `BallChaseController` / `BallChaseFollower` — see the
[calibration table](../ftc_ball_chase_lib/README.md#calibrate-before-running).