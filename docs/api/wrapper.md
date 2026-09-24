# Wrapper — the fluent verb API

`wrapper/BallWrangler.java` (package `org.firstinspires.ftc.teamcode.wrapper`) —
a drivetrain-agnostic, fluent API for ball collecting. Subclasses supply only
the motion; everything else (perception, deciding, intake, chaining) lives in
`BallWrangler`.

```java
BallHunt hunt = new BallHunt(follower, limelight, intake);
if (hunt.canSee(RED)) hunt.grab(RED).within(6).go(this);
else                  hunt.scan().then(hunt.grabNearest()).go(this);
hunt.grabTwo(RED, BLUE).thenReturnTo(scorePose).go(this);
```

## Constructors

| Signature | Notes |
|-----------|-------|
| `BallWrangler(Limelight3A, DcMotor intakeOrNull)` | own tracker from the Limelight |
| `BallWrangler(Limelight3A, DcMotor, FullSensor)` | + full-intake sensor |
| `BallWrangler(BallTracker, DcMotor, FullSensor)` | bring your own tracker |

`FullSensor.isFull()` short-circuits `grabAll()`/`grabUpTo()`.
`PickupSensor` (set via `withPickupSensor`) confirms a pickup physically.

## Colors (alliance filter)

| Method | What it does |
|--------|--------------|
| `reds()` · `blues()` · `yellows()` | chase only that class (chain from here) |
| `alliance()` | red + blue (default) |
| `everything()` | red + blue + neutral yellow |
| `setColors(Integer... ids)` | arbitrary class set |

## Sensing — look, don't drive

| Method | What it does |
|--------|--------------|
| `canSee()` / `canSee(BallColor)` | any / that-color ball visible this frame |
| `count()` / `count(BallColor)` | how many |
| `findNearest()` | nearest allowed ball (`Target` or null) |
| `find(BallColor c)` | nearest of one color |
| `findLeftmost()` / `findRightmost()` | by camera bearing |
| `findBiggest()` | largest apparent area (else nearest) |
| `findBestScore()` | best `scoreOf` |
| `findNearestAtPose()` | `findNearest()` with field coords attached (needs localization) |

A `Target` exposes `color`, `bearingDeg`, `elevDeg`, `distIn`, `confidence`,
`score`, `fieldX/fieldY` (NaN until projected), plus `withField(x,y)` and
`hasField()`.

## Scoring & projection

| Method | What it does |
|--------|--------------|
| `static scoreOf(distIn, bearingDeg, confidence)` | lower = better: closer, on-axis, confident balls win |
| `static projectBall(txDeg, tyDeg, RobotPose)` | camera sighting → field coordinates (camera mount offsets rotated in) |
| `atField(Target t, RobotPose pose)` | attach field coords to a sighting |

`RobotPose(double x, double y, double headingRad)` (+ `.inDeg(deg)` helper)
is the wrapper's pose type. `BallColor` is the enum: `RED/BLUE/YELLOW` with
`.classId` and `fromClassId(int)`.

## Grabbing — drive + intake

| Method | What it does |
|--------|--------------|
| `grabNearest()` | chase the nearest allowed ball |
| `grab(BallColor c)` | chase nearest of a color |
| `grab(Target t)` | chase a specific sighting |
| `grabLeftmost()` / `grabRightmost()` | chase by edge |
| `grabAll()` | keep grabbing until nothing's seen or `FullSensor` trips |
| `grabUpTo(int n)` | grab at most `n` balls (re-gathers intelligently) |
| `grabTwo(BallColor a, BallColor b)` | `grab(a)` then `grab(b)` |
| `and(BallColor c)` | synonym for `grab(c)` — reads like prose (`grabTwo` uses it) |

A grab step ends when: the pickup dwell completes and a ball is counted, the
ball is lost for `CHASE_LOST_MS` (`giveUp`), or the per-chase budget elapses.

## Searching

| Method | What it does |
|--------|--------------|
| `scan()` | sweep right to find any allowed ball, then stop facing it |
| `scanLeft()` / `scanRight()` | sweep left/right; stops at the sweep limit, not the timeout |
| `lookFor(BallColor c)` | scan until that color appears |
| `searchAt(RobotPose pose)` | drive to a pose, settle, scan (needs localization) |

## Moving around a ball

| Method | What it does |
|--------|--------------|
| `approach(Target t)` | drive to `STOP_DIST` of the ball, stop short, intake off — no count |
| `alignTo(Target t)` | turn to face the ball without driving forward |
| `alignTo(BallColor c)` | align to the nearest of a color |
| `nudge(Target t)` | creep forward a set distance, intake running |
| `backOff(double inches)` | reverse that many inches (time-based, no encoder needed) |

## Intake

| Method | What it does |
|--------|--------------|
| `intakeOn()` | spin the intake |
| `intakeOff()` | stop it |
| `reverse(double sec)` | run it backwards for `sec` (un-jam) |
| `isFull()` | true when the `FullSensor` reports full |
| `withPickupSensor(PickupSensor s)` | require physical confirmation before counting a pickup / reporting success (beam break or current spike); absent = optimistic count |

## Chaining & conditions

| Method | What it does |
|--------|--------------|
| `ifSeen(BallColor c)` / `when(BallColor c)` | start a `When` block |
| `orElse(Runnable fallback)` | run the fallback if the conditional step does NOT win |
| `within(double sec)` | hard budget for the whole scheduled chain |
| `orGiveUp(double sec)` | per-step timeout for the grab/scan steps |
| `then(RobotPose pose)` | go to a pose next (needs localization; fails fast without) |
| `then(BallWrangler other)` | **cross-chain**: move `other`'s scheduled steps onto this chain — pickups still count on `other` |
| `thenReturnTo(RobotPose pose)` | `then(pose)` — "and come back here" spelling |

### `When` (from `ifSeen(color)`) methods

`grab(color)` · `approach(color)` · `backOff(inches)` — the guarded verb (runs
only if the color was seen). `.orElse(fallback)` fires when the guard fails.
Delegated verb-style: `within(sec)`, `orGiveUp(sec)`, `then(pose)`,
`then(other)`, `thenReturnTo(pose)`, `go(opMode)`.

## Lifecycle & queries

| Method | What it does |
|--------|--------------|
| `go(LinearOpMode op)` | **blocking**: runs the schedule, owns the loop (Pedro: includes `follower.update()`), returns balls picked |
| `start()` | begin pumping (loop-driven use) |
| `update()` | one step of the schedule (loop-driven) |
| `abort()` | stop motors + intake, mark inactive |
| `isDone()` / `isActive()` | finished / running |
| `getPickups()` / `got()` | balls collected |
| `lastError()` | error string from the last failed step (e.g. pose step without localization) |
| `scheduled()` | verbs still queued (debug) |
| `colorOf(Target t)` | `Target` → `BallColor` |
| `addTelemetry(Telemetry t)` | stream chain state |

## Semantics that matter

- **One-shot** — `go()`/`update()` leave steps queued, so `go()` again REPLAYS
  them; `clear()` empties the chain for the next schedule.
- **Timers start on run** — each step's timeout/last-seen clock begins when the
  step first ACTUALLY runs, not when it was built.
- **Cross-chain** — `then(other)` steps stay bound to `other`; keep
  `orElse`/fallbacks on the **receiving** wrangler.

## Subclasses (the motion)

| Class | What it supplies |
|-------|------------------|
| `PedroWrangler(Follower, BallTracker\|Limelight3A, DcMotor[+ FullSensor])` | pose via the Pedro `Follower`; `BallHunt` extends it (adds `collect(n)`, `state()`) |
| `MecanumWrangler(BallTracker\|Limelight3A, lf, rf, lb, rb, DcMotor intakeOrNull[, PoseRouter, HeadingSource, FullSensor])` | dead-reckoned mecanum: integrated heading + feet estimates so `backOff`/pose steps work without a localizer; `configureMecanumDirections(...)` static helper |

Both inherit every method above unchanged.