# Limelight 3A ball chase — BallTracker + drivers

Shared perception + drive logic to make a robot pick up Uberbuzz (TCR:
`yellow_pollen` / `red_nectar` / `blue_nectar`) balls using a Limelight 3A SSD
neural detector. Everything is camera-relative or field-coordinate; no extra
helper download needed (the FTC SDK ships `Limelight3A` natively). Works with
FTC SDK **11.x** and **Pedro Pathing 2.x**.

## Layout

- **`final/`** — the ship-ready library. `BallTracker` (perception), the two
  chase drivers, and `BallHunt`; copy them into your
  `org.firstinspires.ftc.teamcode` package.
- **`wrapper/`** — the fluent, drivetrain-agnostic verb API (`BallWrangler`
  + `PedroWrangler` / `MecanumWrangler`); see [MODULES.md](MODULES.md).
- **[MODULES.md](MODULES.md)** — full class-level reference (state machines,
  lock semantics, verb API) that used to live in the Java file banners.

## Using it (teleop)

```java
BallTracker tracker = new BallTracker(limelight);
tracker.setAllowedClasses(BallTracker.CLASSES_RED_BLUE);   // or CLASSES_ALL

BallChaseController chase = new BallChaseController(tracker, lf, rf, lb, rb, intake);
chase.setMaxPickups(3);
chase.start();
while (opModeIsActive() && !chase.isDone()) {
    chase.update();
    chase.addTelemetry(telemetry);
}
chase.abort();
```

## Using it (auto, with Pedro)

```java
// YOU own follower.update() - this class never calls it
BallChaseFollower hunt = new BallChaseFollower(follower, tracker, intake);
hunt.setAllowedClasses(BallTracker.CLASSES_RED_BLUE);
hunt.setMaxPickups(3); hunt.setTimeBudgetSec(20);
hunt.start();
while (opModeIsActive() && !hunt.isDone()) {
    follower.update();
    hunt.update();
    hunt.addTelemetry(telemetry);
}
hunt.abort();
```

## Using it (auto, the one-liner)

For an autonomous that just wants a few balls, `BallHunt` wraps the same
state machine so the whole hunt is one line — it owns `follower.update()`,
streams telemetry, and aborts cleanly:

```java
@Autonomous
class MyAuto extends LinearOpMode {
    @Override public void runOpMode() {
        Follower follower = FtcConfig.buildFollower(this);   // YOUR team's follower
        Limelight3A limelight = hardwareMap.get(Limelight3A.class, "limelight");
        limelight.start();

        // "get 2 red balls, 20 seconds max" — drives the whole approach:
        int picked = new BallHunt(follower, limelight, intake)
                .reds().collect(2).within(20)
                .go(this);

        // ...drive to the backdrop and score them...
    }
}
```

Fluent options: `.reds()` `.blues()` `.yellows()` `.alliance()` (default)
`.everything()`  ·  `.collect(n)` balls (≤0 = unlimited)  ·  `.within(sec)`.
Loop-driven users can instead `start()` / `update()` / `abort()` / `isDone()`.

## Calibrate before running

| Constant | Where | Meaning |
|----------|-------|---------|
| `CAM_PITCH_DEG`, `CAM_H`, `BALL_H` | `final/BallTracker.java` | Camera tilt below horizontal, lens height, ball-center height (inches). |
| `CAM_X_OFFSET`, `CAM_Y_OFFSET` | `final/BallTracker.java` | Camera mount offset from robot center (forward +X, LEFT +Y, inches). Applied when projecting field coords in `wrapper/BallWrangler.projectBall` and `final/BallChaseFollower` field projection. |
| `MIN_CONF`, `MAX_STALENESS_MS` | `final/BallTracker.java` | Detection gating. `MAX_STALENESS_MS=120`; the SDK reports staleness in MILLISECONDS. Check the 3A firmware: confidence 0-1 or 0-100? |
| Class ids 0/1/2 | `final/BallTracker.java` | CONFIRM against the Limelight web UI pipeline label list — nothing checks the model at runtime. |
| Drive/turn/coast gains | `final/BallChaseController.java` / `final/BallChaseFollower.java` | All `public static`, tune in place or via a config system. |
| `HFOV_DEG`/`VFOV_DEG`, pod/offset geometry | `final/BallChaseFollower.java` | Camera FOV and mount offsets for field projection. |

## Compile check (structure verification)

The whole tree (`final/` + `wrapper/`) is verified to compile with `javac`
against the official jars (FTC SDK 11.2.1 + Pedro Pathing 2.1.2). On this
dev box the jars and a JDK are staged under `%LOCALAPPDATA%\ftc_compile`, so
just run:

```powershell
.\compile_check.cmd -runmath      # compile everything + run the BallMath self-test
.\compile_check.cmd -runwrapper   # compile everything + run the wrapper logic self-test
```

For another machine, pull the Maven Central coordinates into a classpath
(AARs ship a `classes.jar` — extract it onto the classpath):

- `org.firstinspires.ftc:RobotCore:11.2.1`, `org.firstinspires.ftc:Hardware:11.2.1`
- `com.pedropathing:core:2.1.2`, `com.pedropathing:ftc:2.1.2`

```powershell
javac -cp "RobotCore.jar;Hardware.jar;core-2.1.2.jar;ftc.jar" -d out `
  final\*.java wrapper\*.java
```