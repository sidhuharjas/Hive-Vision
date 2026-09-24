# `BallTracker` — perception core

`final/BallTracker.java` · shared by every chase driver. Input: raw Limelight
3A neural-detector output (`DetectionSource`). Output: **one chosen ball per
frame** (`Sighting`), solving the two-ball flip-flop problem.

## Constructors

| Signature | Notes |
|-----------|-------|
| `BallTracker(Limelight3A limelight)` | normal robot use; wraps the Limelight in a `LimelightSource` |
| `BallTracker(DetectionSource source)` | bring your own source — used for laptop unit tests / sim |

`DetectionSource` is the live polling contract: `List<RawDet> latest()` plus
`long stalenessMs()`. `RawDet` carries `classId`, `txDeg`, `tyDeg`,
`confidence`, `area`.

## Class filter

| Method | What it does |
|--------|--------------|
| `setAllowedClasses(Set<Integer> classes)` | replace the whole allowed set (clears any lock) |
| `setAllowedClasses(Integer... classes)` | varargs form — `CLASS_RED, CLASS_BLUE` |
| `addAllowedClass(int classId)` | add one class |
| `isAllowed(int classId)` | is this class chaseable right now? |
| `getAllowedClasses()` | current set |

`CLASS_YELLOW_NEUTRAL = 0`, `CLASS_RED = 1`, `CLASS_BLUE = 2`;
`CLASSES_RED_BLUE` and `CLASSES_ALL` preset sets.

## Frame pipeline

| Method | What it does |
|--------|--------------|
| `Sighting update()` | **call once per loop.** Returns the chosen ball, or `null` when nothing is usable and no lock exists to coast on |
| `Sighting getLast()` | the result of the last `update()` (no new read) |
| `long getStalenessMs()` | age of the latest frame (ms) |
| `List<RawDet> getConfidentResults()` | all detections passing confidence + staleness (any class) |
| `List<RawDet> getGatedDetections()` | same, additionally filtered to the allowed classes |
| `double groundRange(double tyDeg)` | inches from the camera to the ball's ground point for a given elevation angle |
| `void reset()` | drop the lock + last sighting — call when starting a fresh hunt |
| `void lockOn(int classId, double txDeg, double tyDeg, double confidence)` | pin the lock to a specific ball (used by the chase steps' interior logic) |

### `Sighting`

Fields: `classId`, `txDeg` (+ = right), `tyDeg` (+ = up), `confidence`,
`predicted` (true while coasting on a momentarily-missing locked ball),
`dist` — ground range in inches.

## How the lock works

Gating first (`MIN_CONF`, `MAX_STALENESS_MS` — the SDK reports staleness in
**milliseconds**). Then:

1. Among allowed detections, the tracker sticks with the one **angularly
   closest to where the locked ball was last frame** (within `LOCK_GATE_DEG`),
   even if another ball is now nearer — no oscillation between two balls.
2. The lock is **color-bound**: while alive, only detections of the locked
   class may re-adopt it, so `grab(RED)` never chases a nearby blue.
3. If the locked ball can't be matched for `LOCK_LOST_MS`, `update()` can still
   return a **`predicted`** sighting with the last-known angles (driver keeps
   facing it) until the lock drops — then it falls back to nearest allowed.

## Tunables

| Constant | Default | Meaning |
|----------|---------|---------|
| `MIN_CONF` | 0.44 | confidence floor (check firmware scale 0–1 vs 0–100) |
| `MAX_STALENESS_MS` | 120 | drop frames older than this |
| `LOCK_GATE_DEG` | 12.0 | max frame-to-frame angular jump to count as the same ball |
| `LOCK_LOST_MS` | 300 | drop the lock if unmatched this long |
| `CAM_PITCH_DEG`, `CAM_H`, `BALL_H` | 25.0, 9.2, 3.0 | camera tilt below horizontal (deg), lens height, ball-center height (in) |
| `CAM_X_OFFSET`, `CAM_Y_OFFSET` | 0.0, 0.0 | camera FORWARD / LEFT of robot center (in) — applied when projecting field coords |

## Why the `DetectionSource` indirection

`BallTracker` never touches the Limelight directly. The default
`LimelightSource` wraps `getLatestResult()`, but your unit tests / simulator
can feed a scripted source and exercise the exact gating, lock, and prediction
logic on a laptop — that's how the repo's self-tests run.