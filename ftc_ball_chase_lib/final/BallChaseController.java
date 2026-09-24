/*
 * BallChaseController - no-odometry ball chase, packaged as a reusable tool
 * instead of an OpMode (SEARCHING -> CHASING -> COASTING -> PICKUP -> DONE).
 * Class docs moved to MODULES.md (#ballchasecontroller).
 */
package org.firstinspires.ftc.teamcode;

import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorSimple;
import com.qualcomm.robotcore.util.ElapsedTime;
import com.qualcomm.robotcore.util.Range;
import org.firstinspires.ftc.robotcore.external.Telemetry;

public class BallChaseController {

    public enum State { IDLE, SEARCHING, CHASING, COASTING, PICKUP, DONE }

    /* ---- approach ---- */
    public static double STOP_DIST    = 16.0;  // camera-to-ball floor distance at pickup, in
    public static double AIM_TOL_DEG  = 5.0;   // "aimed enough" to pick up
    public static double DRIVE_MIN_TX = 15.0;  // beyond this many degrees off, turn in place only
    public static double DRIVE_KP     = 0.03;  // forward power per inch of distance error
    public static double MIN_FWD      = 0.15;  // overcome static friction
    public static double MAX_FWD      = 0.7;

    /* ---- turning ---- */
    public static double TURN_KP      = 0.025; // power per degree of tx
    public static double MIN_TURN     = 0.08;  // friction floor when outside the tolerance
    public static double MAX_TURN     = 0.6;
    public static double SEARCH_TURN  = 0.30;  // clockwise scan when nothing is visible
    public static double SEARCH_SWEEP_DEG = 360.0;  // give up (DONE) if we dead-reckon this far with nothing in view
    public static double TURN_RATE_RAD_PER_POWER_SEC = 3.5; // heading estimate for the sweep cap

    /* ---- close-range coast + pickup ---- */
    public static double COAST_MAX_DIST = 26.0;   // only coast if last seen within this, in
    public static double COAST_POWER    = 0.25;
    public static long   COAST_MS       = 450;
    public static long   PICKUP_DWELL_MS = 600;
    public static double INTAKE_POWER   = 1.0;

    private final BallTracker tracker;
    private final DcMotor lf, rf, lb, rb;
    private final DcMotor intake;                         // may be null

    private State state = State.IDLE;
    private int pickups = 0;
    private int maxPickups = 0;                           // 0 = unlimited

    private final ElapsedTime lastSeen = new ElapsedTime();
    private final ElapsedTime pickupTimer = new ElapsedTime();
    private double lastSeenDist = Double.MAX_VALUE;

    private final ElapsedTime searchClock = new ElapsedTime();  // sweep-cap bookkeeping
    private double searchSwept = 0;                            // dead-reckoned rotation while searching
    private boolean searchingWas = false;                      // was the LAST drive a search turn?

    public BallChaseController(BallTracker tracker, DcMotor lf, DcMotor rf,
                               DcMotor lb, DcMotor rb, DcMotor intakeOrNull) {
        this.tracker = tracker;
        this.lf = lf; this.rf = rf; this.lb = lb; this.rb = rb;
        this.intake = intakeOrNull;
    }

    /** Standard symmetric mecanum directions (LF F, LB F, RF R, RB R). Call once
     *  in your OpMode init unless a flipped gearbox changes the wiring. */
    public static void configureMecanumDirections(DcMotor lf, DcMotor rf,
                                                  DcMotor lb, DcMotor rb) {
        lf.setDirection(DcMotorSimple.Direction.FORWARD);
        lb.setDirection(DcMotorSimple.Direction.FORWARD);
        rf.setDirection(DcMotorSimple.Direction.REVERSE);
        rb.setDirection(DcMotorSimple.Direction.REVERSE);
    }

    /* ---------------- public API ---------------- */

    public void setMaxPickups(int n) { maxPickups = n; }
    public int getPickups()           { return pickups; }
    public State getState()           { return state; }
    public boolean isDone()           { return state == State.DONE; }
    public boolean isActive()         { return state != State.IDLE && state != State.DONE; }

    /** Begin a fresh hunt: reset pickups and the target lock. */
    public void start() {
        pickups = 0;
        tracker.reset();
        searchSwept = 0;
        searchingWas = false;
        state = State.SEARCHING;
    }

    /** Stop everything: motors zeroed, intake off, back to IDLE. */
    public void abort() {
        drive(0, 0, 0);
        setIntake(0);
        tracker.reset();
        state = State.IDLE;
    }

    /** Drive the state machine one step. Call every loop while active. */
    public void update() {
        if (!isActive()) return;

        if (state == State.PICKUP) {
            doPickup();
            return;
        }

        BallTracker.Sighting s = tracker.update();

        if (s == null) {
            noTarget();
            return;
        }

        searchSwept = 0;        // something is visible: a fresh sweep starts if it vanishes again
        searchingWas = false;

        double tx = s.txDeg;

        if (!s.predicted) {
            // trusted fresh detection: refresh the "last seen" close-distance bookkeeping
            lastSeen.reset();
            lastSeenDist = s.distIn;
        } else {
            // ball is momentarily missing but the lock is still fresh: mirror the
            // real-loss handling for close balls (coast, else pickup); otherwise keep
            // facing the last-known angle without advancing or resetting lastSeen.
            boolean wasClose = lastSeenDist <= COAST_MAX_DIST;
            if (wasClose) {
                if (lastSeen.milliseconds() < COAST_MS) {
                    state = State.COASTING;
                    setIntake(INTAKE_POWER);
                    drive(COAST_POWER, 0, 0);
                    return;
                }
                state = State.PICKUP;
                pickupTimer.reset();
                drive(0, 0, 0);
                return;
            }
            double turn = Range.clip(tx * TURN_KP, -MAX_TURN, MAX_TURN);
            if (Math.abs(turn) < MIN_TURN) turn = Math.copySign(MIN_TURN, tx);
            drive(0, 0, turn);
            return;
        }

        // ---- real target: chase it ----
        boolean aimed = Math.abs(tx) < AIM_TOL_DEG;

        if (s.distIn <= STOP_DIST && aimed) {
            state = State.PICKUP;
            pickupTimer.reset();
            drive(0, 0, 0);
            return;
        }

        double turn = Range.clip(tx * TURN_KP, -MAX_TURN, MAX_TURN);
        if (!aimed && Math.abs(turn) < MIN_TURN) turn = Math.copySign(MIN_TURN, tx);

        double fwd = 0.0;
        if (s.distIn > STOP_DIST && Math.abs(tx) < DRIVE_MIN_TX) {
            fwd = Range.clip((s.distIn - STOP_DIST) * DRIVE_KP, MIN_FWD, MAX_FWD);
        }

        state = State.CHASING;
        setIntake(INTAKE_POWER);
        drive(fwd, 0, turn);
    }

    public void addTelemetry(Telemetry t) {
        BallTracker.Sighting s = tracker.getLast();
        t.addData("chase", "%s  pickups=%d/%s", state, pickups,
                  maxPickups > 0 ? Integer.toString(maxPickups) : "unlimited");
        if (s != null) {
            t.addData("chase target", "tx=%+.1f ty=%+.1f dist=%.1f in cls=%d conf=%.2f%s",
                    s.txDeg, s.tyDeg, s.distIn, s.classId, s.confidence,
                    s.predicted ? " (predicted)" : "");
        } else {
            t.addData("chase target", "none");
        }
    }

    /* ---------------- states ---------------- */

    private void doPickup() {
        drive(0, 0, 0);
        setIntake(INTAKE_POWER);
        if (pickupTimer.milliseconds() < PICKUP_DWELL_MS) return;

        pickups++;
        tracker.reset();                 // drop any lock so the next ball can be picked
        if (maxPickups > 0 && pickups >= maxPickups) {
            setIntake(0);
            state = State.DONE;
        } else {
            state = State.SEARCHING;
        }
    }

    private void noTarget() {
        boolean wasClose = (state == State.CHASING || state == State.COASTING)
                && lastSeenDist <= COAST_MAX_DIST;

        if (wasClose) {
            if (lastSeen.milliseconds() < COAST_MS) {
                state = State.COASTING;
                setIntake(INTAKE_POWER);
                drive(COAST_POWER, 0, 0);   // ball went under the intake: keep going straight
            } else {
                state = State.PICKUP;
                pickupTimer.reset();
                drive(0, 0, 0);
            }
            return;
        }

        state = State.SEARCHING;
        searchTurn();
    }

    /** Rotate in place looking for a ball, but only inside the dead-reckoned
     *  SEARCH_SWEEP_DEG budget: an empty field makes the hunt end cleanly
     *  instead of spinning until the opmode is stopped. Only time actually
     *  spent turning counts (a pickup dwell or chase gap doesn't consume sweep). */
    private void searchTurn() {
        if (searchingWas) {
            double dt = Math.min(searchClock.seconds(), 0.1);
            searchSwept += SEARCH_TURN * TURN_RATE_RAD_PER_POWER_SEC * dt;
            if (searchSwept >= Math.toRadians(SEARCH_SWEEP_DEG)) {
                setIntake(0);
                drive(0, 0, 0);
                state = State.DONE;
                return;
            }
        }
        searchClock.reset();
        searchingWas = true;
        setIntake(0);
        drive(0, 0, SEARCH_TURN);
    }

    /* ---------------- helpers ---------------- */

    /** Mecanum: fwd + = forward, strafe + = right, turn + = clockwise. Normalized. */
    private void drive(double fwd, double strafe, double turn) {
        double pLf = fwd + strafe + turn;
        double pRf = fwd - strafe - turn;
        double pLb = fwd - strafe + turn;
        double pRb = fwd + strafe - turn;

        double max = Math.max(1.0,
                Math.max(Math.max(Math.abs(pLf), Math.abs(pRf)),
                         Math.max(Math.abs(pLb), Math.abs(pRb))));

        lf.setPower(pLf / max);
        rf.setPower(pRf / max);
        lb.setPower(pLb / max);
        rb.setPower(pRb / max);
    }

    private void setIntake(double power) {
        if (intake != null) intake.setPower(power);
    }
}