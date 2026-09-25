package org.firstinspires.ftc.teamcode.tests;

import com.pedropathing.follower.Follower;
import com.pedropathing.geometry.Pose;
import com.qualcomm.hardware.limelightvision.Limelight3A;
import com.qualcomm.robotcore.hardware.DcMotorSimple;
import org.firstinspires.ftc.teamcode.BallChaseController;
import org.firstinspires.ftc.teamcode.BallChaseFollower;
import org.firstinspires.ftc.teamcode.BallHunt;
import org.firstinspires.ftc.teamcode.BallMath;
import org.firstinspires.ftc.teamcode.BallTracker;
import org.firstinspires.ftc.teamcode.wrapper.BallColor;
import org.firstinspires.ftc.teamcode.wrapper.BallWrangler;
import org.firstinspires.ftc.teamcode.wrapper.MecanumWrangler;
import org.firstinspires.ftc.teamcode.wrapper.PedroWrangler;
import org.firstinspires.ftc.teamcode.wrapper.RobotPose;
import org.firstinspires.ftc.teamcode.wrapper.Target;

import java.util.ArrayList;
import java.util.List;

public class AllTests {

    static final class Drv extends MecanumWrangler {
        Drv(BallTracker t, MockMotor lf, MockMotor rf, MockMotor lb, MockMotor rb) {
            super(t, lf, rf, lb, rb, null);
        }
        Drv(BallTracker t, MockMotor lf, MockMotor rf, MockMotor lb, MockMotor rb, PoseRouter router) {
            super(t, lf, rf, lb, rb, null, router);
        }
        Drv(BallTracker t, MockMotor lf, MockMotor rf, MockMotor lb, MockMotor rb, HeadingSource h) {
            super(t, lf, rf, lb, rb, null, null, h, null);
        }
        void driveNow(double f, double s, double t) { driveRobot(f, s, t); }
        double headingNow() { return headingRad(); }
    }

    static final class PedDrv extends PedroWrangler {
        PedDrv(Follower f, BallTracker t) { super(f, t, null); }
        RobotPose poseNow() { return robotPose(); }
        double headNow() { return headingRad(); }
        void driveNow(double f, double s, double t) { driveRobot(f, s, t); }
        boolean gotoNow(RobotPose p) { return goToPose(p); }
        void stopNow() { stopPoseMove(); }
    }

    static final class CapRouter implements MecanumWrangler.PoseRouter {
        final List<RobotPose> targets = new ArrayList<>();
        double lastPower = -1;
        int stopCalls = 0;
        @Override public RobotPose getPose() { return new RobotPose(10, 10, 0); }
        @Override public void startGoTo(RobotPose target, double speedPower) {
            targets.add(target);
            lastPower = speedPower;
        }
        @Override public boolean isBusy() { return false; }
        @Override public void stop() { stopCalls++; }
    }

    private static int failures = 0;

    private static void check(boolean ok, String what) {
        System.out.println((ok ? "  [ok] " : "  [FAIL] ") + what);
        if (!ok) failures++;
    }

    private static boolean close(double a, double b) { return Math.abs(a - b) < 1e-9; }
    private static boolean near(double a, double b, double tol) { return Math.abs(a - b) <= tol; }

    private static void sleepMs(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    private static List<BallTracker.RawDet> dets(BallTracker.RawDet... d) {
        List<BallTracker.RawDet> l = new ArrayList<>();
        for (BallTracker.RawDet x : d) l.add(x);
        return l;
    }

    private static BallTracker.RawDet red(double tx, double ty, double conf) {
        return new BallTracker.RawDet(BallTracker.CLASS_RED, tx, ty, conf, 0.3);
    }

    private static void pumpController(BallChaseController c, long deadlineMs) {
        long t0 = System.currentTimeMillis();
        while (!c.isDone() && System.currentTimeMillis() - t0 < deadlineMs) {
            c.update();
            sleepMs(2);
        }
    }

    private static void pumpFollower(RobotHarness rig, BallChaseFollower h, long deadlineMs) {
        long t0 = System.currentTimeMillis();
        while (!h.isDone() && System.currentTimeMillis() - t0 < deadlineMs) {
            h.update();
            rig.tickFollower(1);
            sleepMs(2);
        }
    }

    private static void pumpWrangler(BallWrangler w, long deadlineMs) {
        w.start();
        long t0 = System.currentTimeMillis();
        while (!w.isDone() && System.currentTimeMillis() - t0 < deadlineMs) {
            w.update();
            sleepMs(2);
        }
    }

    public static void main(String[] args) {
        testPureValues();
        testTracker();
        testTrackerLockOn();
        testControllerConfigureAndAbort();
        testControllerSearchSweep();
        testControllerChaseAndPickup();
        testControllerPredictedCoast();
        testFollowerSearchExhausts();
        testFollowerChaseAndPickup();
        testFollowerTravelTimeout();
        testFollowerAbortAndBudget();
        testFinalBallHunt();
        testMecanumMixdown();
        testMecanumRouter();
        testMecanumDeadReckonAndHeading();
        testPedroWrangler();
        testWrapperBallHunt();

        if (failures > 0) {
            System.out.println("ALL TESTS FAILED: " + failures + " assertion(s)");
            System.exit(1);
        }
        System.out.println("ALL TESTS OK: pure values, tracker, controller, follower, final BallHunt, "
                + "MecanumWrangler, PedroWrangler, wrapper BallHunt");
    }

    private static void testPureValues() {
        System.out.println("pure values: BallMath / groundRange / Target / RobotPose / BallColor");
        double[] v = BallMath.camRay(0, -20, 25);
        check(near(Math.hypot(v[0], Math.hypot(v[1], v[2])), 1.0, 1e-9), "camRay(0,-20,25) is a unit vector");
        double[] v2 = BallMath.camRay(30, 10, 25);
        check(near(Math.hypot(v2[0], Math.hypot(v2[1], v2[2])), 1.0, 1e-9), "camRay off-axis is a unit vector");
        double[] over = BallMath.ballFieldPos(0, 0, 0, 0, 60, 25, 12, 3);
        check(over == null, "ballFieldPos -> null above the horizon (ty=+60)");
        double[] under = BallMath.ballFieldPos(0, 0, 0, 0, -20, 25, 12, 3);
        check(under != null && under[2] > 0, "ballFieldPos works below the horizon");
        check(BallMath.rangeFromTa(0.0, 100.0) == null, "rangeFromTa -> null on zero angular size");
        check(close(BallMath.rangeFromTa(0.25, 100.0), 20.0), "rangeFromTa(0.25, 100) == 20");
        double[] pk = BallMath.pickupPose(10, 10, 45, 5);
        check(close(Math.hypot(pk[0] - 10, pk[1] - 10), 5.0) && close(pk[2], 45.0),
                "pickupPose sits intakeReach behind the ball at the same heading");
        BallMath.ballFieldPos(10, -20, 40, 95, 60, 25, 12, 3);
        BallMath.ballFieldPos(0, 0, -90, 40, -10, 25, 12, 3);

        BallTracker t = new BallTracker((Limelight3A) null);
        check(close(t.groundRange(0), 6.2 / Math.tan(Math.toRadians(25))), "groundRange(0) == 13.3 in at pitch 25");
        check(t.groundRange(25) == 999.0, "groundRange at/above horizon == 999");
        check(t.groundRange(24) > 100, "groundRange just under the horizon is huge");

        Target tr = new Target(BallColor.RED, 3, -5, 12, 0.9, 1.0);
        check(!tr.hasField(), "fresh Target has no field position");
        Target tf = tr.withField(20, 30);
        check(tf.hasField() && tf.fieldX == 20 && tf.fieldY == 30, "withField projects a field position");
        check(tr.bearingDeg == 3 && tr.elevDeg == -5 && tr.distIn == 12 && tr.confidence == 0.9 && tr.score == 1.0,
                "Target public fields survive construction");
        check(tr.toString().contains("RED"), "Target toString carries the color");

        RobotPose p0 = new RobotPose(1, 2, 0.0);
        check(close(p0.inDeg(90).headingRad, Math.PI / 2), "RobotPose.inDeg(90) == pi/2 rad");
        check(new RobotPose(3, 4, 1.0).toString().contains("3.0"), "RobotPose.toString sanity");

        check(BallColor.fromClassId(BallTracker.CLASS_RED) == BallColor.RED, "fromClassId(RED) -> RED");
        check(BallColor.fromClassId(BallTracker.CLASS_BLUE) == BallColor.BLUE, "fromClassId(BLUE) -> BLUE");
        check(BallColor.fromClassId(BallTracker.CLASS_YELLOW_NEUTRAL) == BallColor.YELLOW, "fromClassId(YELLOW) -> YELLOW");
        check(BallColor.fromClassId(-1) == null, "fromClassId(-1) -> null");
    }

    private static void testTracker() {
        System.out.println("tracker: allowed classes + gating + selection");
        RobotHarness rig = new RobotHarness();
        BallTracker t = rig.newTracker();
        check(t.isAllowed(BallTracker.CLASS_RED) && t.isAllowed(BallTracker.CLASS_BLUE)
                && !t.isAllowed(BallTracker.CLASS_YELLOW_NEUTRAL), "default allowed = red + blue");
        t.setAllowedClasses(BallTracker.CLASS_YELLOW_NEUTRAL);
        check(!t.isAllowed(BallTracker.CLASS_RED) && t.isAllowed(BallTracker.CLASS_YELLOW_NEUTRAL),
                "setAllowedClasses(varargs) replaces the set");
        t.addAllowedClass(BallTracker.CLASS_RED);
        check(t.getAllowedClasses().contains(BallTracker.CLASS_RED)
                && t.getAllowedClasses().contains(BallTracker.CLASS_YELLOW_NEUTRAL)
                && !t.getAllowedClasses().contains(BallTracker.CLASS_BLUE),
                "addAllowedClass grows the set");
        check(t.update() == null, "no detections yet -> null");
        rig.source.dets = dets(red(0, -5, 0.3));
        check(t.update() == null, "sub-MIN_CONF detection is ignored");
        rig.source.dets = dets(red(0, -10, 0.9), red(-2, -5, 0.8));
        BallTracker.Sighting s = t.update();
        check(s != null && !s.predicted && close(s.distIn, t.groundRange(-10)),
                "nearest ball wins (lowest ty, dist " + (s == null ? "?" : s.distIn) + ")");
        rig.source.dets = dets(red(1, -6, 0.9),
                new BallTracker.RawDet(BallTracker.CLASS_BLUE, 5, -4, 0.9, 0.3));
        List<BallTracker.RawDet> conf = t.getConfidentResults();
        List<BallTracker.RawDet> gated = t.getGatedDetections();
        check(conf != null && conf.size() == 2, "getConfidentResults ignores the class filter");
        check(gated != null && gated.size() == 1 && gated.get(0).classId == BallTracker.CLASS_RED,
                "getGatedDetections honors the allowed-class filter");
        rig.source.staleness = BallTracker.MAX_STALENESS_MS + 1;
        check(t.getConfidentResults() == null && t.getGatedDetections() == null,
                "both scan helpers go null on a stale frame");
        BallTracker.Sighting stale = t.update();
        check(stale != null && stale.predicted, "stale frame with a fresh lock coasts on predicted angles");
        t.reset();
        check(t.update() == null, "reset() -> stale frame yields nothing");
        rig.source.staleness = 0;
        check(t.getStalenessMs() == 0, "getStalenessMs passes through the source");
    }

    private static void testTrackerLockOn() {
        System.out.println("tracker: lockOn holds a color until the lock expires");
        RobotHarness rig = new RobotHarness();
        BallTracker t = rig.newTracker();
        t.lockOn(BallTracker.CLASS_RED, 2, -4, 0.9);
        BallTracker.Sighting s1 = t.update();
        check(s1 != null && s1.classId == BallTracker.CLASS_RED && s1.predicted
                && close(s1.distIn, t.groundRange(-4)),
                "lockOn immediately coasts the locked red on predicted angles");
        rig.source.dets = dets(new BallTracker.RawDet(BallTracker.CLASS_BLUE, 0, -6, 0.95, 0.3));
        BallTracker.Sighting s2 = t.update();
        check(s2 != null && s2.classId == BallTracker.CLASS_RED && s2.predicted,
                "a blue ball does not steal the locked red");
        sleepMs(320);
        check(t.update() == null, "lock dropped after LOCK_LOST_MS");
    }

    private static void testControllerConfigureAndAbort() {
        System.out.println("controller: configure directions + abort");
        RobotHarness rig = new RobotHarness();
        MockMotor lf = new MockMotor(), rf = new MockMotor(), lb = new MockMotor(), rb = new MockMotor();
        MockMotor intake = new MockMotor();
        BallChaseController.configureMecanumDirections(lf, rf, lb, rb);
        check(lf.direction == DcMotorSimple.Direction.FORWARD && lb.direction == DcMotorSimple.Direction.FORWARD
                && rf.direction == DcMotorSimple.Direction.REVERSE && rb.direction == DcMotorSimple.Direction.REVERSE,
                "configureMecanumDirections: LF/LB forward, RF/RB reverse");
        BallChaseController c = new BallChaseController(rig.newTracker(), lf, rf, lb, rb, intake);
        c.setMaxPickups(0);
        check(c.getState() == BallChaseController.State.IDLE && !c.isActive() && !c.isDone(),
                "fresh controller is IDLE");
        c.start();
        check(c.getState() == BallChaseController.State.SEARCHING && c.isActive(), "start() -> SEARCHING");
        c.update();
        c.abort();
        check(c.getState() == BallChaseController.State.IDLE && !c.isActive() && !c.isDone(), "abort() -> IDLE");
        check(lf.power == 0 && rf.power == 0 && lb.power == 0 && rb.power == 0 && intake.power == 0,
                "abort zeroes motors and intake");
    }

    private static void testControllerSearchSweep() {
        System.out.println("controller: empty field ends after the search sweep");
        double oldSweep = BallChaseController.SEARCH_SWEEP_DEG;
        BallChaseController.SEARCH_SWEEP_DEG = 10;
        try {
            RobotHarness rig = new RobotHarness();
            MockMotor lf = new MockMotor(), rf = new MockMotor(), lb = new MockMotor(), rb = new MockMotor();
            BallChaseController c = new BallChaseController(rig.newTracker(), lf, rf, lb, rb, null);
            c.setMaxPickups(1);
            c.start();
            c.update();
            check(close(lf.power, BallChaseController.SEARCH_TURN)
                    && close(rf.power, -BallChaseController.SEARCH_TURN)
                    && close(lb.power, BallChaseController.SEARCH_TURN)
                    && close(rb.power, -BallChaseController.SEARCH_TURN),
                    "search turn is a clockwise spin (LF/LB +, RF/RB -)");
            pumpController(c, 3000);
            check(c.isDone(), "empty field hits the dead-reckoned sweep cap -> DONE");
            check(c.getPickups() == 0, "no pickups on an empty field");
            check(lf.power == 0 && rf.power == 0 && lb.power == 0 && rb.power == 0, "motors zeroed at DONE");
        } finally {
            BallChaseController.SEARCH_SWEEP_DEG = oldSweep;
        }
    }

    private static void testControllerChaseAndPickup() {
        System.out.println("controller: chase -> pickup -> done");
        long oldDwell = BallChaseController.PICKUP_DWELL_MS;
        BallChaseController.PICKUP_DWELL_MS = 20;
        try {
            RobotHarness rig = new RobotHarness();
            MockMotor lf = new MockMotor(), rf = new MockMotor(), lb = new MockMotor(), rb = new MockMotor();
            MockMotor intake = new MockMotor();
            BallChaseController c = new BallChaseController(rig.newTracker(), lf, rf, lb, rb, intake);
            c.setMaxPickups(1);
            c.start();
            rig.source.dets = dets(red(3, 5, 0.9));
            c.update();
            check(c.getState() == BallChaseController.State.CHASING, "far on-axis ball -> CHASING");
            check(intake.power == BallChaseController.INTAKE_POWER, "intake on while chasing");
            check(close(lf.power, 0.225) && close(rf.power, 0.075)
                    && close(lb.power, 0.225) && close(rb.power, 0.075),
                    String.format("chase mixes fwd 0.15 + turn 0.075 into the wheels (got lf=%.6f rf=%.6f lb=%.6f rb=%.6f)",
                            lf.power, rf.power, lb.power, rb.power));
            rig.source.dets = dets(red(1, -3, 0.9));
            c.update();
            check(c.getState() == BallChaseController.State.PICKUP, "close + aimed -> PICKUP");
            check(lf.power == 0 && rf.power == 0 && lb.power == 0 && rb.power == 0, "stopped at pickup");
            pumpController(c, 1500);
            check(c.isDone() && c.getPickups() == 1, "dwell -> 1 pickup -> DONE (maxPickups=1)");
            check(intake.power == 0, "intake off once done");
        } finally {
            BallChaseController.PICKUP_DWELL_MS = oldDwell;
        }
    }

    private static void testControllerPredictedCoast() {
        System.out.println("controller: fresh-lock miss coasts, then picks up");
        long oldCoast = BallChaseController.COAST_MS;
        long oldDwell = BallChaseController.PICKUP_DWELL_MS;
        long oldLost = BallTracker.LOCK_LOST_MS;
        BallChaseController.COAST_MS = 40;
        BallChaseController.PICKUP_DWELL_MS = 20;
        BallTracker.LOCK_LOST_MS = 1000;
        try {
            RobotHarness rig = new RobotHarness();
            MockMotor lf = new MockMotor(), rf = new MockMotor(), lb = new MockMotor(), rb = new MockMotor();
            MockMotor intake = new MockMotor();
            BallChaseController c = new BallChaseController(rig.newTracker(), lf, rf, lb, rb, intake);
            c.setMaxPickups(1);
            c.start();
            rig.source.dets = dets(red(1, 5, 0.9));
            c.update();
            check(c.getState() == BallChaseController.State.CHASING && intake.power == BallChaseController.INTAKE_POWER,
                    "locked onto a close-ish ball");
            rig.source.dets = dets();
            c.update();
            check(c.getState() == BallChaseController.State.COASTING, "fresh-lock miss -> COASTING");
            check(close(lf.power, BallChaseController.COAST_POWER) && close(rf.power, BallChaseController.COAST_POWER)
                    && close(lb.power, BallChaseController.COAST_POWER) && close(rb.power, BallChaseController.COAST_POWER),
                    "coast drives straight at COAST_POWER");
            check(intake.power == BallChaseController.INTAKE_POWER, "intake stays on while coasting");
            pumpController(c, 1500);
            check(c.isDone() && c.getPickups() == 1, "coast -> pickup -> DONE with 1 pickup");
        } finally {
            BallChaseController.COAST_MS = oldCoast;
            BallChaseController.PICKUP_DWELL_MS = oldDwell;
            BallTracker.LOCK_LOST_MS = oldLost;
        }
    }

    private static void testFollowerSearchExhausts() {
        System.out.println("follower: empty field exhausts SEARCH_MAX_STEPS");
        long oS = BallChaseFollower.SETTLE_MS, oC = BallChaseFollower.SCAN_MS, oT = BallChaseFollower.TURN_TIMEOUT_MS;
        int oM = BallChaseFollower.SEARCH_MAX_STEPS;
        BallChaseFollower.SETTLE_MS = 5;
        BallChaseFollower.SCAN_MS = 5;
        BallChaseFollower.TURN_TIMEOUT_MS = 20;
        BallChaseFollower.SEARCH_MAX_STEPS = 2;
        try {
            RobotHarness rig = new RobotHarness();
            BallChaseFollower h = new BallChaseFollower(rig.follower, rig.newTracker(), null);
            h.setAllowedClasses(BallTracker.CLASS_RED, BallTracker.CLASS_BLUE);
            h.setMaxPickups(1);
            h.start();
            check(h.getState() == BallChaseFollower.State.SCAN && h.isActive(), "start() -> SCAN");
            boolean sawScan = false, sawTurn = false;
            pumpFollower(rig, h, 3000);
            check(h.isDone(), "finished after SEARCH_MAX_STEPS empty scans");
            check(h.getPickups() == 0, "no pickups on an empty field");
            check(rig.drivetrain.runDriveCalls > 0, "search turns flow through Pedro to the drivetrain");
            check(rig.drivetrain.breakCalls >= 1, "finish() broke following");
        } finally {
            BallChaseFollower.SETTLE_MS = oS;
            BallChaseFollower.SCAN_MS = oC;
            BallChaseFollower.TURN_TIMEOUT_MS = oT;
            BallChaseFollower.SEARCH_MAX_STEPS = oM;
        }
    }

    private static void testFollowerChaseAndPickup() {
        System.out.println("follower: scan -> turn -> chase -> pickup -> done");
        long oS = BallChaseFollower.SETTLE_MS, oC = BallChaseFollower.SCAN_MS, oT = BallChaseFollower.TURN_TIMEOUT_MS;
        long oD = BallChaseFollower.PICKUP_DWELL_MS;
        BallChaseFollower.SETTLE_MS = 5;
        BallChaseFollower.SCAN_MS = 5;
        BallChaseFollower.TURN_TIMEOUT_MS = 20;
        BallChaseFollower.PICKUP_DWELL_MS = 20;
        try {
            RobotHarness rig = new RobotHarness();
            BallChaseFollower h = new BallChaseFollower(rig.follower, rig.newTracker(), null);
            h.setAllowedClasses(BallTracker.CLASS_RED);
            h.setMaxPickups(1);
            rig.source.dets = dets(red(0, -15, 0.9));
            h.start();
            pumpFollower(rig, h, 3000);
            check(h.isDone() && h.getPickups() == 1, "near in-view ball -> PICKUP -> DONE with 1");
            check(rig.drivetrain.breakCalls >= 1, "done path broke following");
        } finally {
            BallChaseFollower.SETTLE_MS = oS;
            BallChaseFollower.SCAN_MS = oC;
            BallChaseFollower.TURN_TIMEOUT_MS = oT;
            BallChaseFollower.PICKUP_DWELL_MS = oD;
        }
    }

    private static void testFollowerTravelTimeout() {
        System.out.println("follower: far ball plans a Pedro path, times out, re-plans, ends");
        long oS = BallChaseFollower.SETTLE_MS, oC = BallChaseFollower.SCAN_MS, oT = BallChaseFollower.TURN_TIMEOUT_MS;
        long oTv = BallChaseFollower.TRAVEL_TIMEOUT_MS;
        int oM = BallChaseFollower.SEARCH_MAX_STEPS;
        BallChaseFollower.SETTLE_MS = 5;
        BallChaseFollower.SCAN_MS = 5;
        BallChaseFollower.TURN_TIMEOUT_MS = 20;
        BallChaseFollower.TRAVEL_TIMEOUT_MS = 30;
        BallChaseFollower.SEARCH_MAX_STEPS = 1;
        try {
            RobotHarness rig = new RobotHarness();
            BallChaseFollower h = new BallChaseFollower(rig.follower, rig.newTracker(), null);
            h.setAllowedClasses(BallTracker.CLASS_RED);
            h.setMaxPickups(1);
            rig.source.dets = dets(red(0, 15, 0.9));
            h.start();
            boolean sawTravel = false;
            long t0 = System.currentTimeMillis();
            while (!h.isDone() && System.currentTimeMillis() - t0 < 3000) {
                h.update();
                rig.tickFollower(1);
                if (h.getState() == BallChaseFollower.State.TRAVEL) sawTravel = true;
                sleepMs(2);
            }
            check(sawTravel, "far ball planned a path -> TRAVEL");
            check(h.isDone(), "TRAVEL timeout -> re-scan throws in-view ball -> SEARCH_MAX_STEPS -> DONE");
            check(h.getPickups() == 0, "no pickup made");
            check(rig.drivetrain.breakCalls >= 1, "travel timeout broke following");
        } finally {
            BallChaseFollower.SETTLE_MS = oS;
            BallChaseFollower.SCAN_MS = oC;
            BallChaseFollower.TURN_TIMEOUT_MS = oT;
            BallChaseFollower.TRAVEL_TIMEOUT_MS = oTv;
            BallChaseFollower.SEARCH_MAX_STEPS = oM;
        }
    }

    private static void testFollowerAbortAndBudget() {
        System.out.println("follower: abort + time budget");
        RobotHarness rig = new RobotHarness();
        BallChaseFollower h = new BallChaseFollower(rig.follower, rig.newTracker(), null);
        h.setMaxPickups(1);
        h.start();
        h.abort();
        check(h.getState() == BallChaseFollower.State.IDLE && !h.isActive(), "abort() -> IDLE");
        check(rig.drivetrain.breakCalls >= 1, "abort broke following");
        check(rig.drivetrain.teleopCalls >= 1, "abort put the follower back in teleop drive");

        RobotHarness rig2 = new RobotHarness();
        BallChaseFollower h2 = new BallChaseFollower(rig2.follower, rig2.newTracker(), null);
        h2.setMaxPickups(1);
        h2.setTimeBudgetSec(0.02);
        h2.start();
        long t0 = System.currentTimeMillis();
        pumpFollower(rig2, h2, 2000);
        long dt = System.currentTimeMillis() - t0;
        check(h2.isDone(), "time budget finished the hunt");
        check(h2.getPickups() == 0, "no pickups");
        check(dt < 1500, "budget enforced quickly (" + dt + " ms)");
    }

    private static void testFinalBallHunt() {
        System.out.println("BallHunt (final): fluent config + budgeted run");
        RobotHarness rig = new RobotHarness();
        BallHunt hunt = new BallHunt(rig.follower, rig.newTracker(), null);
        hunt.reds().collect(1).within(0.05);
        hunt.start();
        long t0 = System.currentTimeMillis();
        while (!hunt.isDone() && System.currentTimeMillis() - t0 < 2000) {
            hunt.update();
            rig.tickFollower(1);
            sleepMs(2);
        }
        check(hunt.isDone(), "budgeted final BallHunt finished");
        check(hunt.got() == 0, "nothing picked on an empty field");
        check(hunt.state().equals("DONE"), "state() reports DONE");

        BallHunt h2 = new BallHunt(rig.follower, rig.newTracker(), null);
        h2.alliance().everything().reds().blues().yellows().collect(0).within(0.03);
        h2.start();
        t0 = System.currentTimeMillis();
        while (!h2.isDone() && System.currentTimeMillis() - t0 < 2000) {
            h2.update();
            rig.tickFollower(1);
            sleepMs(2);
        }
        check(h2.isDone() && h2.got() == 0, "color-verb chain also finishes cleanly");
    }

    private static void testMecanumMixdown() {
        System.out.println("MecanumWrangler: wheel mixdown + normalization");
        RobotHarness rig = new RobotHarness();
        MockMotor lf = new MockMotor(), rf = new MockMotor(), lb = new MockMotor(), rb = new MockMotor();
        Drv m = new Drv(rig.newTracker(), lf, rf, lb, rb);
        m.driveNow(1, 0, 0);
        check(close(lf.power, 1) && close(rf.power, 1) && close(lb.power, 1) && close(rb.power, 1),
                "fwd=1 drives all wheels at full power");
        m.driveNow(0.7, 0.7, 0.7);
        check(close(lf.power, 1.0) && close(rf.power, -1.0 / 3)
                && close(lb.power, 1.0 / 3) && close(rb.power, 1.0 / 3),
                "over-unit mix normalizes (LF full, others 1/3)");
        m.driveNow(0, 0, -0.4);
        check(close(lf.power, -0.4) && close(rf.power, 0.4)
                && close(lb.power, -0.4) && close(rb.power, 0.4),
                "CCW turn power -0.4 keeps turn + = clockwise");
    }

    private static void testMecanumRouter() {
        System.out.println("MecanumWrangler: pose verbs route through the PoseRouter");
        RobotHarness rig = new RobotHarness();
        MockMotor lf = new MockMotor(), rf = new MockMotor(), lb = new MockMotor(), rb = new MockMotor();
        CapRouter r = new CapRouter();
        MecanumWrangler m = new MecanumWrangler(rig.newTracker(), lf, rf, lb, rb, null, r);
        m.then(new RobotPose(24, 24, Math.toRadians(90)));
        pumpWrangler(m, 1500);
        check(m.isDone(), "pose verb completed via the router");
        check(r.targets.size() == 1, "router received exactly one go-to");
        check(close(r.lastPower, MecanumWrangler.GO_TO_POWER), "go-to used GO_TO_POWER");
        check(r.stopCalls >= 1, "finish explicitly stops the router's go-to");
    }

    private static void testMecanumDeadReckonAndHeading() {
        System.out.println("MecanumWrangler: dead-reckon heading + HeadingSource");
        double oldRate = MecanumWrangler.TURN_RATE_RAD_PER_POWER_SEC;
        MecanumWrangler.TURN_RATE_RAD_PER_POWER_SEC = 1.0;
        try {
            RobotHarness rig = new RobotHarness();
            MockMotor lf = new MockMotor(), rf = new MockMotor(), lb = new MockMotor(), rb = new MockMotor();
            Drv m = new Drv(rig.newTracker(), lf, rf, lb, rb);
            check(close(m.headingNow(), 0), "dead-reckon heading starts at 0");
            m.driveNow(0, 0, 0.5);
            sleepMs(60);
            m.driveNow(0, 0, 0);
            check(near(m.headingNow(), -0.5 * 0.06, 0.02),
                    "CW turn integrates into a CCW dead-reckon heading (got " + m.headingNow() + ")");

            RobotHarness rig2 = new RobotHarness();
            MockMotor lf2 = new MockMotor(), rf2 = new MockMotor(), lb2 = new MockMotor(), rb2 = new MockMotor();
            Drv m2 = new Drv(rig2.newTracker(), lf2, rf2, lb2, rb2, (MecanumWrangler.HeadingSource) () -> 1.2);
            check(close(m2.headingNow(), 1.2), "HeadingSource supplies the heading reading");
        } finally {
            MecanumWrangler.TURN_RATE_RAD_PER_POWER_SEC = oldRate;
        }
    }

    private static void testPedroWrangler() {
        System.out.println("PedroWrangler: pose, heading, drive, pathing through a real follower");
        RobotHarness rig = new RobotHarness();
        rig.localizer.pose = new Pose(12, 34, 0.5);
        PedDrv m = new PedDrv(rig.follower, rig.newTracker());
        m.start();
        check(rig.drivetrain.teleopCalls >= 1, "start() put the follower in teleop drive");
        RobotPose p = m.poseNow();
        check(p.x == 12 && p.y == 34 && close(p.headingRad, 0.5), "robotPose maps the follower pose");
        check(close(m.headNow(), 0.5), "headingRad comes from the follower pose");
        m.driveNow(0.5, 0.2, 0.1);
        rig.tickFollower(3);
        check(rig.drivetrain.runDriveCalls > 0, "camera-relative drive flows through Pedro to the drivetrain");
        check(m.gotoNow(new RobotPose(24, 0, 0)), "goToPose accepted a target");
        check(rig.follower.isBusy(), "follower reports pathing after followPath");
        m.stopNow();
        check(rig.drivetrain.breakCalls >= 1 && rig.drivetrain.teleopCalls >= 2,
                "stopPoseMove breaks following and returns to teleop");
    }

    private static void testWrapperBallHunt() {
        System.out.println("wrapper BallHunt: fluent chain + budgeted runs");
        RobotHarness rig = new RobotHarness();
        org.firstinspires.ftc.teamcode.wrapper.BallHunt w =
                new org.firstinspires.ftc.teamcode.wrapper.BallHunt(rig.follower, rig.newTracker(), null);
        check(w.state().equals("idle"), "wrapper BallHunt is idle before start");
        w.reds().collect(2).within(0.05);
        w.start();
        long t0 = System.currentTimeMillis();
        while (!w.isDone() && System.currentTimeMillis() - t0 < 2000) {
            w.update();
            rig.tickFollower(1);
            sleepMs(2);
        }
        check(w.isDone() && w.state().equals("done"), "budgeted wrapper hunt finished");
        w.blues().yellows().alliance().everything().clear().within(0.01).collect(1);
        w.start();
        t0 = System.currentTimeMillis();
        while (!w.isDone() && System.currentTimeMillis() - t0 < 2000) {
            w.update();
            rig.tickFollower(1);
            sleepMs(2);
        }
        check(w.isDone(), "reused wrapper hunt finished a second run");
    }
}