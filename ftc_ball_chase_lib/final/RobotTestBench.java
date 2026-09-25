/*
 * RobotTestBench - ONE clinical OpMode to sanity-check every drive/detector
 * driver on the real robot before trusting it in matches. Pick the test with
 * the LEFT BUMPER, hold A to run it (state machines), release to abort.
 * BACK kills everything and returns to manual drive.
 *
 *   LEFT BUMPER  cycle the selected test (name shown in telemetry)
 *   A (hold)     run the selected test; release = abort to manual
 *   BACK         hard abort + zero motors/intake
 *
 *   WHEELS   manual drive: D-pad fwd/back/strafe, right stick X = spin.
 *            Verifies the mecanum mixdown + directions are wired right.
 *   SPIN     A/B/X/Y = spin LF/RF/LB/RB alone at 0.4 to check each motor.
 *   DETECT   live Limelight readout. D-pad = class filter, A = lockOn the
 *            nearest visible ball, B = drop the lock.
 *   CHASE    BallChaseController full SEARCHING..DONE chase (intake on).
 *   INTAKE   X on, Y off, B reverse.
 *   MECANUM  MecanumWrangler wrapper verbs (scan + grab nearest, 1 ball).
 *   HUNT     BallChaseFollower state machine. Needs buildFollower() below
 *            wired with your Pedro follower (constants + localizer +
 *            drivetrain), otherwise it reports "no follower".
 */
package org.firstinspires.ftc.teamcode;

import com.pedropathing.follower.Follower;
import com.qualcomm.hardware.limelightvision.Limelight3A;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.HardwareMap;
import org.firstinspires.ftc.teamcode.wrapper.MecanumWrangler;

import java.util.List;

@TeleOp(name = "All-Code Test Bench", group = "Hive Vision")
public class RobotTestBench extends LinearOpMode {

    private static final int PIPELINE_INDEX = 0;
    private static final double WHEEL_TEST_POWER = 0.4;

    private enum Test { WHEELS, SPIN, DETECT, CHASE, INTAKE, MECANUM, HUNT }

    private DcMotor lf, rf, lb, rb, intake;
    private Limelight3A limelight;
    private BallTracker tracker;
    private BallChaseController chase;
    private BallChaseFollower hunt;
    private MecanumWrangler mec;
    private Follower follower;
    private String followerErr = null;

    private Test selected = Test.WHEELS;
    private boolean armed = false;

    /** Wire your real Pedro follower here (constants + localizer + drivetrain).
     *  Leave null to skip the HUNT test. */
    protected Follower buildFollower(HardwareMap hw) { return null; }

    @Override
    public void runOpMode() {
        lf = hardwareMap.get(DcMotor.class, "leftFront");
        rf = hardwareMap.get(DcMotor.class, "rightFront");
        lb = hardwareMap.get(DcMotor.class, "leftBack");
        rb = hardwareMap.get(DcMotor.class, "rightBack");
        intake = hardwareMap.tryGet(DcMotor.class, "intake");

        BallChaseController.configureMecanumDirections(lf, rf, lb, rb);
        for (DcMotor m : new DcMotor[]{lf, rf, lb, rb}) {
            m.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
            m.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
        }

        limelight = hardwareMap.get(Limelight3A.class, "limelight");
        limelight.pipelineSwitch(PIPELINE_INDEX);
        limelight.setPollRateHz(100);
        limelight.start();
        tracker = new BallTracker(limelight);

        while (!isStarted() && !isStopRequested()) {
            if (gamepad1.dpad_up)    tracker.setAllowedClasses(BallTracker.CLASS_RED);
            if (gamepad1.dpad_down)  tracker.setAllowedClasses(BallTracker.CLASS_BLUE);
            if (gamepad1.dpad_left)  tracker.setAllowedClasses(BallTracker.CLASSES_RED_BLUE);
            if (gamepad1.dpad_right) tracker.setAllowedClasses(BallTracker.CLASSES_ALL);
            telemetry.addLine("Init: D-pad pick classes (UP red / DOWN blue / LEFT R+B / RIGHT all).");
            telemetry.addLine("Hit START when ready.");
            telemetry.update();
        }

        waitForStart();

        boolean lastLb = false, lastBack = false;
        while (opModeIsActive()) {
            if (gamepad1.back && !lastBack) hardAbort();
            lastBack = gamepad1.back;

            if (gamepad1.left_bumper && !lastLb) {
                abortRun();
                zeroAndStop();
                selected = next(selected);
                armed = false;
            }
            lastLb = gamepad1.left_bumper;

            switch (selected) {
                case WHEELS:   driveManual();    break;
                case SPIN:     spinControls();   break;
                case DETECT:   detectControls(); break;
                case INTAKE:   intakeControls(); break;
                case CHASE:
                case MECANUM:
                case HUNT:
                    if (gamepad1.a && !armed) startRun();
                    else if (!gamepad1.a && armed) abortRun();
                    armed = gamepad1.a;
                    if (armed) stepRun(); else zeroAndStop();
                    break;
            }

            commonTelemetry();
            telemetry.update();
        }

        hardAbort();
        limelight.stop();
    }

    /* ---------------- test selection / lifecycle ---------------- */

    private static Test next(Test t) {
        Test[] all = Test.values();
        return all[(t.ordinal() + 1) % all.length];
    }

    private void hardAbort() {
        abortRun();
        zeroAndStop();
        selected = Test.WHEELS;
        armed = false;
    }

    private void startRun() {
        abortRun();
        switch (selected) {
            case CHASE:
                if (chase == null) chase = new BallChaseController(tracker, lf, rf, lb, rb, intake);
                chase.setMaxPickups(1);
                chase.start();
                break;
            case MECANUM:
                mec = new MecanumWrangler(tracker, lf, rf, lb, rb, intake);
                mec.alliance().grabUpTo(1).within(2.0);
                mec.start();
                break;
            case HUNT:
                ensureFollower();
                if (follower != null) {
                    if (hunt == null) hunt = new BallChaseFollower(follower, tracker, intake);
                    hunt.setMaxPickups(1);
                    hunt.setTimeBudgetSec(8.0);
                    hunt.start();
                }
                break;
            default: break;
        }
    }

    private void stepRun() {
        if (selected == Test.CHASE) {
            if (chase != null && chase.isActive()) chase.update(); else zeroAndStop();
        } else if (selected == Test.MECANUM) {
            if (mec != null && mec.isActive()) mec.update(); else zeroAndStop();
        } else if (selected == Test.HUNT) {
            if (hunt != null && hunt.isActive()) {
                hunt.update();
                if (follower != null) follower.update();
            } else zeroAndStop();
        }
    }

    private void abortRun() {
        if (chase != null && chase.isActive()) chase.abort();
        if (mec != null && mec.isActive()) mec.abort();
        if (hunt != null && hunt.isActive()) hunt.abort();
    }

    private void ensureFollower() {
        followerErr = null;
        if (follower != null) return;
        try {
            follower = buildFollower(hardwareMap);
        } catch (Throwable t) {
            follower = null;
            followerErr = String.valueOf(t.getMessage());
        }
    }

    private void zeroAndStop() {
        lf.setPower(0); rf.setPower(0); lb.setPower(0); rb.setPower(0);
        if (intake != null) intake.setPower(0);
    }

    /* ---------------- self-service tests ---------------- */

    private void driveManual() {
        double f = -gamepad1.left_stick_y;
        double s = gamepad1.left_stick_x;
        double t = gamepad1.right_stick_x;
        double pLf = f + s + t;
        double pRf = f - s - t;
        double pLb = f - s + t;
        double pRb = f + s - t;
        double max = Math.max(1.0,
                Math.max(Math.max(Math.abs(pLf), Math.abs(pRf)),
                         Math.max(Math.abs(pLb), Math.abs(pRb))));
        lf.setPower(pLf / max);
        rf.setPower(pRf / max);
        lb.setPower(pLb / max);
        rb.setPower(pRb / max);
    }

    private void spinControls() {
        zeroMotors();
        if (gamepad1.a) lf.setPower(WHEEL_TEST_POWER);
        if (gamepad1.b) rf.setPower(WHEEL_TEST_POWER);
        if (gamepad1.x) lb.setPower(WHEEL_TEST_POWER);
        if (gamepad1.y) rb.setPower(WHEEL_TEST_POWER);
    }

    private void detectControls() {
        if (gamepad1.dpad_up)    tracker.setAllowedClasses(BallTracker.CLASS_RED);
        if (gamepad1.dpad_down)  tracker.setAllowedClasses(BallTracker.CLASS_BLUE);
        if (gamepad1.dpad_left)  tracker.setAllowedClasses(BallTracker.CLASSES_RED_BLUE);
        if (gamepad1.dpad_right) tracker.setAllowedClasses(BallTracker.CLASSES_ALL);
        if (gamepad1.a) {
            List<BallTracker.RawDet> dets = tracker.getGatedDetections();
            BallTracker.RawDet best = null;
            if (dets != null) {
                for (BallTracker.RawDet d : dets) {
                    if (best == null || d.tyDeg < best.tyDeg) best = d;
                }
            }
            if (best != null) tracker.lockOn(best.classId, best.txDeg, best.tyDeg, best.confidence);
        }
        if (gamepad1.b) tracker.reset();
    }

    private void intakeControls() {
        zeroMotors();
        if (intake != null) {
            if (gamepad1.x) intake.setPower(1.0);
            if (gamepad1.y) intake.setPower(0.0);
            if (gamepad1.b) intake.setPower(-1.0);
        }
    }

    private void zeroMotors() {
        lf.setPower(0); rf.setPower(0); lb.setPower(0); rb.setPower(0);
    }

    /* ---------------- telemetry ---------------- */

    private void commonTelemetry() {
        telemetry.addData("test", "%s%s | LB=next A=run BACK=abort",
                selected, armed ? " (RUN)" : "");
        telemetry.addLine("hint: " + hint());
        telemetry.addLine("classes: " + tracker.getAllowedClasses());

        BallTracker.Sighting s = tracker.getLast();
        if (s != null) {
            telemetry.addData("target", "cls=%d tx=%+.1f ty=%+.1f dist=%.0f conf=%.2f%s",
                    s.classId, s.txDeg, s.tyDeg, s.distIn, s.confidence, s.predicted ? " (pred)" : "");
        } else {
            telemetry.addLine("target: none");
        }
        telemetry.addData("staleness", "%d ms", tracker.getStalenessMs());

        if (selected == Test.DETECT) {
            List<BallTracker.RawDet> dets = tracker.getGatedDetections();
            if (dets != null && !dets.isEmpty()) {
                int i = 0;
                for (BallTracker.RawDet d : dets) {
                    if (i++ >= 6) break;
                    telemetry.addData("det " + i, "cls=%d tx=%+.1f ty=%+.1f conf=%.2f",
                            d.classId, d.txDeg, d.tyDeg, d.confidence);
                }
            } else {
                telemetry.addLine("detections: (stale or none)");
            }
        }

        if (chase != null) {
            telemetry.addData("chase", "%s pickups=%d/%s", chase.getState(), chase.getPickups(),
                    chase.isDone() ? "done" : "run");
        }
        if (mec != null) {
            telemetry.addData("mec", "done=%s pickups=%d err=%s",
                    mec.isDone(), mec.getPickups(), mec.lastError() == null ? "-" : mec.lastError());
        }
        if (followerErr != null) {
            telemetry.addLine("follower build failed: " + followerErr);
        }
        if (selected == Test.HUNT && follower == null) {
            telemetry.addLine("NO follower: override buildFollower() with your Pedro follower");
        }
        if (hunt != null) {
            telemetry.addData("hunt", "%s pickups=%d", hunt.getState(), hunt.getPickups());
        }
    }

    private String hint() {
        switch (selected) {
            case WHEELS:  return "left stick drive+strafe, right stick X spin";
            case SPIN:    return "A=LF B=RF X=LB Y=RB at 0.4 (release = stop)";
            case DETECT:  return "D-pad class filter; A = lockOn nearest; B = drop lock";
            case CHASE:   return "hold A to chase 1 ball; release = abort";
            case INTAKE:  return "X on, Y off, B reverse";
            case MECANUM: return "hold A: wrapper scan+grab 1 ball; release = abort";
            case HUNT:    return "hold A: follower hunt 1 ball; release = abort";
            default:      return "";
        }
    }
}