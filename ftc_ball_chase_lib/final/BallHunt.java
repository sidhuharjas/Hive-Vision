/*
 * BallHunt - one-liner ball collecting for AUTO, wrapping BallChaseFollower
 * behind a fluent API. Class docs moved to MODULES.md (#ballhunt).
 */
package org.firstinspires.ftc.teamcode;

import com.pedropathing.follower.Follower;
import com.qualcomm.hardware.limelightvision.Limelight3A;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import org.firstinspires.ftc.robotcore.external.Telemetry;

public class BallHunt {

    private final BallChaseFollower hunt;
    private final Follower follower;

    /** Defaults: alliance colors (red+blue), 1 ball, 15 s budget. */
    public BallHunt(Follower follower, Limelight3A limelight, DcMotor intakeOrNull) {
        this(follower, new BallTracker(limelight), intakeOrNull);
    }

    public BallHunt(Follower follower, BallTracker tracker, DcMotor intakeOrNull) {
        this.hunt = new BallChaseFollower(follower, tracker, intakeOrNull);
        this.follower = follower;
        hunt.setAllowedClasses(BallTracker.CLASSES_RED_BLUE);
        hunt.setMaxPickups(1);
        hunt.setTimeBudgetSec(15.0);
    }

    /* ---------------- which balls ---------------- */

    public BallHunt reds() {
        hunt.setAllowedClasses(BallTracker.CLASS_RED);
        return this;
    }

    public BallHunt blues() {
        hunt.setAllowedClasses(BallTracker.CLASS_BLUE);
        return this;
    }

    public BallHunt yellows() {
        hunt.setAllowedClasses(BallTracker.CLASS_YELLOW_NEUTRAL);
        return this;
    }

    public BallHunt alliance() {
        hunt.setAllowedClasses(BallTracker.CLASSES_RED_BLUE);
        return this;
    }

    public BallHunt everything() {
        hunt.setAllowedClasses(BallTracker.CLASSES_ALL);
        return this;
    }

    /* ---------------- how much / how long ---------------- */

    /** Pick up n balls on this hunt (n <= 0 = keep going). */
    public BallHunt collect(int n) {
        hunt.setMaxPickups(n);
        return this;
    }

    /** Hard time budget in seconds before handing back control. */
    public BallHunt within(double seconds) {
        hunt.setTimeBudgetSec(seconds);
        return this;
    }

    /* ---------------- run it ---------------- */

    /** Blocking: run the whole hunt, own follower.update(), stream telemetry,
     *  abort() when done, and return how many balls were picked. */
    public int go(LinearOpMode op) {
        hunt.start();
        while (op.opModeIsActive() && !hunt.isDone()) {
            follower.update();
            hunt.update();
            op.telemetry.addData("hunt", "%s  picked=%d", state(), got());
            hunt.addTelemetry(op.telemetry);
            op.telemetry.update();
        }
        hunt.abort();
        return got();
    }

    /** Loop-driven: call once to arm, then update() each loop, then abort(). */
    public void start() {
        hunt.start();
    }

    public void update() {
        hunt.update();
    }

    public void abort() {
        hunt.abort();
    }

    /* ---------------- status ---------------- */

    public boolean isDone() {
        return hunt.isDone();
    }

    public int got() {
        return hunt.getPickups();
    }

    public String state() {
        return hunt.getState().name();
    }

    public void addTelemetry(Telemetry t) {
        hunt.addTelemetry(t);
    }
}