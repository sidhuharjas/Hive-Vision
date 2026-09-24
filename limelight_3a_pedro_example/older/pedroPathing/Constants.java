/*
 * Constants - Pedro Pathing 2.x follower builder (placeholder project adapters).
 *
 * This is the replacement for the Pedro 1.x quickstart file that used to sit at
 * org.firstinspires.ftc.teamcode.pedroPathing.Constants with a
 * `createFollower(hardwareMap)` helper. The legacy demos in this `older/` folder
 * (BallPickupOpMode, BallHunterTest) still call that helper; it now builds a
 * 2.1.2 Follower through the official FollowerBuilder.
 *
 * CAUTION: every number/name below is a placeholder so the file COMPILES.
 * TUNE ME for the actual robot before trusting it on the field - motor names,
 * IMU/encoder names, pod offsets and CP tuning. The modern BallTracker +
 * BallChaseController/BallChaseFollower in the parent folder do NOT use this
 * file; it exists only to keep the archived examples buildable.
 */
package org.firstinspires.ftc.teamcode.pedroPathing;

import com.pedropathing.follower.Follower;
import com.pedropathing.follower.FollowerConstants;
import com.pedropathing.ftc.FollowerBuilder;
import com.pedropathing.ftc.drivetrains.MecanumConstants;
import com.pedropathing.ftc.localization.constants.ThreeWheelIMUConstants;
import com.qualcomm.hardware.rev.RevHubOrientationOnRobot;
import com.qualcomm.robotcore.hardware.DcMotorSimple;
import com.qualcomm.robotcore.hardware.HardwareMap;

public final class Constants {

    /* ---- TUNE ME: match your robot config ---- */
    public static final String LEFT_FRONT = "leftFront";   // also the left encoder
    public static final String LEFT_REAR  = "leftRear";
    public static final String RIGHT_FRONT = "rightFront"; // also the right encoder
    public static final String RIGHT_REAR  = "rightRear";  // also the strafe encoder
    public static final String IMU_NAME    = "imu";

    private Constants() {
    }

    /** Build a Pedro 2.1.2 Follower from the hardware map (placeholder values). */
    public static Follower createFollower(HardwareMap hardwareMap) {
        FollowerConstants followerConstants = new FollowerConstants();

        MecanumConstants mecanum = new MecanumConstants()
                .xVelocity(25.0)     // TUNE ME: max drivetrain speed (in/s)
                .yVelocity(25.0)
                .maxPower(1.0)
                .leftFrontMotorName(LEFT_FRONT)
                .leftRearMotorName(LEFT_REAR)
                .rightFrontMotorName(RIGHT_FRONT)
                .rightRearMotorName(RIGHT_REAR)
                .leftFrontMotorDirection(DcMotorSimple.Direction.FORWARD)
                .leftRearMotorDirection(DcMotorSimple.Direction.FORWARD)
                .rightFrontMotorDirection(DcMotorSimple.Direction.REVERSE)
                .rightRearMotorDirection(DcMotorSimple.Direction.REVERSE);

        ThreeWheelIMUConstants localizer = new ThreeWheelIMUConstants()
                .forwardTicksToInches(0.05)   // TUNE ME
                .strafeTicksToInches(0.05)
                .turnTicksToInches(0.05)
                .leftPodY(5.0)                // TUNE ME: pod offsets in inches
                .rightPodY(-5.0)
                .strafePodX(2.0)
                .IMU_HardwareMapName(IMU_NAME)
                .leftEncoder_HardwareMapName(LEFT_FRONT)
                .rightEncoder_HardwareMapName(RIGHT_FRONT)
                .strafeEncoder_HardwareMapName(RIGHT_REAR)
                .IMU_Orientation(new RevHubOrientationOnRobot(
                        RevHubOrientationOnRobot.LogoFacingDirection.UP,
                        RevHubOrientationOnRobot.UsbFacingDirection.FORWARD))  // TUNE ME
                .leftEncoderDirection(1.0)
                .rightEncoderDirection(-1.0)
                .strafeEncoderDirection(-1.0);

        return new FollowerBuilder(followerConstants, hardwareMap)
                .mecanumDrivetrain(mecanum)
                .threeWheelIMULocalizer(localizer)
                .build();
    }
}