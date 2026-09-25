package org.firstinspires.ftc.teamcode.tests;

import com.pedropathing.drivetrain.Drivetrain;
import com.pedropathing.follower.Follower;
import com.pedropathing.follower.FollowerConstants;
import com.pedropathing.geometry.Pose;
import com.pedropathing.localization.Localizer;
import com.pedropathing.math.Vector;
import com.pedropathing.paths.PathConstraints;
import org.firstinspires.ftc.teamcode.BallTracker;

import java.util.ArrayList;
import java.util.List;

public final class RobotHarness {

    public static final class ScriptedSource implements BallTracker.DetectionSource {
        public List<BallTracker.RawDet> dets = new ArrayList<>();
        public long staleness = 0;
        @Override public List<BallTracker.RawDet> latest() { return dets; }
        @Override public long stalenessMs() { return staleness; }
    }

    public static final class FloatLocalizer implements Localizer {
        public Pose pose = new Pose(0, 0, 0);
        public int updateCalls = 0;
        @Override public Pose getPose() { return pose; }
        @Override public Pose getVelocity() { return new Pose(0, 0, 0); }
        @Override public Vector getVelocityVector() { return new Vector(0, 0); }
        @Override public void setStartPose(Pose p) { pose = p; }
        @Override public void setPose(Pose p) { pose = p; }
        @Override public void update() { updateCalls++; }
        @Override public double getTotalHeading() { return pose.getHeading(); }
        @Override public double getForwardMultiplier() { return 1; }
        @Override public double getLateralMultiplier() { return 1; }
        @Override public double getTurningMultiplier() { return 1; }
        @Override public void resetIMU() throws InterruptedException { }
        @Override public double getIMUHeading() { return pose.getHeading(); }
        @Override public boolean isNAN() { return false; }
    }

    public static final class FrameDrivetrain extends Drivetrain {
        public int runDriveCalls = 0;
        public int breakCalls = 0;
        public int teleopCalls = 0;
        public String debug = "test";
        @Override public double[] calculateDrive(Vector v1, Vector v2, Vector v3, double d) {
            return new double[]{0, 0, 0, 0};
        }
        @Override public void updateConstants() { }
        @Override public void breakFollowing() { breakCalls++; }
        @Override public void runDrive(double[] p) { runDriveCalls++; }
        @Override public void startTeleopDrive() { teleopCalls++; }
        @Override public void startTeleopDrive(boolean b) { teleopCalls++; }
        @Override public double xVelocity() { return 0; }
        @Override public double yVelocity() { return 0; }
        @Override public void setXVelocity(double v) { }
        @Override public void setYVelocity(double v) { }
        @Override public double getVoltage() { return 12; }
        @Override public String debugString() { return debug; }
    }

    public final ScriptedSource source = new ScriptedSource();
    public final FloatLocalizer localizer = new FloatLocalizer();
    public final FrameDrivetrain drivetrain = new FrameDrivetrain();
    public final Follower follower;

    public RobotHarness() {
        follower = new Follower(new FollowerConstants(), localizer, drivetrain,
                new PathConstraints(12, 30, 0.5, 0.5));
    }

    public BallTracker newTracker() {
        return new BallTracker(source);
    }

    public void tickFollower(int n) {
        for (int i = 0; i < n; i++) follower.update();
    }
}