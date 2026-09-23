/*
 * Hive Vision — Control Hub track #2 (Lab chromaticity, no model)
 * Middle-ground ball detector for FTC: no coprocessor (unlike the YOLO
 * track) but built around CIELAB chromaticity instead of raw HSV, so its
 * color match is far more robust to illumination changes — the classic
 * HSV failure (a shadowed ball reads as a different color) mostly goes
 * away because Lab separates "what color" (a*/b* hue) from "how bright"
 * (L*).
 *
 * How the numbers were learned (see hive-vision/lab/README.md for the
 * full story): the flagship YOLO model (hive-vision/neural-net/weights/best.pt)
 * was run over real match footage and every true ball box was sampled.
 * Inside the boxes the chromaticity hue atan2(b*, a*) concentrates in a
 * tight band per color; the sat floor RESPONDS to the frame's ambient L*:
 *
 *     scl      = clamp(meanL / ref_l, scl_lo, scl_hi)
 *     satFloor = sat_floor * scl        (per-color sat_floor)
 *
 * So in a dim frame the chroma bar is lowered to match the reduced chroma,
 * not raised by accident (which is what HSV S/V thresholds do in shadow).
 * The model is "hue band + adaptive chroma floor + L floor", not a raw RGB
 * threshold. L* is otherwise ignored (only kills near-blacks).
 *
 * Measured vs YOLO truth on the dev match video (the SAME truth for Lab
 * and HSV; full table in lab/docs/lab_detector_report.md):
 *                            recall  y/r/b      precision  y/r/b
 *   Lab (this pipeline)     39.7/40.7/54.5%      6.3/5.6/8.1%
 *   HSV reference           62.0/34.5/52.8%     10.5/8.6/11.1%
 *
 * HARD PHYSICS LIMIT: in deep shadow a ball's chroma collapses toward
 * zero (the sensor stops reporting color). At that point NO color-only
 * detector — HSV or Lab — can see it (here, 1 yellow + 11 blue dark
 * frames are unwinnable for both). If a match hides balls there, you
 * need the YOLO track. Lab's real win: moderate shadows where hue
 * survives; e.g. red dark-frame recall is ~94% here vs ~40% for HSV.
 *
 * Hard-code the constants below (they come from lab/tools/lab_tuned.json),
 * or tune live with lab/tools/lab_tuner.py and copy the new values back.
 */
package org.firstinspires.ftc.teamcode;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import org.firstinspires.ftc.vision.VisionProcessor;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection; // not used, but common import

public class LabBallDetectorPipeline implements VisionProcessor {

    /* ------------------------------------------------------------------
     * 1) Per-color chromaticity model.
     *
     * hue_band is recorded in Lab chromaticity degrees:
     *     theta = atan2(b* - 128, a* - 128)        (OpenCV 8-bit Lab: a/b
     *     centered at 128, ratio is scale-free so degrees match Python)
     * The band is applied with CROSS-PRODUCT edge tests (cheap, opencv
     * element-wise ops only, no per-pixel atan2 needed):
     *     edge F at hue_band[0], edge B at hue_band[1] (+360 for wrap)
     *     inBand <=> (Fx*db - Fy*da >= 0) && (da*By - db*Bx >= 0)
     * sat_floor is the chroma floor at reference brightness (ref_l), scaled
     * each frame by scl = clamp(meanL/ref_l, scl_lo, scl_hi). l_floor only
     * kills near-black pixels (shadow-tough).
     * ----------------------------------------------------------------*/
    public enum BallColor { YELLOW, RED, BLUE }

    // theta = atan2(b*-128, a*-128) in degrees. Red wraps through 0.
    private static final double[] YELLOW_LO = { 22.3, 105.0 };
    private static final double   YELLOW_SAT = 26.08;
    private static final double[] RED_LO    = { 335.0, 62.0 };
    private static final double   RED_SAT   = 12.36;
    private static final double[] BLUE_LO   = { 258.0, 326.0 };
    private static final double   BLUE_SAT  = 7.44;
    private static final double   MIN_L     = 8.0;

    private static final double REF_L  = 51.7;   // ambient L* the fitter measured
    private static final double SCL_LO = 0.25;   // sat-floor gain limits
    private static final double SCL_HI = 1.5;

    // Edge unit-vectors of each hue band (precomputed from the degrees).
    private static final double[] YELLOW_F = { Math.cos(Math.toRadians(YELLOW_LO[0])),
                                               Math.sin(Math.toRadians(YELLOW_LO[0])) };
    private static final double[] YELLOW_B = { Math.cos(Math.toRadians(YELLOW_LO[1])),
                                               Math.sin(Math.toRadians(YELLOW_LO[1])) };
    // red wraps: back-edge angle = 62.0 + 360 (use 62.0 directly, cos/sin coincide)
    private static final double[] RED_F = { Math.cos(Math.toRadians(RED_LO[0])),
                                            Math.sin(Math.toRadians(RED_LO[0])) };
    private static final double[] RED_B = { Math.cos(Math.toRadians(RED_LO[1])),
                                            Math.sin(Math.toRadians(RED_LO[1])) };
    private static final double[] BLUE_F = { Math.cos(Math.toRadians(BLUE_LO[0])),
                                             Math.sin(Math.toRadians(BLUE_LO[0])) };
    private static final double[] BLUE_B = { Math.cos(Math.toRadians(BLUE_LO[1])),
                                             Math.sin(Math.toRadians(BLUE_LO[1])) };

    /* ------------------------------------------------------------------
     * 2) Shape/size gating. The numbers came from YOLO boxes at 1080p
     *    dev footage; Lab/gate are scaled to the live camera resolution:
     *        scale = (w*h) / (1920*1080)
     *    aspect and fill are scale-free. This assumes the dev video and
     *    the robot camera view the field from a similar distance.
     * ----------------------------------------------------------------*/
    private static final double REF_MPIX  = 1920.0 * 1080.0;
    private static final double MIN_AREA_RATIO = 510.0 / REF_MPIX;
    private static final double MAX_AREA_RATIO = 19720.0 / REF_MPIX;
    private static final double MIN_ASPECT = 0.744;
    private static final double MAX_ASPECT = 1.509;
    private static final double MIN_FILL = 0.5;
    private static final double MAX_FILL = 1.3;
    private static final boolean USE_MORPH = true;

    private static final double EXPECT_YELLOW_AREA = 1950.0;
    private static final double EXPECT_RED_AREA    = 2986.0;
    private static final double EXPECT_BLUE_AREA   = 3182.0;

    private static double expectArea(BallColor c) {
        switch (c) {
            case YELLOW: return EXPECT_YELLOW_AREA;
            case BLUE:   return EXPECT_BLUE_AREA;
            default:     return EXPECT_RED_AREA;
        }
    }

    private boolean suspend = false;
    private boolean detect  = true;
    private final Object lock = new Object();
    private List<BallBlob> detections = new ArrayList<>();

    /** One detected ball candidate for the OpMode. */
    public static class BallBlob {
        public final BallColor color;
        public final Rect   rect;
        public final double area;
        public final double cx, cy;
        public double cxNorm = 0, cyNorm = 0, wNorm = 0, hNorm = 0;

        BallBlob(BallColor c, Rect r, double a) {
            color = c;
            rect = r;
            area = a;
            cx = r.x + r.width / 2.0;
            cy = r.y + r.height / 2.0;
        }

        void normalize(int frameW, int frameH) {
            cxNorm = cx / frameW;
            cyNorm = cy / frameH;
            wNorm = rect.width / (double) frameW;
            hNorm = rect.height / (double) frameH;
        }
    }

    /** Most ball-like detected blob of a color (candidate, not ground truth). */
    public BallBlob bestOf(BallColor color) {
        BallBlob best = null;
        double bestCost = Double.MAX_VALUE;
        synchronized (lock) {
            for (BallBlob b : detections) {
                if (b.color != color) continue;
                double fill = b.area / (double) (b.rect.width * b.rect.height);
                double cost = Math.abs(b.area / expectArea(color) - 1.0)
                            + 4.0 * (1.0 - fill);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = b;
                }
            }
        }
        return best;
    }

    public List<BallBlob> getDetections() {
        synchronized (lock) {
            return new ArrayList<>(detections);
        }
    }

    public void suspendDetection() { detect = false; }
    public void resumeDetection()  { detect = true; }
    public void suspend()          { suspend = true; }
    public void resume()           { suspend = false; }

    @Override
    public void init(int width, int height, org.firstinspires.ftc.vision.CameraCalibration calibration) {
        // Nothing per-resolution; area gates scale from the 1080p reference.
    }

    @Override
    public Mat processFrame(Mat input, long captureTimeNanos) {
        if (suspend) {
            return input; // return unmodified
        }
        List<BallBlob> out = new ArrayList<>();
        if (detect) {
            double scale = (input.cols() * (double) input.rows()) / REF_MPIX;
            int minAreaPx = (int) Math.max(40.0, MIN_AREA_RATIO * scale);
            int maxAreaPx = (int) Math.max(minAreaPx + 1, MAX_AREA_RATIO * scale);

            // One Lab conversion + one set of chroma Mats for all three colors.
            Mat lab = new Mat();
            Imgproc.cvtColor(input, lab, Imgproc.COLOR_RGB2Lab);
            Mat L = new Mat(), a = new Mat(), b = new Mat();
            Core.extractChannel(lab, L, 0);
            Core.extractChannel(lab, a, 1);
            Core.extractChannel(lab, b, 2);
            lab.release();

            Mat da = new Mat(), db = new Mat();
            a.convertTo(da, CvType.CV_32F);          // a - 128 (a/b are 0..255, centered 128)
            b.convertTo(db, CvType.CV_32F);
            a.release(); b.release();
            Core.subtract(da, new Scalar(128.0), da);
            Core.subtract(db, new Scalar(128.0), db);

            Mat sat = new Mat();
            Mat da2 = new Mat(), db2 = new Mat();
            Core.multiply(da, da, da2);
            Core.multiply(db, db, db2);
            Core.add(da2, db2, sat);
            da2.release(); db2.release();
            Core.sqrt(sat, sat);                    // sat = hypot(a*, b*)

            // Adaptive chroma gain from the frame's ambient brightness.
            double ambL = Core.mean(L).val[0];
            double scl = Math.min(SCL_HI, Math.max(SCL_LO, ambL / REF_L));

            maskAndFind(da, db, sat, L, BallColor.YELLOW,
                        YELLOW_F, YELLOW_B, YELLOW_SAT, scl, MIN_L, minAreaPx, maxAreaPx, out);
            maskAndFind(da, db, sat, L, BallColor.RED,
                        RED_F, RED_B, RED_SAT, scl, MIN_L, minAreaPx, maxAreaPx, out);
            maskAndFind(da, db, sat, L, BallColor.BLUE,
                        BLUE_F, BLUE_B, BLUE_SAT, scl, MIN_L, minAreaPx, maxAreaPx, out);

            da.release(); db.release(); sat.release(); L.release();
        }
        for (BallBlob b : out) b.normalize(input.cols(), input.rows());

        synchronized (lock) { detections = out; }

        Mat output = input.clone();
        if (detect) drawOverlay(output, out);
        return output;
    }

    @Override
    public void onDrawFrame(android.graphics.Canvas canvas, int onscreenWidth, int onscreenHeight,
                            float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        // No custom canvas drawing needed.
    }

    private void maskAndFind(Mat da, Mat db, Mat sat, Mat Lm, BallColor color,
                             double[] f, double[] bvec, double satFloorRef, double scl,
                             double minL, int minAreaPx, int maxAreaPx, List<BallBlob> out) {
        double satFloor = satFloorRef * scl;

        // inBand <=> (fx*db - fy*da >= 0) && (da*by - db*bx >= 0)
        Mat c1 = new Mat(), c2 = new Mat(), xf = new Mat(), xb = new Mat();
        Core.multiply(db, new Scalar(f[0]), c1);          //  fx*db
        Core.multiply(da, new Scalar(f[1]), c2);          //  fy*da
        Core.subtract(c1, c2, xf);                        //  fx*db - fy*da
        Core.multiply(da, new Scalar(bvec[1]), c1);       //  da*by
        Core.multiply(db, new Scalar(bvec[0]), c2);       //  db*bx
        Core.subtract(c1, c2, xb);                        //  da*by - db*bx

        Mat m1 = new Mat(), m2 = new Mat(), m3 = new Mat(), m4 = new Mat();
        Core.compare(xf, new Scalar(0.0), m1, Core.CMP_GE);
        Core.compare(xb, new Scalar(0.0), m2, Core.CMP_GE);
        Core.compare(sat, new Scalar(satFloor), m3, Core.CMP_GT);
        Core.compare(Lm, new Scalar(minL), m4, Core.CMP_GT);   // Lm is CV_8U; scalar compares fine
        xf.release(); xb.release(); c1.release(); c2.release();

        Mat mask = new Mat();
        Core.bitwise_and(m1, m2, mask);
        Core.bitwise_and(mask, m3, mask);
        Core.bitwise_and(mask, m4, mask);
        m1.release(); m2.release(); m3.release(); m4.release();

        if (USE_MORPH) {
            Mat el = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                                                   new org.opencv.core.Size(5, 5));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, el);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, el);
            el.release();
        }

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hier = new Mat();
        Imgproc.findContours(mask, contours, hier, Imgproc.RETR_EXTERNAL,
                             Imgproc.CHAIN_APPROX_SIMPLE);
        hier.release();
        mask.release();
        for (MatOfPoint ct : contours) {
            double area = Imgproc.contourArea(ct);
            Rect r = Imgproc.boundingRect(ct);
            ct.release();
            if (area < minAreaPx || area > maxAreaPx) continue;
            if (r.width <= 0 || r.height <= 0) continue;
            double aspect = (double) r.width / r.height;
            if (aspect < MIN_ASPECT || aspect > MAX_ASPECT) continue;
            double fill = area / (double) (r.width * r.height);
            if (fill < MIN_FILL || fill > MAX_FILL) continue;
            out.add(new BallBlob(color, r, area));
        }
    }

    private void drawOverlay(Mat frame, List<BallBlob> blobs) {
        for (BallBlob b : blobs) {
            Scalar col = b.color == BallColor.YELLOW ? new Scalar(0, 255, 255)
                       : b.color == BallColor.RED    ? new Scalar(0, 0, 255)
                       : new Scalar(255, 0, 0);
            Imgproc.rectangle(frame, b.rect, col, 2);
            Imgproc.putText(frame, b.color.name(),
                            new Point(b.rect.x, b.rect.y - 6),
                            Imgproc.FONT_HERSHEY_PLAIN, 1.0, col, 1);
        }
    }
}