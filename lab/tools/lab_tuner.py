#!/usr/bin/env python3
"""Live Lab chromaticity calibration for the Lab ball detector.

Shows the hue-band mask + detected blobs for one color at a time with drag
trackbars. Use a saved frame (or webcam) from your actual field camera and
tune until only the ball lights up. Unlike an HSV tuner, the knobs here are
the Lab-native ones: the chromaticity hue band (center +/- width in degrees)
and the chroma floor (scaled by the frame's ambient L*).

Usage:
    python lab_tuner.py path/to/frame.jpg   # single saved frame
    python lab_tuner.py --camera 0           # live webcam

Keys:
    1/2/3   pick color yellow/red/blue
    r       print current settings (copy into LabBallDetectorPipeline.java)
    s       save current frame + mask preview to tune-<color>.jpg
    q       quit
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import lab_lib
from lab_lib import COLORS

BASE = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE / "lab_tuned.json"
WIN = "lab_tuner"
NAMES = {1: "yellow", 2: "red", 3: "blue"}


def nothing(_):
    pass


def color_index():
    return {v: k for k, v in NAMES.items()}


def windows(cfg, color):
    cv2.destroyWindow(WIN)
    cv2.namedWindow(WIN)
    m = cfg["colors"][color]
    lo, hi = m["hue_band"]
    center = lo if lo <= hi else lo - 360.0          # normalize wrap to -180..180
    width = hi - lo if lo <= hi else (hi + 360.0) - lo
    am = cfg["ambient"]
    cv2.createTrackbar("hue center", WIN, int(center), 359, nothing)
    cv2.createTrackbar("hue width", WIN, int(width), 359, nothing)
    cv2.createTrackbar("sat floor", WIN, int(m["sat_floor"]), 255, nothing)
    cv2.createTrackbar("L floor", WIN, int(m["l_floor"]), 255, nothing)
    cv2.createTrackbar("ref L", WIN, int(am["ref_l"]), 255, nothing)
    cv2.createTrackbar("scl lo x100", WIN, int(am["scl_lo"] * 100), 300, nothing)
    cv2.createTrackbar("scl hi x100", WIN, int(am["scl_hi"] * 100), 300, nothing)


def model_from_track(cfg, color):
    am = cfg["ambient"]
    return {
        "hue_band": _band(
            cv2.getTrackbarPos("hue center", WIN),
            cv2.getTrackbarPos("hue width", WIN)),
        "sat_floor": cv2.getTrackbarPos("sat floor", WIN),
        "l_floor": float(cv2.getTrackbarPos("L floor", WIN)),
        "_amb": {
            "ref_l": cv2.getTrackbarPos("ref L", WIN),
            "scl_lo": cv2.getTrackbarPos("scl lo x100", WIN) / 100.0,
            "scl_hi": cv2.getTrackbarPos("scl hi x100", WIN) / 100.0,
        },
    }


def _band(center, width):
    width = max(2, width)
    lo = (center - width / 2.0) % 360.0
    hi = (center + width / 2.0) % 360.0
    if lo < 1e-6 or hi < 1e-6:
        # touch 0: normalize the wrap so lo > hi
        lo = (center - width / 2.0) % 360.0
        hi = (center + width / 2.0) % 360.0
    if lo > hi:
        pass  # red-style wrap band
    return [round(lo, 1), round(hi, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="image file path (or use --camera)")
    ap.add_argument("--camera", type=int, help="webcam index")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args()

    cfg = lab_lib.load_config(a.config)
    if a.source and a.camera is None:
        cap = None
        frame = cv2.imread(a.source)
        if frame is None:
            raise SystemExit(f"cannot read image: {a.source}")
    elif a.camera is not None:
        cap = cv2.VideoCapture(a.camera)
        ok, frame = cap.read()
    else:
        print("give an image path or --camera")
        sys.exit(1)

    color = 1
    windows(cfg, NAMES[color])
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        feat = lab_lib.features(frame)
        m = model_from_track(cfg, NAMES[color])
        amb = m["_amb"]
        scl = lab_lib.adapt_scale(feat["amb"], amb["ref_l"],
                                  amb["scl_lo"], amb["scl_hi"])
        model = m
        mask = lab_lib.mask_for(feat, model, scl)
        nm = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(nm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()
        for ct in cnts:
            x, y, w, h = cv2.boundingRect(ct)
            area = cv2.contourArea(ct)
            if area < 50 or w <= 0 or h <= 0:
                continue
            if not (0.4 < area / (w * h) < 1.6):
                continue
            cv2.rectangle(vis, (x, y), (x + w, y + h),
                          (0, 255, 255) if color == 1 else
                          (0, 0, 255) if color == 2 else (255, 0, 0), 2)
        cv2.putText(vis, f"{NAMES[color]}  ambL={feat['amb']:.0f}  scl={scl:.2f} "
                         f"band={model['hue_band']}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("mask", nm)
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('1'), ord('2'), ord('3')):
            color = k - ord('0')
            windows(cfg, NAMES[color])
        if k == ord('r'):
            print(f"{NAMES[color]:7s} hue_band={model['hue_band']} "
                  f"sat_floor={model['sat_floor']:.0f}  l_floor={model['l_floor']:.0f}  "
                  f"ref_l={amb['ref_l']:.0f}  scl=[{amb['scl_lo']:.2f},{amb['scl_hi']:.2f}]")
        if k == ord('s'):
            cv2.imwrite(f"tune_{NAMES[color]}.jpg",
                        np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
            print("saved tune_{}.jpg".format(NAMES[color]))
        if k in (ord('q'), 27):
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()