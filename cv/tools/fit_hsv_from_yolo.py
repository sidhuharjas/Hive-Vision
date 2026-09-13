#!/usr/bin/env python3
"""Mine HSV limits for the ball colors from the REAL match video, using
fused_v7f YOLO detections as the label source instead of render crops.

Why: the mined-from-render ranges were wide enough to also match floor,
field and robot colors. Real footage gives the actual color distributions.

For every YOLO ball box (conf >= MIN_CONF, after the sphere filter drops
giant robot panels) the box pixels are collected. Robust percentiles per
class are reported. Byte-order note: red's hue wraps through 0, so its range
is split into two segments.

Usage:
    python fit_hsv_from_yolo.py [--source path/video.mp4]
                                [--weights runs/detect/runs/fused/fused_v7f/weights/best.pt]
                                [--conf 0.35] [--imgsz 960]
                                [--pctl 2,98] [--out hsv_from_yolo.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
NAMES = {0: "yellow", 1: "red", 2: "blue"}
DEFAULT_W = str(BASE.parent.parent / "yolo" / "weights" / "best.pt")
SPLIT_H = 8
SV_MIN = (25, 35)          # drop shadows / desaturated edges during sampling
DEBUG = False


def ball_like(x1, y1, x2, y2, W, H):
    w, h = float(x2 - x1), float(y2 - y1)
    return (0.55 <= w / h <= 1.8) and (w * h <= 0.031 * W * H)


def _band(hue, pct=97.0, pad=3.0):
    """(lo, hi) for one hue population >= 30 px, wrap-aware."""
    th = hue * np.pi / 90.0
    m = float(np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) / 2.0 % 180.0)
    d = np.abs(hue - m)
    d = np.minimum(d, 180.0 - d)
    half = float(np.percentile(d, pct)) + pad if len(d) else 5.0
    return max(0.0, m - half), min(180.0, m + half)


def _segments_for(lo, hi, s_lo, v_lo, v_hi):
    if lo < 0:                                  # wraps through 0
        return [[[0, s_lo, v_lo], [int(hi), 255, v_hi]],
                [[int(180 + lo), s_lo, v_lo], [179, 255, v_hi]]]
    if hi > 180:
        return [[[0, s_lo, v_lo], [int(min(180, hi)), 255, v_hi]],
                [[int(lo), s_lo, v_lo], [179, 255, v_hi]]]
    return [[[int(lo), s_lo, v_lo], [int(hi), 255, v_hi]]]


def to_segments(px, s_floor=60, s_cap=90, v_floor=35, v_cap=45,
                band_pct=97.0, pad=3.0):
    """HSV segments from vivid ball pixels; red's wrap gives two segments.

    Two-stage: (1) keep only hues within 15 units of the dominant histogram
    peak so robot-panel colors can't drag the range wide; (2) split the core
    at hue 90 so red's 0-side and 180-side each get their own tight band.

    S/V lows are clamped to recall-friendly floors: a ball in shadow or at a
    steep angle drops way below its brightly-lit percentile, so the range
    floor must not ride the lit-ball percentile.
    """
    vivid = px[(px[:, 1] >= 50) & (px[:, 2] >= 50)]
    if len(vivid) == 0:
        vivid = px
    hist = np.bincount(vivid[:, 0].astype(int), weights=vivid[:, 1], minlength=180)
    peak = int(np.argmax(hist))
    hf = vivid[:, 0].astype(np.int32)          # avoid uint8 wrap in subtraction
    lin = (hf - peak + 90.0) % 180.0 - 90.0    # -90..90 about the peak
    core = vivid[np.abs(lin) <= 15]
    if len(core) < 50:
        core = vivid
    s_lo = int(np.percentile(core[:, 1], 10))
    s_lo = min(max(s_lo, s_floor), s_cap)              # keep it in 60..90
    v_lo = max(v_floor, min(v_cap, int(np.percentile(core[:, 2], 1))))
    v_hi = int(np.percentile(core[:, 2], 98))
    hc = core[:, 0]
    if DEBUG:
        lo_bin, hi_bin = hc.min(), hc.max()
        n_lo = int((hc <= 90).sum())
        n_hi = int((hc > 90).sum())
        print(f"    debug: core n={len(core)} hue[{int(lo_bin)},{int(hi_bin)}] "
              f"n<=90={n_lo} n>90={n_hi} peak={peak}")
    segs = []
    for side in (hc <= 90, hc > 90):           # red is bimodal across the wrap
        g = hc[side]
        if len(g) >= 30:
            segs += _segments_for(*_band(g, band_pct, pad), s_lo, v_lo, v_hi)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="",
                    help="REQUIRED: input video to mine ball pixels from")
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", default="hsv_from_yolo.json")
    ap.add_argument("--s-floor", type=int, default=60)
    ap.add_argument("--s-cap", type=int, default=90)
    ap.add_argument("--v-floor", type=int, default=35)
    ap.add_argument("--v-cap", type=int, default=45)
    ap.add_argument("--band-pct", type=float, default=97.0)
    ap.add_argument("--pad", type=float, default=3.0)
    a = ap.parse_args()

    if not a.source:
        raise SystemExit("fit_hsv_from_yolo.py: pass --source path/to/video.mp4")
    if not Path(a.weights).exists():
        raise SystemExit(f"weights not found: {a.weights}")
    model = YOLO(a.weights)
    if not cv2.VideoCapture(a.source).isOpened():
        raise SystemExit(f"cannot open video: {a.source}")

    # Pass 1: run YOLO, keep ball-like boxes per frame.
    frame_boxes = defaultdict(list)
    n_boxes = 0
    fidx = 0
    for result in model.predict(source=a.source, stream=True, imgsz=a.imgsz,
                                conf=a.conf, verbose=False, half=True):
        w0, h0 = result.orig_shape[1], result.orig_shape[0]
        for b in result.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            if not ball_like(x1, y1, x2, y2, w0, h0):
                continue
            c = int(b.cls[0])
            if c in NAMES:
                frame_boxes[fidx].append((c, x1, y1, x2, y2))
                n_boxes += 1
        fidx += 1

    # Pass 2: read frames, collect HSV pixels inside each YOLO box.
    per_class_px = defaultdict(list)
    cap = cv2.VideoCapture(a.source)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fidx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if fidx in frame_boxes:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for (c, x1, y1, x2, y2) in frame_boxes[fidx]:
                x0, y0 = max(0, int(x1)), max(0, int(y1))
                x1i, y1i = min(W, int(x2)), min(H, int(y2))
                if x1i - x0 < 4 or y1i - y0 < 4:
                    continue
                px = hsv[y0:y1i, x0:x1i].reshape(-1, 3)
                px = px[(px[:, 1] >= SV_MIN[0]) & (px[:, 2] >= SV_MIN[1])]
                if len(px) < 30:
                    continue
                per_class_px[c].append(px)
        fidx += 1
    cap.release()

    limits = {}
    for c in sorted(per_class_px):
        if not per_class_px[c]:
            print(f"{NAMES[c]}: NO pixels collected")
            continue
        px = np.concatenate(per_class_px[c], axis=0)
        segs = to_segments(px, a.s_floor, a.s_cap, a.v_floor, a.v_cap,
                           a.band_pct, a.pad)
        limits[NAMES[c]] = {"segments": segs}
        vivid = px[(px[:, 1] >= 50) & (px[:, 2] >= 50)]
        hist = np.bincount(vivid[:, 0].astype(int), weights=vivid[:, 1], minlength=180)
        peak = int(np.argmax(hist))
        print(f"{NAMES[c]:8s} pixel n={len(px):6d}  peakH={peak}")
        for seg in segs:
            print(f"    segment Lower {seg[0]}  Upper {seg[1]}")

    out = BASE / a.out
    out.write_text(json.dumps(limits, indent=2))
    print(f"\nYOLO labels: {n_boxes} ball boxes over {fidx} frames; wrote {out}")


if __name__ == "__main__":
    main()