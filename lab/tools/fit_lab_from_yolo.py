# -*- coding: utf-8 -*-
"""Fit the Lab-track detector from the flagship YOLO model.

Runs the shipped Hive Vision YOLO weights over real footage and learns, per
ball color, a chromaticity model for the CIELAB hue band + chroma floor that
the Control Hub pipeline uses. This mirrors fit_hsv_from_yolo.py for the raw-CV
track, but in the space where a shadow keeps hue nearly constant:

  * hue_band  = circular median of theta = atan2(b*, a*) +/- a robust half-width
  * sat_floor = chroma required at reference lighting (scaled down in shadow
                by the frame illuminance so shadowed balls stay above it)
  * l_floor   = L* floor to drop near-black robot parts

Everything stored in the output JSON is exactly what lab_lib.detect() and
LabBallDetectorPipeline.java consume, so retraining = copy the constants into
the Java file.

Usage:
    python fit_lab_from_yolo.py --source path/to/match.mp4
                                [--weights neural-net/weights/best.pt]
                                [--conf 0.35] [--imgsz 960]
                                [--band-pct 92.0] [--out lab_tuned.json]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lab_lib import AMBI_STRIDE, COLORS, MIN_L
from truth_lib import build_truth

BASE = Path(__file__).resolve().parent
DEFAULT_TRUTH = str(BASE / ".." / "docs" / "yolo_truth_lab.json")
DEFAULT_OUT = str(BASE / "lab_tuned.json")
BALL_FILL = 0.785          # circle-in-box fill used to convert box->contour area


def hvars(px):
    """Return (hue, sat, L) arrays for Lab ball pixels."""
    L = px[:, 0]
    a = px[:, 1] - 128.0
    b = px[:, 2] - 128.0
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    sat = np.sqrt(a * a + b * b)
    return hue, sat, L


def circular(xs):
    th = np.radians(xs)
    return float(np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360.0)


def band_for(hue, pct, hw_clamp=(18.0, 60.0)):
    """(lo, hi) hue band from chromatically meaningful pixels: circular median
    +/- the pct circular-width. lo > hi encodes wrap-through-0 (e.g. red)."""
    med = circular(hue)
    if len(hue) < 50:
        half = 25.0
    else:
        d = np.abs(hue - med) % 360.0
        d = np.minimum(d, 360.0 - d)
        half = float(np.clip(np.percentile(d, pct), *hw_clamp))
    lo, hi = med - half, med + half
    if lo < 0:
        return round(lo + 360.0, 1), round(hi, 1)      # wraps through 0
    if hi > 360:
        return round(lo, 1), round(hi - 360.0, 1)
    return round(lo, 1), round(hi, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="",
                    help="REQUIRED: input video to mine ball pixels from")
    ap.add_argument("--weights", default="")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--band-pct", type=float, default=92.0,
                    help="circular half-width percentile kept in each hue band")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--truth-json", default=DEFAULT_TRUTH,
                    help="cache/load YOLO truth (skip the model on re-runs)")
    ap.add_argument("--frame-sample", type=int, default=1,
                    help="sample every Nth frame for fitting (default 1 = all)")
    a = ap.parse_args()

    if not a.source:
        raise SystemExit("fit_lab_from_yolo.py: pass --source path/to/video.mp4")
    weights = a.weights or __import__("truth_lib").WEIGHTS
    truth = build_truth(a.source, a.truth_json, weights, a.conf, a.imgsz)

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {a.source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    px_per_color = {c: [] for c in COLORS}
    color_areas = {c: [] for c in COLORS}
    all_areas = []
    all_aspects = []
    ambient_ls = []
    n_used = 0

    fidx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if fidx in truth and fidx % a.frame_sample == 0:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            ambient_ls.append(float(np.median(
                lab[::AMBI_STRIDE, ::AMBI_STRIDE, 0])))
            for (c, x1, y1, x2, y2) in truth[fidx]:
                x0, y0 = max(0, int(x1)), max(0, int(y1))
                x1i, y1i = min(W, int(x2)), min(H, int(y2))
                if x1i - x0 < 4 or y1i - y0 < 4:
                    continue
                bw = max(1.0, float(x2 - x1))
                bh = max(1.0, float(y2 - y1))
                color_areas[c].append(bw * bh * BALL_FILL)
                all_areas.append(bw * bh * BALL_FILL)
                all_aspects.append(bw / bh)
                crop = lab[y0:y1i, x0:x1i]
                h, w = crop.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w]
                rr = 0.55 * min(h, w)
                spatial = ((xx - (w - 1) / 2.0) ** 2
                           + (yy - (h - 1) / 2.0) ** 2) <= rr * rr
                px = crop[spatial].reshape(-1, 3)
                px = px[(px[:, 0] > MIN_L) & (px[:, 0] < 248)]
                if len(px) < 30:
                    continue
                px_per_color[c].append(px)
                n_used += 1
        fidx += 1
    cap.release()

    if not any(px_per_color[c] for c in COLORS):
        raise SystemExit("no ball pixels collected - is the video the right source?")

    ref_l = float(np.mean(ambient_ls)) if ambient_ls else 120.0
    limits = {"colors": {}, "ambient": {"ref_l": round(ref_l, 1)},
              "gate": {}, "expect_area": {}, "best": {"fill_w": 4.0, "edge_w": 1.0}
              , "_meta": {
                  "source": a.source,
                  "truth_frames": len(truth),
                  "boxes_sampled": n_used,
                  "frames_sampled": len(ambient_ls),
                  "learned_from": "flagship Hive Vision YOLO (neural-net/weights/best.pt)",
                  "hue_bands": "360-degree chromaticity hue; sat floors scale by amb/ref_l",
              }}

    for c in COLORS:
        if not px_per_color[c]:
            print(f"{c}: NO pixels collected - skipping")
            continue
        px = np.concatenate(px_per_color[c], axis=0).astype(np.float32)
        hue, sat, L = hvars(px)
        lit = L >= 40.0
        chrom = lit & (sat >= 15.0)              # chromatic pixels define the band
        lo, hi = band_for(hue[chrom] if chrom.sum() > 200 else hue, a.band_pct)
        sat_floor = max(2.0, float(np.percentile(sat[lit], 25)) * 0.9)
        l_floor = MIN_L             # only kills near-blacks; shadows keep hue
        areas = np.asarray(color_areas[c])
        limits["colors"][c] = {
            "hue_band": [lo, hi],
            "sat_floor": round(sat_floor, 1),
            "l_floor": round(l_floor, 1),
        }
        limits["expect_area"][c] = round(float(np.median(areas)), 0)
        print(f"{c:8s} px={len(px):6d}  hue_band=[{lo},{hi}]  "
              f"sat_floor={sat_floor:4.1f}  l_floor={l_floor:4.1f}")

    a_all = np.asarray(all_areas)
    s_all = np.asarray(all_aspects)
    limits["gate"] = {
        "min_area_px": int(np.percentile(a_all, 2)),
        "max_area_px": int(np.percentile(a_all, 98)),
        "min_aspect": round(float(np.percentile(s_all, 2)), 3),
        "max_aspect": round(float(np.percentile(s_all, 98)), 3),
        "min_fill": 0.5,
        "max_fill": 1.3,
    }
    print("gate (all colors): "
          f"area {int(np.percentile(a_all,2))}-{int(np.percentile(a_all,98))}, "
          f"aspect {limits['gate']['min_aspect']}-{limits['gate']['max_aspect']}")

    out = Path(a.out)
    out.write_text(json.dumps(limits, indent=2))
    print(f"\nwrote {out}  (ref_l={ref_l:.1f}, {len(ambient_ls)} sampled frames)")
    sys.stdout.write("copy the values into LabBallDetectorPipeline.java to deploy\n")


if __name__ == "__main__":
    main()