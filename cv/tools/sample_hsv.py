#!/usr/bin/env python3
"""Mine HSV limits for the ball colors directly from existing YOLO datasets.

Recycles the labels + images you already trained on, so you never have to
re-collect data for a raw-OpenCV (coprocessor-free) detector.

For every labeled ball box it crops the ball, converts to HSV and collects
pixel samples. It drops shadow/highlight noise (very low V, low saturation)
and reports robust percentiles per class. Red is split into the two hue
segments that wrap around 0 (0..10 and 170..180).

Usage:
    python sample_hsv.py [--data V7f/dataset/concat]
                         [--classes 0,1,2]
                         [--sample 1500]      # max labeled images to scan
                         [--seed 0]

Output is printed and written to hsv_limits.json (OpenCV convention:
H in 0..180, S and V in 0..255 -- exactly what the Java pipeline uses).

Pair with tools/hsv_tuner.py to re-tune the mined ranges live on your
real field camera.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "yellow", 1: "red", 2: "blue"}
PCT_LO, PCT_HI = 2.0, 98.0     # percentile range over collected ball pixels
SPLIT_H = 10                     # where red's hue wraps: [0,SPLIT_H] + [180-SPLIT_H,180]


def hsv_wrap(h_lo, h_hi, lo, hi):
    """If the hue range crosses the 0/180 boundary, split into two segments."""
    if h_lo > SPLIT_H:
        return [("single", lo, hi)]
    if h_hi < 180 - SPLIT_H:
        return [("single", lo, hi)]
    segA = (np.array([lo[0], lo[1], lo[2]]), np.array([SPLIT_H, hi[1], hi[2]]))
    segB = (np.array([180 - SPLIT_H, lo[1], lo[2]]), np.array([hi[0], hi[1], hi[2]]))
    return [("wrapA", segA[0], segA[1]), ("wrapB", segB[0], segB[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"V7f/dataset/concat")
    ap.add_argument("--classes", default="0,1,2")
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = Path(a.data)
    img_d = data / "images" / "train"
    lbl_d = data / "labels" / "train"
    wanted = {int(c) for c in a.classes.split(",")}

    samples = defaultdict(list)   # class -> list of (H,S,V) triplets
    n_files = 0
    n_boxes = 0
    rng = np.random.default_rng(a.seed)
    files = sorted(lbl_d.glob("*.txt"))
    rng.shuffle(files)
    for lbf in files:
        if n_files >= a.sample:
            break
        rows = [l.split() for l in lbf.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        im = img_d / (lbf.with_suffix(".png").name if (img_d / lbf.with_suffix(".png").name).exists()
                      else lbf.with_suffix(".jpg").name)
        if not im.exists():
            continue
        img = cv2.imread(str(im))
        if img is None:
            continue
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for r in rows:
            c = int(float(r[0]))
            if c not in wanted:
                continue
            nx, ny, nw, nh = map(float, r[1:5])
            x0, y0 = int(nx * W - nw * W / 2), int(ny * H - nh * H / 2)
            x1, y1 = int(nx * W + nw * W / 2), int(ny * H + nh * H / 2)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            px = hsv[y0:y1, x0:x1].reshape(-1, 3)
            px = px[(px[:, 1] >= 25) & (px[:, 2] >= 35)]   # drop shadows/grey
            if len(px) < 30:
                continue
            samples[c].append(px)
            n_boxes += 1
        n_files += 1

    print(f"\nscanned {n_files} labeled images using {a.data}")
    print(f"ball boxes sampled: {n_boxes}\n")
    limits = {}
    for c in sorted(samples):
        px = np.concatenate(samples[c], axis=0)
        lo = np.percentile(px, PCT_LO, axis=0).astype(int)
        hi = np.percentile(px, PCT_HI, axis=0).astype(int)
        raw = (lo, hi)
        segs = hsv_wrap(int(lo[0]), int(hi[0]), lo, hi)
        limits[NAMES[c]] = {"images_sampled": len(samples[c]),
                            "raw_lower": raw[0].tolist(), "raw_upper": raw[1].tolist(),
                            "segments": [[s[1].tolist(), s[2].tolist()] for s in segs]}
        print(f"{NAMES[c]:8s} pixel n={len(px):6d}  raw H[{lo[0]}, {hi[0]}] "
              f"S[{lo[1]}, {hi[1]}] V[{lo[2]}, {hi[2]}]")
        for i, (tag, l, u) in enumerate(segs):
            print(f"    range {i}: Lower {l.tolist()}  Upper {u.tolist()}")

    out = Path(__file__).resolve().parent / "hsv_limits.json"
    out.write_text(json.dumps(limits, indent=2))
    print(f"\nlimits written to {out}")
    print("copy the 'segments' of each color into BallDetectorPipeline.HSV_* and re-tune live with hsv_tuner.py")


if __name__ == "__main__":
    main()