# -*- coding: utf-8 -*-
"""Fast Lab-config sweep: evaluates many threshold variants against YOLO truth
in a single video pass (frame features computed once, reused by every combo),
so you can find a shadow-tolerant config in ~1 minute instead of per-combo runs.

Grid: scl_lo x sat-floor multiplier. scl is the luminance-adaptive floor scale
(amb/ref_l clamped); floor_mult shrinks the per-color trained sat_floor, which
helps shadowed balls whose chroma also collapses.

Usage:
    python sweep_lab.py --source path/to/match.mp4 \
        [--base lab_tuned.json] [--truth-json ../docs/yolo_truth_lab.json] \
        [--scl 0.3,0.5,0.7] [--floor 0.4,0.6,0.8,1.0]
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

import lab_lib
from eval_lab_vs_yolo import overlap, pick_best
from truth_lib import build_truth

BASE = Path(__file__).resolve().parent
DEFAULT_TRUTH = str(BASE / ".." / "docs" / "yolo_truth_lab.json")
COLORS = lab_lib.COLORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="")
    ap.add_argument("--base", default=str(BASE / "lab_tuned.json"))
    ap.add_argument("--truth-json", default=DEFAULT_TRUTH)
    ap.add_argument("--weights", default="")
    ap.add_argument("--scl", default="0.3,0.5,0.7",
                    help="comma list of scl_lo values to try")
    ap.add_argument("--floor", default="0.4,0.6,0.8,1.0",
                    help="comma list of sat-floor multipliers")
    a = ap.parse_args()

    if not a.source:
        raise SystemExit("sweep_lab.py: pass --source path/to/video.mp4")

    base = lab_lib.load_config(a.base)
    combos = []
    for scl in map(float, a.scl.split(",")):
        for fm in map(float, a.floor.split(",")):
            v = copy.deepcopy(base)
            v["ambient"]["scl_lo"] = scl
            for c in v["colors"]:
                v["colors"][c]["sat_floor"] = round(v["colors"][c]["sat_floor"] * fm, 2)
            combos.append((f"scl={scl} floor={fm}", v))

    weights = a.weights or __import__("truth_lib").WEIGHTS
    truth = build_truth(a.source, a.truth_json, weights)
    cap = cv2.VideoCapture(a.source)
    scores = {p: {c: dict(tp=0, fn=0, fp=0, yf=0, dtp=0, dfn=0, df=0)
                  for c in COLORS} for p, _ in combos}

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        feat = lab_lib.features(frame)
        Ls = feat["L"][::4, ::4]
        lbl = truth.get(n, [])
        dark_l_by = {}
        by_color = {c: [b for b in lbl if b[0] == c] for c in COLORS}
        for c in COLORS:
            boxes = by_color[c]
            votes = []
            if boxes:
                for (_c, x1, y1, x2, y2) in boxes:
                    x0, y0 = max(0, int(x1 // 4)), max(0, int(y1 // 4))
                    x1i, y1i = min(Ls.shape[1], int(x2 // 4)), min(Ls.shape[0], int(y2 // 4))
                    if x1i > x0 and y1i > y0:
                        votes.append(float(Ls[y0:y1i, x0:x1i].mean()))
                dark_l_by[c] = min(votes) if votes else 120.0
            else:
                dark_l_by[c] = 120.0

        for name, cfg in combos:
            am = cfg["ambient"]
            scl = lab_lib.adapt_scale(feat["amb"], am["ref_l"], am["scl_lo"], am["scl_hi"])
            gate_cfg = cfg["gate"]
            s = scores[name]
            for c in COLORS:
                boxes = by_color[c]
                dark = dark_l_by[c] < 40.0
                if boxes:
                    s[c]["yf"] += 1
                    if dark:
                        s[c]["df"] += 1
                mask = lab_lib.mask_for(feat, cfg["colors"][c], scl)
                blobs = lab_lib.blobs_from_mask(
                    mask, gate_cfg) if mask.any() else []
                det = pick_best(blobs, "mixed", cfg["best"]["fill_w"],
                                cfg["best"]["edge_w"], cfg["expect_area"][c],
                                frame.shape[1::-1])
                if det is None:
                    if boxes:
                        s[c]["fn"] += 1
                        if dark:
                            s[c]["dfn"] += 1
                elif any(overlap(det, b) for b in boxes):
                    s[c]["tp"] += 1
                    if dark:
                        s[c]["dtp"] += 1
                elif boxes:
                    s[c]["fn"] += 1
                    s[c]["fp"] += 1
                    if dark:
                        s[c]["dfn"] += 1
                else:
                    s[c]["fp"] += 1
        n += 1
    cap.release()

    print(f"{'combo':22s} | {'rec y/r/b':>20s} | {'dark_rec y/r/b':>24s} | {'prec y/r/b':>20s}")
    for name, _ in combos:
        s = scores[name]
        rec = " ".join(f"{s[c]['tp']/(s[c]['tp']+s[c]['fn'])*100 if s[c]['tp']+s[c]['fn'] else 0:5.1f}"
                       for c in COLORS)
        dr = " ".join(f"{s[c]['dtp']/(s[c]['dtp']+s[c]['dfn'])*100 if s[c]['dtp']+s[c]['dfn'] else 0:5.1f}"
                      for c in COLORS)
        pr = " ".join(f"{s[c]['tp']/(s[c]['tp']+s[c]['fp'])*100 if s[c]['tp']+s[c]['fp'] else 0:5.1f}"
                      for c in COLORS)
        print(f"{name:22s} | {rec:>20s} | {dr:>24s} | {pr:>20s}")


if __name__ == "__main__":
    main()