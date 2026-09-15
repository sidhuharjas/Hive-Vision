# -*- coding: utf-8 -*-
"""Score Lab-track configs (and, optionally, the HSV config) against flagship
YOLO ground truth on the real match video.

Truth = ball-like boxes from the flagship model (conf >= 0.35) cached in
lab/docs/yolo_truth_lab.json. Detection mirrors what an OpMode trusts:
gated blobs -> best-blob-per-color -> check the best blob overlaps a YOLO box
of the same color. Scoring is frame-based like the CV track, so the Lab and
HSV numbers in docs/lab_detector_report.md are directly comparable.

Usage:
    python eval_lab_vs_yolo.py --source path/to/match.mp4 --config lab_tuned.json
    python eval_lab_vs_yolo.py --source path/to/match.mp4 --config lab_tuned.json \
                                --compare-hsv        # run cv/tools/hsv_tuned.json too
    python eval_lab_vs_yolo.py --source path/to/match.mp4 --config lab_tuned.json \
                                --truth-json ../docs/yolo_truth_lab.json   # skip model
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import lab_lib
from truth_lib import build_truth

BASE = Path(__file__).resolve().parent
DEFAULT_TRUTH = str(BASE / ".." / "docs" / "yolo_truth_lab.json")
DEFAULT_HSV = str(BASE / ".." / ".." / "cv" / "tools" / "hsv_tuned.json")
COLORS = lab_lib.COLORS


def load_hsv_config(path):
    raw = json.loads(Path(path).read_text())
    segs = {c: [(np.array(s[0], dtype=np.int32), np.array(s[1], dtype=np.int32))
                for s in raw[c]["segments"]]
            for c in COLORS}
    # reuses lab_lib's numeric defaults for gates + expect so the comparison is
    # apples-to-apples with the Lab config on the SAME geometry scoring.
    return {"hsv": {c: {"segments": segs[c]} for c in COLORS},
            "gate": dict(lab_lib.DEFAULT_GATE),
            "expect_area": dict(lab_lib.DEFAULT_EXPECT),
            "best": dict(lab_lib.DEFAULT_BEST)}


def pick_best(blobs, style, fw, ew, expect, frame_size):
    if not blobs:
        return None
    if style == "biggest":
        return max(blobs, key=lambda b: b[4])
    if style == "nearest":
        return min(blobs, key=lambda b: abs(b[4] / expect - 1.0))
    if style == "mixed":
        W, H = frame_size

        def cost(b):
            x, y, w, h, a = b
            edge = ew if (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1) else 0.0
            fill = a / (w * h)
            return abs(a / expect - 1.0) + fw * (1.0 - fill) + edge

        return min(blobs, key=cost)
    raise SystemExit(f"unknown score style: {style}")


def overlap(blob, box):
    x, y, w, h, _ = blob
    bx1, by1, bx2, by2 = box[1:]
    cx, cy = x + w / 2, y + h / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="",
                    help="REQUIRED: input video (must match the truth labels)")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--compare-hsv", action="store_true",
                    help="also score cv/tools/hsv_tuned.json on the same truth")
    ap.add_argument("--truth-json", default=DEFAULT_TRUTH)
    ap.add_argument("--weights", default="")
    ap.add_argument("--score", default="mixed")
    ap.add_argument("--fill-w", type=float, default=4.0)
    ap.add_argument("--edge-w", type=float, default=1.0)
    ap.add_argument("--fnlog", type=str, default="")
    ap.add_argument("--dumpdir", type=str, default="")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="freeze the sat-floor scale at 1.0 (no luminance-adaptive gain)")
    a = ap.parse_args()

    if not a.source:
        raise SystemExit("eval_lab_vs_yolo.py: pass --source path/to/video.mp4")
    weights = a.weights or __import__("truth_lib").WEIGHTS
    truth = build_truth(a.source, a.truth_json, weights)
    print(f"truth frames: {len(truth)}")

    cfgs = [("lab", str(Path(p).resolve())) for p in a.config]
    if a.compare_hsv:
        cfgs.append(("hsv", a.compare_hsv if isinstance(a.compare_hsv, str)
                     else DEFAULT_HSV))
        if not Path(cfgs[-1][1]).exists():
            raise SystemExit(f"HSV config not found: {cfgs[-1][1]}")

    dumpdir = Path(a.dumpdir) if a.dumpdir else None
    if dumpdir:
        dumpdir.mkdir(parents=True, exist_ok=True)
    fnlog = open(a.fnlog, "w") if a.fnlog else None
    if fnlog:
        fnlog.write("frame,color,n_inrange,reason\n")
    dumped = {c: 0 for c in COLORS}
    MAX_DUMP = 40

    scores = {p: {c: dict(tp=0, fn=0, fp=0, yolo_frames=0,
                          dark_tp=0, dark_fn=0, dark_frames=0)
                  for c in COLORS}
              for (_k, p) in cfgs}
    loaded = {}
    cap = cv2.VideoCapture(a.source)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lbl = truth.get(n, [])
        by_color = {c: [b for b in lbl if b[0] == c] for c in COLORS}
        feat = lab_lib.features(frame)
        Ls = feat["L"][::4, ::4]
        box_means = {}
        for c in COLORS:
            if by_color[c]:
                vals = []
                for (_c, x1, y1, x2, y2) in by_color[c]:
                    x0, y0 = max(0, int(x1 // 4)), max(0, int(y1 // 4))
                    x1i, y1i = min(Ls.shape[1], int(x2 // 4)), min(Ls.shape[0], int(y2 // 4))
                    if x1i > x0 and y1i > y0:
                        vals.append(float(Ls[y0:y1i, x0:x1i].mean()))
                box_means[c] = min(vals) if vals else 120.0
        for kind, path in cfgs:
            if path not in loaded:
                if kind == "hsv":
                    loaded[path] = load_hsv_config(path)
                else:
                    loaded[path] = lab_lib.load_config(path)
            cfg = loaded[path]
            gate_cfg = cfg["gate"]
            if kind == "hsv":
                allb = lab_lib.detect_hsv(frame, cfg)
            else:
                am = cfg["ambient"]
                scl = (1.0 if a.no_adaptive
                       else lab_lib.adapt_scale(feat["amb"], am["ref_l"],
                                                am["scl_lo"], am["scl_hi"]))
                allb = {c: lab_lib.blobs_from_mask(
                            lab_lib.mask_for(feat, cfg["colors"][c], scl),
                            gate_cfg)
                        for c in COLORS}
            det = {c: pick_best(allb[c], a.score, a.fill_w, a.edge_w,
                                cfg["expect_area"][c], frame.shape[1::-1])
                   for c in COLORS}
            s = scores[path]
            for c in COLORS:
                yolo_frames = len(by_color[c]) > 0
                dark = yolo_frames and box_means[c] < 40.0
                if yolo_frames:
                    s[c]["yolo_frames"] += 1
                    if dark:
                        s[c]["dark_frames"] += 1
                blob = det[c]
                if blob is None:
                    if yolo_frames:
                        s[c]["fn"] += 1
                        if dark:
                            s[c]["dark_fn"] += 1
                        n_in, reason = (0, "TOO_SMALL")
                        if fnlog:
                            fnlog.write(f"{n},{c},0,GATE\n")
                        if dumpdir is not None and dumped[c] < MAX_DUMP:
                            cv2.imwrite(str(dumpdir / f"fn_{c}_{n:05d}.jpg"), frame)
                            dumped[c] += 1
                else:
                    if any(overlap(blob, b) for b in by_color[c]):
                        s[c]["tp"] += 1
                        if dark:
                            s[c]["dark_tp"] += 1
                    elif yolo_frames:
                        s[c]["fn"] += 1
                        s[c]["fp"] += 1
                        if fnlog:
                            fnlog.write(f"{n},{c},0,HIT_BUT_BEST_WRONG\n")
                        if dumpdir is not None and dumped[c] < MAX_DUMP:
                            cv2.imwrite(str(dumpdir / f"wrongbest_{c}_{n:05d}.jpg"),
                                        frame)
                            dumped[c] += 1
                    else:
                        s[c]["fp"] += 1
        n += 1
    cap.release()
    if fnlog:
        fnlog.close()

    for (_kind, p) in cfgs:
        print(f"\n{p}")
        for c in COLORS:
            s = scores[p][c]
            prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
            rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
            drec = s["dark_tp"] / (s["dark_tp"] + s["dark_fn"]) \
                if s["dark_tp"] + s["dark_fn"] else 0
            print(f"  {c:8s} yolo_frames={s['yolo_frames']:4d} "
                  f"(dark {s['dark_frames']:3d})  "
                  f"tp={s['tp']:4d} fn={s['fn']:4d} fp={s['fp']:4d}  "
                  f"prec={prec*100:5.1f}%  rec={rec*100:5.1f}%  "
                  f"dark_rec={drec*100:5.1f}%")
    if dumpdir:
        print(f"\ndumped FN frames to {dumpdir}: {dumped}")


if __name__ == "__main__":
    main()