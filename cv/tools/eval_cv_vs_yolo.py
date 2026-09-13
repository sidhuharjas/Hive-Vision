#!/usr/bin/env python3
"""Score an HSV config against YOLO ground truth on the real match video.

Truth = fused_v7f ball-like boxes (conf >= MIN_CONF). A frame gets counted for
a color when YOLO labels that color. The CV side uses gated + best-
blob-per-color (exactly what an OpMode would trust), so:
  TP  CV best blob overlaps a YOLO box of the same color
  FN  YOLO labeled the color but CV fired nothing
  FP  CV produced a blob of a color YOLO didn't label that frame

Usage:
    python eval_cv_vs_yolo.py --config <hsv_x.json>
    python eval_cv_vs_yolo.py --config <a.json> --config <b.json>   # compare

YOLO ground truth is expensive to rebuild (the full video pass). Reuse it
across config experiments:
    python eval_cv_vs_yolo.py --config <cfg>.json --truth-json reports/yolo_truth.json
The second run loads labels from the cache and skips the model entirely.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DEFAULT_W = str(BASE.parent.parent / "yolo" / "weights" / "best.pt")
DEFAULT_TRUTH = str(BASE.parent / "docs" / "yolo_truth.json")
COLORS = ("yellow", "red", "blue")
MIN_CONF = 0.35
EXPECT_AREA = {"yellow": 3500.0, "red": 4200.0, "blue": 4200.0}


def pick_best(blobs, color, style, fw=0.0, ew=0.0, frame_size=None):
    if not blobs:
        return None
    if style == "biggest":
        return max(blobs, key=lambda b: b[4])
    if style == "roundest":
        return max(blobs, key=lambda b: b[4] / (b[2] * b[3]))
    if style == "mixed":
        W, H = frame_size
        def cost(b):
            x, y, w, h, a = b
            edge = 0.0
            if ew:
                edge = ew * (1.0 if (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1) else 0.0)
            fill = a / (w * h)
            return abs(a / EXPECT_AREA[color] - 1.0) + fw * (1.0 - fill) + edge
        return min(blobs, key=cost)
    return min(blobs, key=lambda b: abs(b[4] / EXPECT_AREA[color] - 1.0))


def ball_like(x1, y1, x2, y2, W, H):
    w, h = float(x2 - x1), float(y2 - y1)
    return (0.55 <= w / h <= 1.8) and (w * h <= 0.031 * W * H)


def yolo_labels(source, model):
    frame_boxes = {}
    fidx = 0
    for result in model.predict(source=source, stream=True, imgsz=960,
                                conf=MIN_CONF, verbose=False, half=True):
        w0, h0 = result.orig_shape[1], result.orig_shape[0]
        names = result.names
        lbl = []
        for b in result.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            if not ball_like(x1, y1, x2, y2, w0, h0):
                continue
            c = names[int(b.cls[0])]
            if c in COLORS:
                lbl.append((c, x1, y1, x2, y2))
        if lbl:
            frame_boxes[fidx] = lbl
        fidx += 1
    return frame_boxes


def get_truth(source, model, cache_path):
    """Reuse or build the YOLO ground-truth labels. When stored, subsequent
    evals skip the (slow) model pass entirely and just load the JSON."""
    if cache_path and Path(cache_path).exists():
        raw = json.loads(Path(cache_path).read_text())
        truth = {int(k): v for k, v in raw.items()}
        print(f"loaded {len(truth)} truth frames from {cache_path}")
        return truth
    truth = yolo_labels(source, model)
    if cache_path:
        Path(cache_path).write_text(json.dumps(
            {str(k): v for k, v in truth.items()}, indent=0))
        print(f"cached {len(truth)} truth frames -> {cache_path}")
    return truth


def load_segments(path):
    cfg = json.loads(Path(path).read_text())
    return {c: [(np.array(s[0], dtype=np.int32), np.array(s[1], dtype=np.int32))
                for s in cfg[c]["segments"]]
            for c in COLORS}


def cv_detect_all(frame, segs, amin, amax, asp_lo, asp_hi, fill_lo, fill_hi):
    """All gated blobs per color (for debugging), not just the best."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found = {}
    for color, seglist in segs.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in seglist:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for ct in cnts:
            x, y, w, h = cv2.boundingRect(ct)
            area = cv2.contourArea(ct)
            if area < amin or area > amax or w <= 0 or h <= 0:
                continue
            aspect = w / h
            fill = area / (w * h)
            if aspect < asp_lo or aspect > asp_hi or fill < fill_lo or fill > fill_hi:
                continue
            blobs.append((x, y, w, h, area))
        found[color] = blobs
    return found


def cv_detect(frame, segs, amin, amax, asp_lo, asp_hi, fill_lo, fill_hi):
    found = {}
    for color, blobs in cv_detect_all(frame, segs, amin, amax,
                                      asp_lo, asp_hi, fill_lo, fill_hi).items():
        found[color] = max(blobs, key=lambda b: b[4]) if blobs else None
    return found


def classify_miss(frame, boxes, seglist, amin, amax, asp_lo, asp_hi,
                  fill_lo, fill_hi):
    """Why did CV miss? Union of the YOLO boxes for one color on one frame.
    Returns (n_inrange, gated_ok, coarse_reason)."""
    x1 = max(0, int(min(b[1] for b in boxes)))
    y1 = max(0, int(min(b[2] for b in boxes)))
    x2 = min(frame.shape[1], int(max(b[3] for b in boxes)))
    y2 = min(frame.shape[0], int(max(b[4] for b in boxes)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return (0, False, "tiny_box")
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for lo, hi in seglist:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    n_in = int(mask.sum())
    if n_in < 30:
        return (n_in, False, "COLOR_MISS")
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el)
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, el)
    cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gated = False
    areas = []
    for ct in cnts:
        x, y, w, h = cv2.boundingRect(ct)
        area = cv2.contourArea(ct)
        areas.append(int(area))
        if area < amin or area > amax or w <= 0 or h <= 0:
            continue
        aspect = w / h
        fill = area / (w * h)
        if aspect < asp_lo or aspect > asp_hi or fill < fill_lo or fill > fill_hi:
            continue
        gated = True
    if gated:
        return (n_in, True, "HIT_BUT_BEST_WRONG")
    if max(areas, default=0) < amin:
        return (n_in, False, "TOO_SMALL")
    if max(areas, default=0) > amax:
        return (n_in, False, "TOO_BIG")
    return (n_in, False, "SHAPE_MISS")


def overlap(blob, box):
    x, y, w, h, _ = blob
    bx1, by1, bx2, by2 = box[1:]
    cx, cy = x + w / 2, y + h / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def dump_frames(frame, yolo_boxes, cv_blobs, path, smaller=()):
    """Annotate a frame: YOLO truth in green, CV gated blobs in the class
    color, CV best (if passed as `smaller`) marked. Smaller-on-ball blobs
    are highlighted in white."""
    vis = frame.copy()
    for (_c, x1, y1, x2, y2) in yolo_boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    best = max(cv_blobs, key=lambda b: b[4]) if cv_blobs else None
    for (x, y, w, h, _a) in cv_blobs:
        col = (255, 0, 0) if best == (x, y, w, h, _a) else (0, 165, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
    for (x, y, w, h, _a) in smaller:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--source", default="",
                    help="REQUIRED: input video (matched by truth labels, "
                         "e.g. the match footage these labels came from)")
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--amin", type=int, default=900)
    ap.add_argument("--amax", type=int, default=12000)
    ap.add_argument("--aspect-lo", type=float, default=0.71)
    ap.add_argument("--aspect-hi", type=float, default=1.4)
    ap.add_argument("--fill-lo", type=float, default=0.5)
    ap.add_argument("--fill-hi", type=float, default=1.3)
    ap.add_argument("--score", choices=("biggest", "nearest", "roundest", "mixed"),
                    default="mixed",
                    help="how best-of-per-color picks among gated blobs")
    ap.add_argument("--fill-w", type=float, default=4.0)
    ap.add_argument("--edge-w", type=float, default=1.0)
    ap.add_argument("--dumpdir", type=str, default="",
                    help="save frames where YOLO has a ball but CV is empty")
    ap.add_argument("--fnlog", type=str, default="",
                    help="write per-FN reason lines (frame,color,n_in,reason)")
    ap.add_argument("--truth-json", type=str, default=DEFAULT_TRUTH,
                    help="cache/load YOLO truth to skip the model pass")
    a = ap.parse_args()

    if not a.source:
        raise SystemExit("eval_cv_vs_yolo.py: pass --source path/to/video.mp4 "
                         "(the footage that yolo_truth.json labels came from)")

    model = None
    if a.truth_json and Path(a.truth_json).exists():
        truth = get_truth(None, None, a.truth_json)
    else:
        model = YOLO(a.weights)
        print("running YOLO ground truth...")
        truth = get_truth(a.source, model, a.truth_json or None)
    print(f"truth frames: {len(truth)}")

    dumpdir = Path(a.dumpdir) if a.dumpdir else None
    if dumpdir is not None:
        dumpdir.mkdir(parents=True, exist_ok=True)
    fnlog = open(a.fnlog, "w") if a.fnlog else None
    if fnlog:
        fnlog.write("frame,color,n_inrange,reason\n")
    dumped = {c: 0 for c in COLORS}
    wrong_best = {c: 0 for c in COLORS}
    MAX_DUMP = 40

    cap = cv2.VideoCapture(a.source)
    n = 0
    scores = {p: {c: dict(tp=0, fn=0, fp=0, yolo_frames=0) for c in COLORS}
              for p in a.config}
    segs_of = {p: load_segments(p) for p in a.config}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lbl = truth.get(n, [])
        by_color = {}
        for c in COLORS:
            by_color[c] = [b for b in lbl if b[0] == c]
        for p in a.config:
            allb = cv_detect_all(frame, segs_of[p], a.amin, a.amax,
                                 a.aspect_lo, a.aspect_hi, a.fill_lo, a.fill_hi)
            det = {c: pick_best(allb[c], c, a.score,
                                fw=a.fill_w, ew=a.edge_w,
                                frame_size=frame.shape[1::-1])
                   for c in COLORS}
            for c in COLORS:
                s = scores[p][c]
                yolo_frames = len(by_color[c]) > 0
                if yolo_frames:
                    s["yolo_frames"] += 1
                blob = det[c]
                if blob is None:
                    if yolo_frames:
                        s["fn"] += 1
                        n_in, gated, reason = classify_miss(
                            frame, by_color[c], segs_of[p][c], a.amin, a.amax,
                            a.aspect_lo, a.aspect_hi, a.fill_lo, a.fill_hi)
                        if fnlog:
                            fnlog.write(f"{n},{c},{n_in},{reason}\n")
                        if dumpdir is not None and dumped[c] < MAX_DUMP:
                            dump_frames(frame, by_color[c], [],
                                        dumpdir / f"fn_{c}_{n:05d}.jpg")
                            dumped[c] += 1
                else:
                    hit = any(overlap(blob, b) for b in by_color[c])
                    if hit:
                        s["tp"] += 1
                    elif yolo_frames:
                        s["fp"] += 1
                        s["fn"] += 1
                        if fnlog:
                            _, _, r = classify_miss(frame, by_color[c],
                                                    segs_of[p][c], a.amin, a.amax,
                                                    a.aspect_lo, a.aspect_hi,
                                                    a.fill_lo, a.fill_hi)
                            fnlog.write(f"{n},{c},WB,{r}\n")
                        # was a smaller blob on the ball? -> best-of dropped it
                        smaller = [b for b in allb[c] if any(overlap(b, x)
                                                              for x in by_color[c])]
                        if (dumpdir is not None and wrong_best[c] < MAX_DUMP):
                            dump_frames(frame, by_color[c], allb[c],
                                        dumpdir / f"wrongbest_{c}_{n:05d}.jpg",
                                        smaller=smaller)
                            wrong_best[c] += 1
                    else:
                        s["fp"] += 1
        n += 1
    cap.release()
    if fnlog:
        fnlog.close()

    for p in a.config:
        print(f"\n{p}  (amin={a.amin}, amax={a.amax}, "
              f"aspect {a.aspect_lo}-{a.aspect_hi}, fill {a.fill_lo}-{a.fill_hi}, "
              f"score={a.score})")
        for c in COLORS:
            s = scores[p][c]
            prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
            rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
            print(f"  {c:8s} yolo_frames={s['yolo_frames']:4d}  "
                  f"tp={s['tp']:4d} fn={s['fn']:4d} fp={s['fp']:4d}  "
                  f"prec={prec*100:5.1f}%  rec={rec*100:5.1f}%")
    if str(dumpdir):
        print(f"\ndumped FN frames to {dumpdir}: {dumped}")


if __name__ == "__main__":
    main()