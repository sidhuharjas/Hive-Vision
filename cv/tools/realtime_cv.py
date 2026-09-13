#!/usr/bin/env python3
"""
Real-time OpenCV (color thresholding) viewer for a video file -- same look and
controls as scripts/detect_video_realtime.py but with NO model: it runs pure
HSV thresholding, so it measures what a Control-Hub color detector will see.

Usage:
    py OpenCV-Ball-Detector/tools/realtime_cv.py [--source path/video.mp4]
                                                 [--config hsv_tuned.json]

Keys:
    q / Esc   quit
    p / Space pause
    f         toggle geometry gate (ball-shape/size suppression) on/off
    b         toggle "best blob per color" (the only one an OpMode should use)
    s         save current annotated frame (name includes the timecode)

Defaults are the result of learning from the real match video: the HSV
ranges in hsv_tuned.json were mined from YOLO-confirmed ball pixels
(fit_hsv_from_yolo.py), gated to a real-ball size band (900..12000 px),
and the per-color pick scores closest-to-typical-ball-area + roundness +
edge penalty. Eval vs YOLO ground truth: yellow rec 69% / red rec 80% /
blue rec 82%. Re-tune live on your field camera whenever the lighting
changes:
    py OpenCV-Ball-Detector/tools/hsv_tuner.py path/to/frame.jpg
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE / "hsv_tuned.json"
MIN_AREA_PX = 900      # kills speckle patches (real far-corner balls are >= this)
MAX_AREA_PX = 12000    # kills big robot panels (real balls p95 ~ 8..14k)
MIN_ASPECT = 0.71      # balls are near-square
MAX_ASPECT = 1.4
MIN_FILL = 0.5
MAX_FILL = 1.3
EXPECT_AREA = {"yellow": 3500.0, "red": 4200.0, "blue": 4200.0}  # real-ball medians
FILL_W = 4.0           # best-of score: roundness weight
EDGE_W = 1.0           # best-of score: edge-touching penalty (robot panels)

COLOR_BGR = {"yellow": (0, 255, 255), "red": (0, 0, 255), "blue": (255, 0, 0)}
COLORS = ("yellow", "red", "blue")


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    segs = {}
    for color in COLORS:
        segs[color] = [(np.array(s[0], dtype=np.int32),
                        np.array(s[1], dtype=np.int32))
                       for s in cfg[color]["segments"]]
        print(f"  {color}: {len(segs[color])} HSV segment(s)")
    return segs


def detect(frame, segs, gate_on):
    """Return per-color list of (x, y, w, h, area) blobs, optionally gated."""
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
            if area < MIN_AREA_PX:
                continue
            if gate_on:
                if area > MAX_AREA_PX or w <= 0 or h <= 0:
                    continue
                aspect = w / h
                fill = area / (w * h)
                if aspect < MIN_ASPECT or aspect > MAX_ASPECT or \
                   fill < MIN_FILL or fill > MAX_FILL:
                    continue
            blobs.append((x, y, w, h, area))
        found[color] = blobs
    return found


def best_each(found, frame):
    """Pick the most ball-y gated blob per color -- nearest to the typical
    real-ball area, roundest, and penalized if it touches the frame edge
    (robot panels hug the borders). This is what an OpMode should trust."""
    W, H = frame.shape[1], frame.shape[0]
    out = {}
    for c in COLORS:
        blobs = found[c]
        if not blobs:
            out[c] = []
            continue
        def cost(b):
            x, y, w, h, a = b
            edge = 0.0
            if EDGE_W:
                if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                    edge = EDGE_W
            fill = a / (w * h)
            return abs(a / EXPECT_AREA[c] - 1.0) + FILL_W * (1.0 - fill) + edge
        out[c] = [min(blobs, key=cost)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="video file path or integer camera index (0 = webcam)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-gate", action="store_true", help="start with gate off")
    ap.add_argument("--no-best", action="store_true", help="start with best-of off")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0, help="process at most N frames (smoke test)")
    a = ap.parse_args()

    if not Path(a.config).exists():
        raise SystemExit(f"config not found: {a.config}")
    print("loading HSV segments:")
    segs = load_config(a.config)
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {a.source}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    gate = not a.no_gate
    best = not a.no_best
    paused = False
    t0 = time.perf_counter()
    n = 0
    window = "realtime_cv  q=quit p=pause f=gate b=best s=save"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        if paused:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('p'), ord(' ')):
                paused = False
            elif key == ord('f'):
                gate = not gate
            elif key == ord('b'):
                best = not best
            continue
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        found = detect(frame, segs, gate)
        if best:
            found = best_each(found, frame)
        ann = frame.copy()
        for color in COLORS:
            for (x, y, w, h, area) in found[color]:
                col = COLOR_BGR[color]
                cv2.rectangle(ann, (x, y), (x + w, y + h), col, 2)
                cv2.circle(ann, (x + w // 2, y + h // 2), 3, col, -1)
        fps = n / (time.perf_counter() - t0)
        tsec = max(0.0, (n - 1) / vfps)
        tmm, tss = int(tsec // 60), tsec % 60
        total = sum(len(f) for f in found.values())
        counts = {c: len(found[c]) for c in COLORS}
        h, w = ann.shape[:2]
        disp = cv2.resize(ann, (int(w * a.scale), int(h * a.scale)))
        hud = f"f{n:04d}/{n_frames}  t={tmm:02d}:{tss:05.2f}  " \
              f"y={counts['yellow']} r={counts['red']} b={counts['blue']}  {fps:.1f}fps"
        cv2.putText(disp, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"gate={'ON' if gate else 'off'} best={'ON' if best else 'off'} "
                          f"blobs={total}",
                    (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if gate else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('p'), ord(' ')):
            paused = True
        if key == ord('f'):
            gate = not gate
        if key == ord('b'):
            best = not best
        if key == ord('s'):
            out = Path("reports/video_shots")
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"cvshot_t{tmm:02d}{int(tss):02d}_{n_frames and n:05d}.jpg"
            cv2.imwrite(str(p), ann)
            print(f"saved {p}  (t={tmm:02d}:{tss:05.2f})")
        if a.limit and n >= a.limit:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {n}/{n_frames if n_frames > 0 else '?'} frames, avg {fps:.1f} fps")


if __name__ == "__main__":
    main()