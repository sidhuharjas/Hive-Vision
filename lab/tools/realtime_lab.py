#!/usr/bin/env python3
"""
Real-time Lab chromaticity viewer for a video file or webcam -- the middle
track's companion to realtime_cv.py. Same look and controls, but it runs the
Lab ball detector (hue-band + adaptive-by-ambient-L* chroma floor), i.e. what
the Control-Hub LabBallDetectorPipeline will see. No model is involved.

Usage:
    py hive-vision/lab/tools/realtime_lab.py [--source path/video.mp4]
                                             [--config lab_tuned.json]

Keys:
    q / Esc   quit
    p / Space pause
    f         toggle geometry gate on/off
    b         toggle "best blob per color" on/off
    a         toggle the ambient-L* adaptive chroma floor on/off
    s         save current annotated frame (name includes the timecode)

The numbers in lab_tuned.json were learned from the flagship YOLO model on real
match footage (fit_lab_from_yolo.py) and are the same constants baked into
LabBallDetectorPipeline.java.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

import lab_lib
from lab_lib import COLORS

BASE = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE / "lab_tuned.json"

COLOR_BGR = {"yellow": (0, 255, 255), "red": (0, 0, 255), "blue": (255, 0, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="video file path or integer camera index (0 = webcam)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-gate", action="store_true", help="start with gate off")
    ap.add_argument("--no-best", action="store_true", help="start with best-of off")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="start with the ambient-L* adaptive chroma floor off")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0, help="process at most N frames (smoke test)")
    a = ap.parse_args()

    if not Path(a.config).exists():
        raise SystemExit(f"config not found: {a.config}")
    cfg = lab_lib.load_config(a.config)
    print("loading Lab config:")
    for c in COLORS:
        m = cfg["colors"][c]
        print(f"  {c}: hue_band={m['hue_band']}  sat_floor={m['sat_floor']}  "
              f"l_floor={m['l_floor']}")
    print(f"  ambient ref_l={cfg['ambient']['ref_l']}  "
          f"scl=[{cfg['ambient']['scl_lo']},{cfg['ambient']['scl_hi']}]")

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {a.source}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    gate = not a.no_gate
    best = not a.no_best
    adaptive = not a.no_adaptive
    paused = False
    t0 = time.perf_counter()
    n = 0
    window = "realtime_lab  q=quit p=pause f=gate b=best a=adaptive s=save"
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
            elif key == ord('a'):
                adaptive = not adaptive
            continue
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        found = lab_lib.detect(frame, cfg, gate=gate, adaptive=adaptive)
        feat = lab_lib.features(frame)
        am = cfg["ambient"]
        scl = lab_lib.adapt_scale(feat["amb"], am["ref_l"], am["scl_lo"], am["scl_hi"]) \
            if adaptive else 1.0
        if best:
            found = lab_lib.best_each(found, frame, cfg)

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
                          f"adapt={'ON' if adaptive else 'off'} blobs={total}",
                    (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if adaptive else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"ambL={feat['amb']:4.0f}  scl={scl:.2f}  "
                          f"ref_l={am['ref_l']:.0f}",
                    (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1, cv2.LINE_AA)
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
        if key == ord('a'):
            adaptive = not adaptive
        if key == ord('s'):
            out = Path("reports/video_shots")
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"labshot_t{tmm:02d}{int(tss):02d}_{n_frames and n:05d}.jpg"
            cv2.imwrite(str(p), ann)
            print(f"saved {p}  (t={tmm:02d}:{tss:05.2f})")
        if a.limit and n >= a.limit:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {n}/{n_frames if n_frames > 0 else '?'} frames, avg {fps:.1f} fps")


if __name__ == "__main__":
    main()