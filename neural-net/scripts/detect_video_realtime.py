#!/usr/bin/env python3
"""
Real-time YOLO detection viewer for a video file.

Usage:
    py scripts/detect_video_realtime.py [--weights weights/best.pt]
                                        [--source path/video.mp4]
                                        [--conf 0.25] [--imgsz 960]

Keys:
    q / Esc   quit
    p / Space pause
    f         toggle sphere filter (ball-shape/size suppression) on/off
    s         save current annotated frame (name includes the timecode)

The sphere filter drops giant robot-side panels (ar outside 0.55..1.8 or area
> 3.1% of frame) -- real balls max ~215px here, so nothing true is removed.
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

DEFAULT_W = r"weights/best.pt"


def ball_like(x1, y1, x2, y2, W, H):
    w, h = float(x2 - x1), float(y2 - y1)
    return (0.55 <= w / h <= 1.8) and (w * h <= 0.031 * W * H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--source", default="0",
                    help="video file path or integer camera index (0 = webcam)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="higher imgsz recovers tiny far-corner balls (was 640)")
    ap.add_argument("--no-sphere", action="store_true",
                    help="start with the sphere filter off")
    ap.add_argument("--scale", type=float, default=0.75,
                    help="display downscale factor (fits big videos on screen)")
    a = ap.parse_args()

    if not Path(a.weights).exists():
        raise SystemExit(f"weights not found: {a.weights}")
    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {a.source}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sphere = not a.no_sphere
    paused = False
    t0 = time.perf_counter()
    n = 0
    name = Path(a.weights).name
    window = f"fused v7f [{name}]  q=quit p=pause f=sphere s=save"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for result in model.predict(source=a.source, stream=True, imgsz=a.imgsz,
                                conf=a.conf, iou=a.iou, verbose=False, half=True):
        if paused:
            while paused:
                key = cv2.waitKey(100) & 0xFF
                if key in (ord('q'), 27):
                    cap.release(); cv2.destroyAllWindows(); return
                if key in (ord('p'), ord(' '), ord('f')):
                    if key in (ord('p'), ord(' ')):
                        paused = False
                    elif key == ord('f'):
                        sphere = not sphere
        if sphere:
            w0, h0 = result.orig_shape[1], result.orig_shape[0]
            keep = [ball_like(b.xyxy[0][0], b.xyxy[0][1], b.xyxy[0][2], b.xyxy[0][3], w0, h0)
                    for b in result.boxes]
            if not all(keep):
                result.boxes = result.boxes[torch.tensor(keep, dtype=torch.bool)]
        ann = result.plot()
        n += 1
        fps = n / (time.perf_counter() - t0)
        tsec = max(0.0, (n - 1) / vfps)
        tmm, tss = int(tsec // 60), tsec % 60
        counts = {"yellow": 0, "red": 0, "blue": 0}
        for b in result.boxes:
            counts[result.names[int(b.cls[0])]] += 1
        h, w = ann.shape[:2]
        disp = cv2.resize(ann, (int(w * a.scale), int(h * a.scale)))
        hud = f"f{n:04d}/{n_frames}  t={tmm:02d}:{tss:05.2f}  " \
              f"y={counts['yellow']} r={counts['red']} b={counts['blue']}  {fps:.1f}fps"
        cv2.putText(disp, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"sphere={'ON' if sphere else 'off'}",
                    (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if sphere else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('p'), ord(' ')):
            paused = True
        if key == ord('f'):
            sphere = not sphere
        if key == ord('s'):
            out = Path("reports/video_shots")
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"shot_t{tmm:02d}{int(tss):02d}_{n_frames and n:05d}.jpg"
            cv2.imwrite(str(p), ann)
            print(f"saved {p}  (t={tmm:02d}:{tss:05.2f})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {n}/{n_frames if n_frames > 0 else '?'} frames, avg {fps:.1f} fps")


if __name__ == "__main__":
    main()