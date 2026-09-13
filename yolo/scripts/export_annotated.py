#!/usr/bin/env python3
"""Export annotated video (+ optional timestamp + sphere filter) to MP4."""
import argparse
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=r"weights/best.pt")
    ap.add_argument("--source", default="0",
                    help="video file path or integer camera index (0 = webcam)")
    ap.add_argument("--out", default=r"annotated.mp4")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--no-sphere", action="store_true")
    ap.add_argument("--no-timestamp", action="store_true")
    a = ap.parse_args()

    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def ball_like(x1, y1, x2, y2, W, H):
        w, h = float(x2 - x1), float(y2 - y1)
        ar = w / h
        return (0.55 <= ar <= 1.8) and (w * h <= 0.031 * W * H)

    n = 0
    for res in model.predict(source=a.source, stream=True, imgsz=a.imgsz,
                             conf=a.conf, iou=a.iou, verbose=False, half=True):
        if not a.no_sphere:
            w0, h0 = res.orig_shape[1], res.orig_shape[0]
            keep = [ball_like(b.xyxy[0][0], b.xyxy[0][1], b.xyxy[0][2], b.xyxy[0][3], w0, h0)
                    for b in res.boxes]
            if not all(keep):
                res.boxes = res.boxes[torch.tensor(keep, dtype=torch.bool)]
        ann = res.plot()
        if not a.no_timestamp:
            tsec = n / fps
            tmm, tss = int(tsec // 60), tsec % 60
            cv2.putText(ann, f"t={tmm:02d}:{tss:05.2f}", (16, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3, cv2.LINE_AA)
        vw.write(ann)
        n += 1
        if n % 200 == 0:
            print(f"   {n}/{n_frames}")
    vw.release()
    cap.release()
    print(f"wrote {a.out} ({n} frames, conf={a.conf} imgsz={a.imgsz} "
          f"sphere={not a.no_sphere})")


if __name__ == "__main__":
    main()