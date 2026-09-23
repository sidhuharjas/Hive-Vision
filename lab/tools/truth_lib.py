# -*- coding: utf-8 -*-
"""YOLO ground-truth generation shared by the Lab track tooling.

Reuses the flagship Hive Vision model (neural-net/weights/best.pt) as the label
source: every Control Hub metric in this repo is measured against it, so the
Lab track stays comparable to the CV track.
"""
import json
from pathlib import Path

from ultralytics import YOLO

MIN_CONF = 0.35
DEFAULT_IMGSZ = 960

WEIGHTS = str(Path(__file__).resolve().parent.parent.parent / "yolo" / "weights" / "best.pt")


def ball_like(x1, y1, x2, y2, W, H):
    """Drop giant or elongated boxes (robot panels), keep ball-shaped ones."""
    w, h = float(x2 - x1), float(y2 - y1)
    return (0.55 <= w / h <= 1.8) and (w * h <= 0.031 * W * H)


def _labels(source, weights, conf, imgsz):
    model = YOLO(weights)
    truth = {}
    fidx = 0
    for result in model.predict(source=source, stream=True, imgsz=imgsz,
                                conf=conf, verbose=False, half=True):
        w0, h0 = result.orig_shape[1], result.orig_shape[0]
        names = result.names
        lbl = []
        for b in result.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            if not ball_like(x1, y1, x2, y2, w0, h0):
                continue
            c = names[int(b.cls[0])]
            if c in ("yellow", "red", "blue"):
                lbl.append((c, x1, y1, x2, y2))
        if lbl:
            truth[fidx] = lbl
        fidx += 1
    return truth


def build_truth(source, cache_path, weights=WEIGHTS, conf=MIN_CONF, imgsz=DEFAULT_IMGSZ):
    """Return {frame: [(color, x1, y1, x2, y2), ...]} from the flagship model,
    caching to cache_path so repeated tool runs skip the model pass."""
    truth = {}
    if cache_path and Path(cache_path).exists():
        raw = json.loads(Path(cache_path).read_text())
        truth = {int(k): v for k, v in raw.items()}
        print(f"loaded {len(truth)} truth frames from {cache_path}")
        return truth
    print(f"running flagship YOLO ground truth over {source} ...")
    truth = _labels(source, weights, conf, imgsz)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({str(k): v for k, v in truth.items()},
                                               indent=0))
        print(f"cached {len(truth)} truth frames -> {cache_path}")
    return truth