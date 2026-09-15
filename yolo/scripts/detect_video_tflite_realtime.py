#!/usr/bin/env python3
"""
Real-time detection viewer that runs the Limelight 3A TFLite model live.

This mirrors detect_video_realtime.py (ultralytics / best.pt) but executes the
exact integer-quantized TFLite artifact the Limelight 3A runs, so what you see
on screen is what the co-processor will produce.

Usage:
    py scripts/detect_video_tflite_realtime.py [--source path/video.mp4]
                                               [--weights yolo/weights/best_limelight3a_int8.tflite]
                                               [--conf 0.25] [--iou 0.45]

Keys:
    q / Esc   quit
    p / Space pause
    f         toggle sphere filter (ball-shape/size suppression) on/off
    s         save current annotated frame (name includes the timecode)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

DEFAULT_W = r"yolo/weights/best_limelight3a_float32.tflite"

LABELS = ("yellow_pollen", "red_nectar", "blue_nectar")
COLORS = ((0, 255, 255), (0, 0, 255), (255, 0, 0))


def letterbox(frame, size):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def nms(boxes, iou_thr):
    if not boxes:
        return []
    array = np.array(boxes, dtype=np.float64)
    order = np.argsort(-array[:, 1])
    keep = []
    while order.size:
        i = order[0]
        b = array[i]
        keep.append((int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])))
        rest = order[1:]
        if rest.size == 0:
            break
        ix1, iy1, ix2, iy2 = array[i][2:6]
        jx1 = array[rest][:, 2]
        jy1 = array[rest][:, 3]
        jx2 = array[rest][:, 4]
        jy2 = array[rest][:, 5]
        inter = (np.minimum(ix2, jx2) - np.maximum(ix1, jx1)).clip(0) * \
                (np.minimum(iy2, jy2) - np.maximum(iy1, jy1)).clip(0)
        area_i = (ix2 - ix1) * (iy2 - iy1)
        area_j = (jx2 - jx1) * (jy2 - jy1)
        union = area_i + area_j - inter
        order = rest[inter / np.maximum(union, 1e-9) <= iou_thr]
    return keep


def ball_like(x1, y1, x2, y2, W, H):
    w, h = float(x2 - x1), float(y2 - y1)
    return (0.55 <= w / h <= 1.8) and (w * h <= 0.031 * W * H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--source", default="0",
                    help="video file path or integer camera index (0 = webcam)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--scale", type=float, default=0.75,
                    help="display downscale factor (fits big videos on screen)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    if not Path(a.weights).exists():
        raise SystemExit(f"weights not found: {a.weights}")

    interpreter = Interpreter(model_path=str(a.weights))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    size = int(input_detail["shape"][1])

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video/camera: {a.source}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sphere = True
    paused = False
    t0 = time.perf_counter()
    n = 0
    name = Path(a.weights).name
    window = f"TFLite limelight [{name}]  q=quit p=pause f=sphere s=save"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('q'), 27):
                cap.release(); cv2.destroyAllWindows(); return
            if key in (ord('p'), ord(' '), ord('f')):
                paused = False
                if key == ord('f'):
                    sphere = not sphere
            continue

        ok, frame = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break

        image, scale, pad_x, pad_y = letterbox(frame, size)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        in_scale, in_zero = input_detail["quantization"]
        if in_scale:
            tensor = np.round(rgb / in_scale + in_zero).clip(-128, 127).astype(np.int8)
        else:
            tensor = rgb.astype(np.float32)
        interpreter.set_tensor(input_detail["index"], tensor[None])
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        out_scale, out_zero = output_detail["quantization"]
        if out_scale:
            output = (output - out_zero) * out_scale
        output = output[0]

        raw = []
        for index in range(output.shape[1]):
            class_id = int(np.argmax(output[4:, index]))
            score = float(output[4 + class_id, index])
            if score < a.conf:
                continue
            cx, cy, w, h = output[:4, index]
            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale
            raw.append((class_id, score, x1, y1, x2, y2))

        w0, h0 = frame.shape[1], frame.shape[0]
        boxes = nms(raw, a.iou)
        if sphere:
            boxes = [b for b in boxes if ball_like(b[2], b[3], b[4], b[5], w0, h0)]

        ann = frame.copy()
        counts = {"yellow": 0, "red": 0, "blue": 0}
        for class_id, score, x1, y1, x2, y2 in boxes:
            x1 = max(0, min(w0 - 1, round(x1)))
            y1 = max(0, min(h0 - 1, round(y1)))
            x2 = max(0, min(w0 - 1, round(x2)))
            y2 = max(0, min(h0 - 1, round(y2)))
            color = COLORS[class_id]
            counts[LABELS[class_id].split("_")[0]] += 1
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
            cv2.putText(ann, f"{LABELS[class_id]} {score:.2f}", (x1, max(28, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        n += 1
        fps = n / (time.perf_counter() - t0)
        tsec = max(0.0, (n - 1) / vfps)
        tmm, tss = int(tsec // 60), tsec % 60
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
            p = out / f"tflite_shot_t{tmm:02d}{int(tss):02d}_{n_frames and n:05d}.jpg"
            cv2.imwrite(str(p), ann)
            print(f"saved {p}  (t={tmm:02d}:{tss:05.2f})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {n}/{n_frames if n_frames > 0 else '?'} frames, avg {fps:.1f} fps")


if __name__ == "__main__":
    main()