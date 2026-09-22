#!/usr/bin/env python3
"""
Real-time viewer for the deployed Limelight 3A detector: runs the
SSD-MobileNetV2 300x300 TFLite model (TFLite_Detection_PostProcess outputs)
live over a video file or webcam -- i.e. what the robot's 3A neural detector
will actually report. Nothing else runs: no YOLO, no CV/lab overlay.

Usage:
py yolo/scripts/detect_video_ssd_realtime.py [--source path/video.mp4]
                                                  [--weights yolo/weights/best_limelight3a_ssd_mobilenetv2_300x300.tflite]
                                                  [--conf 0.35]

Keys:
    q / Esc   quit
    p / Space pause
    f         toggle sphere filter (ball-shape/size suppression) on/off
    + / -     adjust the confidence threshold in 0.05 steps
    s         save current frame (name includes the timecode)
"""
import argparse
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorflow.lite as _tfl
    warnings.filterwarnings("ignore", message=r".*tf\.lite\.Interpreter.*")
    Interpreter = _tfl.Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

LABELS = ("yellow_pollen", "red_nectar", "blue_nectar")
COLORS = ((0, 255, 255), (0, 0, 255), (255, 0, 0))
CONF_STEP = 0.05

SCRIPTS = Path(__file__).resolve().parent
YOLO_DIR = SCRIPTS.parent
DEFAULT_WEIGHTS = YOLO_DIR / "weights" / "best_limelight3a_ssd_mobilenetv2_300x300.tflite"
DEFAULT_SOURCE = r"D:\ftc-yolo-synth\ftc-yolo-synth\reports\video_shots\testvid_fused_v7f_raw.mp4"


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


def ball_like(x1, y1, x2, y2, W, H):
    w, h = float(x2 - x1), float(y2 - y1)
    if w <= 0 or h <= 0:
        return False
    return (0.55 <= w / h <= 1.8) and (w * h <= 0.031 * W * H)


def output_slots(output_details):
    num = boxes = scores = classes = None
    for i, d in enumerate(output_details):
        shape = d["shape"]
        if len(shape) == 1:
            num = i
        elif len(shape) == 3 and shape[-1] == 4:
            boxes = i
        elif len(shape) == 2:
            if scores is None:
                scores = i
            else:
                classes = i
    if num is None or boxes is None or scores is None or classes is None:
        raise SystemExit("model outputs don't match a TFLite_Detection_PostProcess contract "
                         "(num[1], scores[N], class_ids[N], boxes[N,4])")
    return (output_details[num]["index"], output_details[boxes]["index"],
            output_details[scores]["index"], output_details[classes]["index"])


def detections(interpreter, slots, frame, conf, sphere_filter):
    num_idx, boxes_idx, scores_idx, classes_idx = slots
    input_detail = interpreter.get_input_details()[0]
    size = int(input_detail["shape"][1])
    image, scale, pad_x, pad_y = letterbox(frame, size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dtype = input_detail["dtype"]
    if dtype == np.uint8:
        tensor = rgb
    elif dtype == np.int8:
        in_scale, in_zero = input_detail["quantization"]
        tensor = np.round(rgb / in_scale + in_zero).clip(-128, 127).astype(np.int8)
    else:
        tensor = rgb.astype(np.float32) / 255.0
    interpreter.set_tensor(input_detail["index"], tensor[None])
    interpreter.invoke()

    count = int(np.asarray(interpreter.get_tensor(num_idx)).reshape(-1)[0])
    boxes_out = interpreter.get_tensor(boxes_idx)[0]
    scores_out = interpreter.get_tensor(scores_idx)[0]
    classes_out = interpreter.get_tensor(classes_idx)[0]

    found = []
    for i in range(min(count, len(scores_out))):
        score = float(scores_out[i])
        if score < conf:
            continue
        class_id = int(classes_out[i])
        ymin, xmin, ymax, xmax = map(float, boxes_out[i])
        x1 = (xmin * size - pad_x) / scale
        y1 = (ymin * size - pad_y) / scale
        x2 = (xmax * size - pad_x) / scale
        y2 = (ymax * size - pad_y) / scale
        found.append((class_id, score, x1, y1, x2, y2))

    if sphere_filter:
        frame_height, frame_width = frame.shape[:2]
        found = [d for d in found if ball_like(d[2], d[3], d[4], d[5], frame_width, frame_height)]
    return found


def draw_boxes(img, found):
    for class_id, score, x1, y1, x2, y2 in found:
        x1 = max(0, min(img.shape[1] - 1, round(x1)))
        y1 = max(0, min(img.shape[0] - 1, round(y1)))
        x2 = max(0, min(img.shape[1] - 1, round(x2)))
        y2 = max(0, min(img.shape[0] - 1, round(y2)))
        color = COLORS[class_id]
        label = f"{LABELS[class_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, max(28, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE if Path(DEFAULT_SOURCE).exists() else "0",
                    help="video file path or integer camera index (0 = webcam)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.35,
                    help="confidence threshold (3A guidance: start at 0.35-0.45)")
    ap.add_argument("--no-sphere", action="store_true", help="start with sphere filter off")
    ap.add_argument("--scale", type=float, default=0.75,
                    help="display downscale factor (fits big videos on screen)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N frames (smoke test)")
    a = ap.parse_args()

    if not Path(a.weights).exists():
        raise SystemExit(f"weights not found: {a.weights}")

    interpreter = Interpreter(model_path=str(a.weights), num_threads=os.cpu_count() or 8)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    slots = output_slots(interpreter.get_output_details())
    classes_shape = interpreter.get_output_details()
    classes_shape = [d for d in classes_shape if d["index"] == slots[3]][0]["shape"]
    print(f"model {Path(a.weights).name}: input {input_detail['shape'][1:3]} "
          f"{input_detail['dtype'].__name__}, "
          f"max detections {int(classes_shape[1])}")

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video/camera: {a.source}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sphere = not a.no_sphere
    conf = a.conf
    paused = False
    t0 = time.perf_counter()
    n = 0
    name = Path(a.weights).name
    window = f"SSD-MobileNetV2 3A [{name}]  q=quit p=pause f=sphere +/-=conf s=save"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('q'), 27):
                cap.release(); cv2.destroyAllWindows(); return
            if key in (ord('p'), ord(' ')):
                paused = False
            elif key == ord('f'):
                sphere = not sphere
            elif key in (ord('+'), ord('=')):
                conf = min(1.0, conf + CONF_STEP)
            elif key == ord('-'):
                conf = max(0.0, conf - CONF_STEP)
            continue

        ok, frame = cap.read()
        if not ok or (a.limit and n >= a.limit):
            break

        found = detections(interpreter, slots, frame, conf, sphere)
        ann = frame.copy()
        draw_boxes(ann, found)

        n += 1
        fps = n / (time.perf_counter() - t0)
        tsec = max(0.0, (n - 1) / vfps)
        tmm, tss = int(tsec // 60), tsec % 60
        counts = {b: 0 for b in ("yellow", "red", "blue")}
        for class_id, *_ in found:
            counts[LABELS[class_id].split("_")[0]] += 1
        h, w = ann.shape[:2]
        disp = cv2.resize(ann, (int(w * a.scale), int(h * a.scale)))
        hud = f"f{n:04d}/{n_frames}  t={tmm:02d}:{tss:05.2f}  " \
              f"y={counts['yellow']} r={counts['red']} b={counts['blue']}  {fps:.1f}fps"
        cv2.putText(disp, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"conf={conf:.2f}  sphere={'ON' if sphere else 'off'}",
                    (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('p'), ord(' ')):
            paused = True
        if key == ord('f'):
            sphere = not sphere
        if key in (ord('+'), ord('=')):
            conf = min(1.0, conf + CONF_STEP)
        if key == ord('-'):
            conf = max(0.0, conf - CONF_STEP)
        if key == ord('s'):
            out = Path("reports/video_shots")
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"ssd_shot_t{tmm:02d}{int(tss):02d}_{n_frames and n:05d}.jpg"
            cv2.imwrite(str(p), ann)
            print(f"saved {p}  (t={tmm:02d}:{tss:05.2f})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {n}/{n_frames if n_frames > 0 else '?'} frames, avg {fps:.1f} fps")


if __name__ == "__main__":
    main()