#!/usr/bin/env python3
"""Run the Limelight 3A TFLite model over a video and write an annotated MP4."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

LABELS = ("yellow_pollen", "red_nectar", "blue_nectar")
COLORS = ((0, 255, 255), (0, 0, 255), (255, 0, 0))


def ball_like(x1, y1, x2, y2, frame_width, frame_height):
    box_width = float(x2 - x1)
    box_height = float(y2 - y1)
    if box_width <= 0 or box_height <= 0:
        return False
    aspect = box_width / box_height
    area_fraction = (box_width * box_height) / (frame_width * frame_height)
    return 0.55 <= aspect <= 1.8 and area_fraction <= 0.031


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


def detections(interpreter, frame, input_detail, output_detail, confidence, sphere_filter):
    size = int(input_detail["shape"][1])
    image, scale, pad_x, pad_y = letterbox(frame, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if input_detail["dtype"] == np.float32:
        tensor = image.astype(np.float32)
    else:
        input_scale, input_zero = input_detail["quantization"]
        info = np.iinfo(input_detail["dtype"])
        tensor = np.round(image / input_scale + input_zero).clip(info.min, info.max)
        tensor = tensor.astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], tensor[None])
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
    if output_detail["dtype"] != np.float32:
        output_scale, output_zero = output_detail["quantization"]
        output = (output - output_zero) * output_scale
    output = output[0]

    candidates = []
    for index in range(output.shape[1]):
        class_id = int(np.argmax(output[4:, index]))
        score = float(output[4 + class_id, index])
        if score < confidence:
            continue
        center_x, center_y, width, height = output[:4, index]
        x1 = (center_x - width / 2 - pad_x) / scale
        y1 = (center_y - height / 2 - pad_y) / scale
        x2 = (center_x + width / 2 - pad_x) / scale
        y2 = (center_y + height / 2 - pad_y) / scale
        candidates.append((class_id, score, x1, y1, x2, y2))

    boxes = []
    for class_id in range(len(LABELS)):
        class_candidates = [item for item in candidates if item[0] == class_id]
        if not class_candidates:
            continue
        nms_boxes = []
        scores = []
        for _, score, x1, y1, x2, y2 in class_candidates:
            nms_boxes.append([round(x1), round(y1), round(x2 - x1), round(y2 - y1)])
            scores.append(score)
        kept = cv2.dnn.NMSBoxes(nms_boxes, scores, confidence, 0.45)
        for kept_index in np.asarray(kept).reshape(-1):
            boxes.append(class_candidates[int(kept_index)])
    if sphere_filter:
        frame_height, frame_width = frame.shape[:2]
        boxes = [box for box in boxes if ball_like(box[2], box[3], box[4], box[5],
                                                    frame_width, frame_height)]
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="yolo/weights/best_limelight3a_float32.tflite")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-sphere", action="store_true",
                        help="keep oversized or non-ball-shaped model boxes")
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.weights)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok or (args.max_frames and frame_number >= args.max_frames):
            break
        found = detections(interpreter, frame, input_detail, output_detail,
                   args.confidence, not args.no_sphere)
        for class_id, score, x1, y1, x2, y2 in found:
            x1 = max(0, min(width - 1, round(x1)))
            y1 = max(0, min(height - 1, round(y1)))
            x2 = max(0, min(width - 1, round(x2)))
            y2 = max(0, min(height - 1, round(y2)))
            color = COLORS[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{LABELS[class_id]} {score:.2f}", (x1, max(28, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TFLite Limelight 3A | frame {frame_number} | detections {len(found)}",
                    (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"processed {frame_number} frames")

    capture.release()
    writer.release()
    print(f"wrote {args.output} ({frame_number} frames)")


if __name__ == "__main__":
    main()
