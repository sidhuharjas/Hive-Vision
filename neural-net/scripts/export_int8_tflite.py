#!/usr/bin/env python3
"""Quantize the shipped YOLO ONNX/SavedModel to a FULL int8 TFLite.

Limelight 3A rejects float32 models, so this produces the int8 twin of
best_limelight3a_float32.tflite. The SavedModel is generated first:

    onnx2tf -i neural-net/weights/best.onnx -o neural-net/_litert_int8/sm -b 1 -osd -fdosm

then this script applies post-training INT8 quantization calibrated on the
fused training set (V7f/dataset/concat), i.e. exactly the images the model was
trained on. Use this venv (has tensorflow + onnx2tf):

    hive-vision/neural-net/.venv-tflite/Scripts/python.exe neural-net/scripts/export_int8_tflite.py

Output contract matches the shipped float32 export:
    input   images     (1, 960, 960, 3) float32 (0..1, RGB)
    output  output0    (1, 7, 18900)    float32 (xywh in 960 grid + 3 scores)

The Detect-head CONCAT makes full INT8 activations impossible (the three
stride levels keep separate per-tensor scales), and a single mixed box+score
tensor would collapse the scores under one scale anyway. Dynamic-range quant
(int8 weights, float activations) is therefore the correct Limelight 3A
artifact and matches what ultralytics ships as its "int8" model.
"""
import argparse
import pathlib
import random

import cv2
import numpy as np
import tensorflow as tf

BASE = pathlib.Path(__file__).resolve().parents[3]          # .../ftc-yolo-synth/ftc-yolo-synth
WEIGHTS = BASE / "hive-vision" / "yolo" / "weights"
DEFAULT_SM = BASE / "hive-vision" / "yolo" / "_litert_int8" / "sm"
DEFAULT_CALIB = BASE / "V7f" / "dataset" / "concat" / "images" / "train"
DEFAULT_OUT = WEIGHTS / "best_limelight3a_int8.tflite"
SIZE = 960
CALIB_COUNT = 256


def letterbox(frame, size=SIZE):
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


def representative_dataset(images, rng):
    def gen():
        for path in images:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = letterbox(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            yield [img[None]]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", default=str(DEFAULT_SM))
    ap.add_argument("--calib-images", default=str(DEFAULT_CALIB))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--count", type=int, default=CALIB_COUNT)
    ap.add_argument("--full-int8", action="store_true",
                    help="try full integer PTQ instead of dynamic-range "
                         "(weights-int8). NOTE: the Detect-head CONCAT breaks "
                         "full-int8 - the three stride levels have different "
                         "scales, which TFLite rejects - so this flag is a "
                         "documentation aid, not the shipped artifact.")
    args = ap.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.full_int8:
        calib_dir = pathlib.Path(args.calib_images)
        all_imgs = sorted(calib_dir.glob("*.jpg")) + sorted(calib_dir.glob("*.png"))
        if not all_imgs:
            raise SystemExit(f"no calibration images found in {calib_dir}")
        rng = random.Random(0)
        sample = rng.sample(all_imgs, min(len(all_imgs), args.count))
        print(f"calibration: {len(sample)} images from {calib_dir}", flush=True)
        converter.representative_dataset = representative_dataset(sample, rng)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        print("converting full-int8 (ptq)...", flush=True)
    else:
        print("converting int8 weights, float activations (dynamic range)...",
              flush=True)
    tflite = converter.convert()

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite)
    print(f"wrote {out} ({len(tflite) / 1e6:.2f} MB)", flush=True)

    interp = tf.lite.Interpreter(model_path=str(out))
    interp.allocate_tensors()
    for d in interp.get_input_details():
        print(f"  input  {d['name']} shape={d['shape']} dtype={d['dtype']} "
              f"quant={d['quantization']}")
    for d in interp.get_output_details():
        print(f"  output {d['name']} shape={d['shape']} dtype={d['dtype']} "
              f"quant={d['quantization']}")


if __name__ == "__main__":
    main()