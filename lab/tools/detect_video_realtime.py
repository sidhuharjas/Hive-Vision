#!/usr/bin/env python3
"""
Real-time Lab detector viewer for a video file -- the Lab-track version of
scripts/detect_video_realtime.py. No model: it runs the shipped Lab config
(learned from the flagship YOLO) exactly like LabBallDetectorPipeline.java on
the Control Hub, so you can watch, on your PC, what the robot will see.

It also draws the flagship YOLO's ground-truth boxes for the dev video in
green when --truth-json is given, so you can see, frame by frame, where the
Lab detector agrees with the model and where it falls behind (deep-shadow
desaturation is a physics limit for any color-only detector).

By default the window shows ONLY the CIELAB view (press c to cycle
Lab -> RGB -> RGB+Lab):
  * left pane  = the L* (lightness) plane in grayscale,
  * right pane = the a*b* chromaticity map: every pixel's hue angle
    atan2(b*, a*) is shown as a color-wheel hue, its chroma as saturation,
    and L* as brightness. A yellow ball stays yellow in ANY lighting here;
    shadows just darken it. This is exactly the space the detector
    thresholds (its hue bands are the colored wedges of that map).

Usage:
    py hive-vision/lab/tools/detect_video_realtime.py [--source path/video.mp4]
                                                      [--config lab_tuned.json]
                                                      [--truth-json yolo_truth_lab.json]

Keys:
    q / Esc   quit
    p / Space pause
    c         cycle view: Lab -> RGB -> RGB+Lab
    f         toggle geometry gate (ball-shape/size suppression) on/off
    b         toggle "best-of per color" (the only thing an OpMode should read)
    a         toggle the ambient-L* adaptive chroma floor on/off
    s         save current frame (name includes the timecode)

Speed: detection is run at the processing resolution, which defaults to
"auto" = 960px wide max (Control Hub cameras actually feed 640x480-1280x720,
so this is the honest on-hub cost -- far below the YOLO track's CNN). Pass
--proc-scale 1.0 to force native-resolution detection.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import lab_lib
from lab_lib import COLORS

BASE = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE / "lab_tuned.json"
DEFAULT_TRUTH = Path(__file__).resolve().parent.parent / "docs" / "yolo_truth_lab.json"
DEFAULT_SOURCE = r"D:\ftc-yolo-synth\ftc-yolo-synth\reports\video_shots\testvid_fused_v7f_raw.mp4"

COLOR_BGR = {"yellow": (0, 255, 255), "red": (0, 0, 255), "blue": (255, 0, 0)}
TRUTH_BGR = (0, 200, 0)


def lab_view(feat):
    """CIELAB render from shared per-frame features: gray = L* plane,
    false-color = chromaticity map.

    The false-color pane maps every pixel of the frame into the a*b* plane
    one-to-one: hue angle atan2(b*, a*) -> color-wheel hue, chroma
    sqrt(a*^2+b*^2) -> saturation, L* -> brightness. So a "yellow" ball is
    yellow here in ANY light, and a mid-grey robot panel (a*=0,b*=0) shows
    as neutral, exactly the information the Lab detector thresholds.
    """
    L8 = feat["L"].astype(np.uint8)
    hsv = cv2.merge([(feat["theta"] * 0.5).astype(np.uint8),       # H 0..180
                     np.clip(feat["sat"] * 2.0, 0, 255).astype(np.uint8),  # S
                     L8])                                          # V = L*
    gray = cv2.cvtColor(L8, cv2.COLOR_GRAY2BGR)
    false = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return gray, false


def draw_boxes(img, found, truth, idx, truth_scale=1.0):
    for color in COLORS:
        col = COLOR_BGR[color]
        for (x, y, w, h, area) in found[color]:
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.circle(img, (x + w // 2, y + h // 2), 4, col, -1)
            cv2.putText(img, color, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if truth and idx in truth:
        for (c, x1, y1, x2, y2) in truth[idx]:
            cv2.rectangle(img, (int(x1 * truth_scale), int(y1 * truth_scale)),
                          (int(x2 * truth_scale), int(y2 * truth_scale)),
                          TRUTH_BGR, 1, cv2.LINE_AA)
            cv2.putText(img, c, (int(x1 * truth_scale), int(y2 * truth_scale) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TRUTH_BGR, 1)


def load_truth(path):
    """Cached YOLO truth keyed by frame index; None if unusable."""
    if not Path(path).exists():
        return None
    try:
        t = json.loads(Path(path).read_text())
    except Exception:
        return None
    t = {int(k): v for k, v in t.items()}
    if t:
        print(f"loaded {len(t)} truth frames from {path} "
              f"(drawn green; this is what the flagship YOLO saw)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE if Path(DEFAULT_SOURCE).exists() else "0",
                    help="video file path or integer camera index (0 = webcam)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--truth-json", default=str(DEFAULT_TRUTH),
                    help="YOLO ground-truth JSON to overlay as green boxes (empty = none)")
    ap.add_argument("--no-gate", action="store_true", help="start with gate off")
    ap.add_argument("--no-best", action="store_true", help="start with best-of off")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="start with the ambient-L* adaptive chroma floor off")
    ap.add_argument("--no-lab-view", action="store_true",
                    help="start in the plain RGB view (no CIELAB panes)")
    ap.add_argument("--proc-scale", type=float, default=-1.0,
                    help="processing resolution multiplier for detection; "
                         "default auto = 960px wide max (on-hub camera size)")
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

    truth = load_truth(a.truth_json)

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {a.source}")
    origW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    origH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ps = a.proc_scale if a.proc_scale > 0 else min(1.0, 960.0 / max(origW, 1))
    pW, pH = max(1, int(origW * ps)), max(1, int(origH * ps))
    print(f"source {origW}x{origH} -> detect/resolve at {pW}x{pH} "
          f"(proc_scale={ps:.2f}); --proc-scale 1.0 for native res")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    gate = not a.no_gate
    best = not a.no_best
    adaptive = not a.no_adaptive
    labmode = 1 if not a.no_lab_view else 0   # 0=RGB 1=Lab 2=both
    paused = False
    t0 = time.perf_counter()
    n = 0
    window = "lab realtime  q=quit p=pause c=view f=gate b=best a=adaptive s=save"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        if paused:
            while paused:
                key = cv2.waitKey(100) & 0xFF
                if key in (ord('q'), 27):
                    cap.release(); cv2.destroyAllWindows(); return
                if key in (ord('p'), ord(' ')):
                    paused = False
                elif key == ord('c'):
                    labmode = (labmode + 1) % 3
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
        frame = frame if ps == 1.0 else cv2.resize(frame, (pW, pH))

        feat = lab_lib.features(frame)          # computed ONCE, reused for everything
        am = cfg["ambient"]
        scl = lab_lib.adapt_scale(feat["amb"], am["ref_l"], am["scl_lo"], am["scl_hi"]) \
            if adaptive else 1.0
        found = lab_lib.detect(frame, cfg, gate=gate, adaptive=adaptive, feat=feat,
                               area_scale=ps * ps)
        if best:
            found = lab_lib.best_each(found, frame, cfg)

        ann = None
        if labmode in (0, 2):
            ann = frame.copy()
            draw_boxes(ann, found, truth, n, truth_scale=ps)
        gray = false = None
        if labmode in (1, 2):
            gray, false = lab_view(feat)
            draw_boxes(false, found, truth, n, truth_scale=ps)

        if labmode == 0:
            disp = ann
        elif labmode == 1:
            disp = np.hstack([gray, false])
        else:
            disp = np.hstack([ann, gray, false])
        del ann, gray, false

        fps = n / (time.perf_counter() - t0)
        tsec = max(0.0, (n - 1) / vfps)
        tmm, tss = int(tsec // 60), tsec % 60
        total = sum(len(f) for f in found.values())
        counts = {c: len(found[c]) for c in COLORS}
        h, w = disp.shape[:2]
        full = disp
        disp = cv2.resize(disp, (int(w * a.scale), int(h * a.scale)))
        hud = f"f{n:04d}/{n_frames}  t={tmm:02d}:{tss:05.2f}  " \
              f"y={counts['yellow']} r={counts['red']} b={counts['blue']}  {fps:.1f}fps"
        cv2.putText(disp, hud, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"view={'Lab' if labmode == 1 else 'RGB' if labmode == 0 else 'RGB+Lab'}  "
                          f"gate={'ON' if gate else 'off'} best={'ON' if best else 'off'} "
                          f"adapt={'ON' if adaptive else 'off'} blobs={total}",
                    (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if adaptive else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"ambL={feat['amb']:4.0f}  scl={scl:.2f}  "
                          f"ref_l={am['ref_l']:.0f}  detect@{pW}x{pH}",
                    (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('p'), ord(' ')):
            paused = True
        if key == ord('c'):
            labmode = (labmode + 1) % 3
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
            cv2.imwrite(str(p), full)
            print(f"saved {p}  (t={tmm:02d}:{tss:05.2f})")
        if a.limit and n >= a.limit:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {n}/{n_frames if n_frames > 0 else '?'} frames, avg {fps:.1f} fps")


if __name__ == "__main__":
    main()