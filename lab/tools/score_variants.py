import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '.')
import lab_lib
from lab_lib import COLORS

BASE = Path(__file__).resolve().parent
SRC = 'D:/ftc-yolo-synth/ftc-yolo-synth/reports/video_shots/testvid_fused_v7f_raw.mp4'
TRUTH = BASE / ".." / "docs" / "yolo_truth_lab.json"


def overlap(b, box):
    x, y, w, h, _ = b
    bx1, by1, bx2, by2 = box[1:]
    cx, cy = x + w / 2, y + h / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def combos_for(c):
    if c == 'yellow':
        bands = [[22.3, 112.0], [35.0, 112.0], [42.0, 112.0], [22.3, 105.0]]
        fls = [1.0, 1.2]
    elif c == 'red':
        bands = [[329.3, 62.0], [325.0, 62.0], [335.0, 62.0], [329.3, 58.0]]
        fls = [1.5, 1.7]
    else:
        bands = [[258.0, 326.0], [262.0, 322.0], [250.0, 326.0], [258.0, 330.0]]
        fls = [1.0, 1.2]
    return [(b, fl) for b in bands for fl in fls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fill', type=float, default=None)
    args = ap.parse_args()
    cfg = lab_lib.load_config(str(BASE / 'lab_tuned.json'))
    truth = {int(k): v for k, v in json.load(open(TRUTH)).items()}

    col_cfg = {c: dict(cfg['colors'][c]) for c in COLORS}
    variants = {c: combos_for(c) for c in COLORS}

    scores = {c: {i: dict(tp=0, fn=0, fp=0, dark_fp_t=0, dark_tp=0, dark_fn=0,
                          dark_frames=0, frames=0)
                  for i in range(len(variants[c]))}
              for c in COLORS}
    order = {c: {tuple(map(float, b)) + (fl,): i for i, (b, fl) in enumerate(variants[c])}
             for c in COLORS}

    cap = cv2.VideoCapture(SRC)
    box_means = {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % args.stride != 0:
            n += 1
            continue
        lbl = truth.get(n, [])
        feat = lab_lib.features(frame)
        amcf = cfg['ambient']
        scl = lab_lib.adapt_scale(feat['amb'], amcf['ref_l'], amcf['scl_lo'], amcf['scl_hi'])
        by = {c: [b for b in lbl if b[0] == c] for c in COLORS}
        Ls = feat['L'][::4, ::4]
        for c in COLORS:
            if by[c]:
                vals = []
                for (_c, x1, y1, x2, y2) in by[c]:
                    x0, y0 = max(0, int(x1 // 4)), max(0, int(y1 // 4))
                    x1i, y1i = min(Ls.shape[1], int(x2 // 4)), min(Ls.shape[0], int(y2 // 4))
                    if x1i > x0 and y1i > y0:
                        vals.append(float(Ls[y0:y1i, x0:x1i].mean()))
                box_means[c] = min(vals) if vals else 120.0
            for i, (band, fl) in enumerate(variants[c]):
                col_cfg[c] = {'hue_band': band,
                              'sat_floor': cfg['colors'][c]['sat_floor'] * fl,
                              'l_floor': cfg['colors'][c]['l_floor']}
                mask = lab_lib.mask_for(feat, col_cfg[c], scl)
                el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                if args.fill is None:
                    gate_cfg = cfg['gate']
                else:
                    gate_cfg = dict(cfg['gate'], min_fill=args.fill)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
                blobs = lab_lib.blobs_from_mask(mask, gate_cfg)
                det = None
                if blobs:
                    W, H = frame.shape[1], frame.shape[0]
                    exp = cfg['expect_area'][c]
                    bw = cfg['best']

                    def cost(b):
                        x, y, w, h, a = b
                        edge = bw['edge_w'] if (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1) else 0.0
                        fill = a / (w * h)
                        return abs(a / exp - 1.0) + bw['fill_w'] * (1.0 - fill) + edge
                    det = min(blobs, key=cost)
                s = scores[c][i]
                s['frames'] += 1
                if by[c]:
                    s['frames'] += 0
                    dark = box_means[c] < 40.0
                    if dark:
                        s['dark_frames'] += 1
                    if det and any(overlap(det, b) for b in by[c]):
                        s['tp'] += 1
                        if dark:
                            s['dark_tp'] += 1
                    else:
                        s['fn'] += 1
                        if dark:
                            s['dark_fn'] += 1
                elif det:
                    s['fp'] += 1
        n += 1
        if n % 200 == 0:
            print(f'frame {n}', flush=True)
    cap.release()

    for c in COLORS:
        print('\n==== %s  (base band=%s floor=%.2f)' % (c, cfg['colors'][c]['hue_band'],
                                                        cfg['colors'][c]['sat_floor']))
        rows = []
        for i, (band, fl) in enumerate(variants[c]):
            s = scores[c][i]
            prec = s['tp'] / (s['tp'] + s['fp']) if s['tp'] + s['fp'] else 0.
            rec = s['tp'] / (s['tp'] + s['fn']) if s['tp'] + s['fn'] else 0.
            drec = s['dark_tp'] / (s['dark_tp'] + s['dark_fn']) if s['dark_tp'] + s['dark_fn'] else 0.
            rows.append((prec * 100, rec * 100, drec * 100, s['tp'], s['fn'], s['fp'],
                         band, fl, s['dark_frames']))
        rows.sort(key=lambda r: (-r[0], -r[1]))
        for prec, rec, drec, tp, fn, fp, band, fl, df in rows:
            print('  prec=%5.1f rec=%5.1f drec=%5.1f tp=%3d fn=%3d fp=%4d dark=%2d  fl=%.1f band=%s'
                  % (prec, rec, drec, tp, fn, fp, df, fl, band))


if __name__ == '__main__':
    main()