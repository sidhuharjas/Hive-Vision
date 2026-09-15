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
WIN = {  # band + floor mult (from score_variants rounds 1-2)
    'yellow': ([22.3, 105.0], 1.0),
    'red': ([335.0, 62.0], 1.5),
    'blue': ([258.0, 326.0], 1.0),
}


def overlap(b, box):
    x, y, w, h, _ = b
    bx1, by1, bx2, by2 = box[1:]
    cx, cy = x + w / 2, y + h / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def main():
    cfg = lab_lib.load_config(str(BASE / 'lab_tuned.json'))
    truth = {int(k): v for k, v in json.load(open(TRUTH)).items()}
    combos = [(slo, fill) for slo in [0.25, 0.35, 0.45] for fill in [0.5, 0.55, 0.62]]
    sc = []
    for _i, (slo, fill) in enumerate(combos):
        sc.append({c: dict(tp=0, fn=0, fp=0, dtp=0, dfn=0, df=0) for c in COLORS})

    col = {c: dict(cfg['colors'][c],
                   sat_floor=cfg['colors'][c]['sat_floor'] * WIN[c][1],
                   hue_band=WIN[c][0])
           for c in COLORS}
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(SRC)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lbl = truth.get(n, [])
        feat = lab_lib.features(frame)
        Ls = feat['L'][::4, ::4]
        dk = {}
        for c in COLORS:
            if [b for b in lbl if b[0] == c]:
                vals = []
                for (_c, x1, y1, x2, y2) in [b for b in lbl if b[0] == c]:
                    x0, y0 = max(0, int(x1 // 4)), max(0, int(y1 // 4))
                    x1i, y1i = min(Ls.shape[1], int(x2 // 4)), min(Ls.shape[0], int(y2 // 4))
                    if x1i > x0 and y1i > y0:
                        vals.append(float(Ls[y0:y1i, x0:x1i].mean()))
                dk[c] = min(vals) < 40.0 if vals else False
        for i, (slo, fill) in enumerate(combos):
            s = sc[i]
            gate = dict(cfg['gate'], min_fill=fill)
            for c in COLORS:
                scl = lab_lib.adapt_scale(feat['amb'], cfg['ambient']['ref_l'], slo,
                                          cfg['ambient']['scl_hi'])
                mask = lab_lib.mask_for(feat, col[c], scl)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
                blobs = lab_lib.blobs_from_mask(mask, gate)
                det = None
                if blobs:
                    W, H = frame.shape[1], frame.shape[0]
                    exp = cfg['expect_area'][c]
                    bw = cfg['best']

                    def cost(b):
                        x, y, w, h, a = b
                        edge = bw['edge_w'] if (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1) else 0.0
                        return abs(a / exp - 1.0) + bw['fill_w'] * (1.0 - a / (w * h)) + edge
                    det = min(blobs, key=cost)
                ybs = [b for b in lbl if b[0] == c]
                if ybs:
                    dark = dk.get(c, False)
                    if dark:
                        s[c]['df'] += 1
                    if det and any(overlap(det, b) for b in ybs):
                        s[c]['tp'] += 1
                        if dark:
                            s[c]['dtp'] += 1
                    else:
                        s[c]['fn'] += 1
                        if dark:
                            s[c]['dfn'] += 1
                elif det:
                    s[c]['fp'] += 1
        n += 1
        if n % 200 == 0:
            print(f'frame {n}', flush=True)
    cap.release()

    print('\n  scl_lo fill | ' + ' | '.join(
        f'{c}: prec/rec/drec' for c in COLORS))
    for (slo, fill), s in zip(combos, sc):
        row = [f'{slo:4.2f}  {fill:4.2f} |']
        for c in COLORS:
            tp, fn, fp = s[c]['tp'], s[c]['fn'], s[c]['fp']
            prec = tp / (tp + fp) if tp + fp else 0.
            rec = tp / (tp + fn) if tp + fn else 0.
            drec = s[c]['dtp'] / (s[c]['dtp'] + s[c]['dfn']) if s[c]['dtp'] + s[c]['dfn'] else 0.
            row.append(f' {prec * 100:4.1f}/{rec * 100:4.1f}/{drec * 100:4.0f} '
                       f'(tp{tp} fn{fn} fp{fp})')
        print(' '.join(row))


if __name__ == '__main__':
    main()