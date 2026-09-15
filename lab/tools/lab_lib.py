# -*- coding: utf-8 -*-
"""Shared detection core for Hive Vision's Lab track (CIELAB / middle track).

The middle-ground Control Hub detector: more compute than the pure-HSV track
(no model, ~color rectangles) but far less than the YOLO track (full CNN on a
coprocessor). It runs a single RGB->Lab conversion, computes one per-pixel hue
angle + chroma, and thresholds them with constants learned from the flagship
YOLO model. It is designed so shadows cannot "trick" it.

Why Lab instead of HSV:
  * HSV's V and S are luminance-coupled; a shadow drags V/S down and the ball
    leaves the tuned window. CIELAB splits lightness (L*) off from
    chromaticity (a*, b*), and shadows move L* far more than hue.
  * We match on the chroma hue angle theta = atan2(b*, a*) which is
    illumination-invariant for the ball surface, then require chroma above a
    floor that is *scaled by the frame's ambient L* vs the stored reference*
    (scl = amb/ref_l). At the reference lighting scl = 1 (the trained floor);
    in shadow scl shrinks so shadowed balls - whose chroma also shrinks with
    illuminance - stay above the floor. Hue invariant + proportional floor =
    a fixed HSV rectangle can't reproduce either.
  * L* is only used as a floor to drop near-black robot parts.

Config layout (lab_tuned.json)::
    colors.<name>.hue_band    [lo, hi] degrees (wrap when lo > hi, e.g. red)
    colors.<name>.sat_floor   chroma required at reference lighting (0..255)
    colors.<name>.l_floor     absolute L* floor (kills near-black)
    ambient.ref_l             frame median L* reference (adaptive baseline)
    ambient.scl_lo/hi         adaptive sat-floor scale clamp
    gate.*                    area / aspect / fill boundaries (px at 1080p)
    expect_area.<name>        typical real-ball contour area (px)

HSV fallback: config file with an "hsv" key (cv/tools/hsv_tuned.json) is used
for side-by-side comparison, driven by detect_hsv().
"""
import json
from pathlib import Path

import cv2
import numpy as np

COLORS = ("yellow", "red", "blue")

DEFAULT_EXPECT = {"yellow": 3500.0, "red": 4200.0, "blue": 4200.0}
DEFAULT_GATE = {
    "min_area_px": 900, "max_area_px": 12000,
    "min_aspect": 0.71, "max_aspect": 1.4,
    "min_fill": 0.5, "max_fill": 1.3,
}
DEFAULT_BEST = {"fill_w": 4.0, "edge_w": 1.0}
DEFAULT_AMBIENT = {"ref_l": 120.0, "scl_lo": 0.5, "scl_hi": 1.5}
AMBI_STRIDE = 4        # subsample stride for the ambient L* estimate
MIN_L = 8.0            # discard specular blowout + super-dark pixels when fitting


def load_config(path):
    """Load a Lab config (colors.* -> arrays). Adds defaults for missing keys."""
    raw = json.loads(Path(path).read_text())
    cfg = {
        "colors": {},
        "ambient": dict(DEFAULT_AMBIENT, **raw.get("ambient", {})),
        "gate": dict(DEFAULT_GATE, **raw.get("gate", {})),
        "expect_area": dict(DEFAULT_EXPECT, **raw.get("expect_area", {})),
        "best": dict(DEFAULT_BEST, **raw.get("best", {})),
    }
    for c in COLORS:
        d = raw["colors"][c]
        cfg["colors"][c] = {
            "hue_band": list(d["hue_band"]),
            "sat_floor": float(d.get("sat_floor", 10.0)),
            "l_floor": float(d.get("l_floor", 8.0)),
        }
    return cfg


def ambient_l_from_lab(lab):
    """Median L* of the frame, subsampled. Used as the per-frame illuminance."""
    return float(np.median(lab[::AMBI_STRIDE, ::AMBI_STRIDE, 0]))


def adapt_scale(amb, ref_l, scl_lo, scl_hi):
    """Scale applied to the trained chroma floor to follow the illumination."""
    if amb <= 4.0 or ref_l <= 0.0:
        return 1.0
    return float(np.clip(amb / ref_l, scl_lo, scl_hi))


def _in_block_rans(lo, hi, hue):
    if lo <= hi:
        return (hue >= lo) & (hue <= hi)
    return (hue >= lo) | (hue <= hi)


def features(frame):
    """One-time per-frame Lab features (shared by all colors/configs).

    Returns dict with theta (chroma hue, deg 0..360), sat (chroma), L
    (lightness), amb (frame median L) and scl (adaptive floor scale). This
    mirrors the Control Hub pipeline, which also converts to Lab once and
    reuses it for every color.

    hue/chroma come from a single OpenCV cartToPolar pass (matches the
    numpy reference to within <0.01 deg / exact chroma, ~15x faster).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    sat, theta = cv2.cartToPolar(a, b, angleInDegrees=True)
    amb = ambient_l_from_lab(lab)
    return {"theta": theta, "sat": sat, "L": L, "amb": amb}


def mask_for(feat, model, scl):
    """Boolean mask for one color from precomputed frame features."""
    in_band = _in_block_rans(model["hue_band"][0], model["hue_band"][1], feat["theta"])
    return (in_band & (feat["sat"] > model["sat_floor"] * scl)
            & (feat["L"] > model["l_floor"])).astype(np.uint8) * 255


def detect(frame, cfg, gate=True, adaptive=True, feat=None, area_scale=1.0):
    """Per-color gated blobs for one frame.

    Returns {color: [(x, y, w, h, area), ...]}. Pass a precomputed
    features() dict as feat to avoid recomputing the Lab conversion +
    hue/chroma for every call (e.g. a viewer that also needs the features).

    area_scale scales the gate area bounds (px) by the ratio of processing
    resolution to the resolution the config was learned at -- pass ps*ps
    when running detection on a downscaled frame (area scales with px^2).
    """
    if feat is None:
        feat = features(frame)
    am = cfg["ambient"]
    scl = adapt_scale(feat["amb"], am["ref_l"], am["scl_lo"], am["scl_hi"]) \
        if adaptive else 1.0
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gate_cfg = cfg["gate"]
    if area_scale != 1.0 and gate:
        g = gate_cfg
        gate_cfg = dict(g, min_area_px=max(30, int(g["min_area_px"] * area_scale)),
                        max_area_px=max(1, int(g["max_area_px"] * area_scale)))
    found = {}
    for c in COLORS:
        mask = mask_for(feat, cfg["colors"][c], scl)
        if gate:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
        found[c] = blobs_from_mask(mask, gate_cfg, gate)
    return found


def blobs_from_mask(mask, gate_cfg, gate=True):
    """Gated blobs [(x, y, w, h, area)] for a single-color Lab mask."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for ct in cnts:
        x, y, w, h = cv2.boundingRect(ct)
        area = cv2.contourArea(ct)
        if area < gate_cfg["min_area_px"]:
            continue
        if gate:
            if area > gate_cfg["max_area_px"] or w <= 0 or h <= 0:
                continue
            aspect = w / h
            fill = area / (w * h)
            if aspect < gate_cfg["min_aspect"] or aspect > gate_cfg["max_aspect"] or \
               fill < gate_cfg["min_fill"] or fill > gate_cfg["max_fill"]:
                continue
        blobs.append((x, y, w, h, area))
    return blobs


def best_each(found, frame, cfg):
    """Most ball-y gated blob per color (nearest expected area + roundness +
    edge penalty). Matches the CV track scoring that OpModes use."""
    W, H = frame.shape[1], frame.shape[0]
    bw = cfg["best"]
    exp = cfg["expect_area"]
    out = {}
    for c in COLORS:
        blobs = found[c]
        if not blobs:
            out[c] = []
            continue

        def cost(b):
            x, y, w, h, a = b
            edge = 0.0
            if bw["edge_w"] and (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1):
                edge = bw["edge_w"]
            fill = a / (w * h)
            return abs(a / exp[c] - 1.0) + bw["fill_w"] * (1.0 - fill) + edge

        out[c] = [min(blobs, key=cost)]
    return out


def detect_hsv(frame, cfg):
    """HSV fallback path for comparing against cv/tools/hsv_tuned.json.

    Parses the same shapes ("segments" list of [lo, hi]) the CV track ships.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    g = cfg["gate"]
    found = {}
    for c in COLORS:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for seg in cfg["hsv"][c]["segments"]:
            lo = np.array(seg[0], dtype=np.int32)
            hi = np.array(seg[1], dtype=np.int32)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for ct in cnts:
            x, y, w, h = cv2.boundingRect(ct)
            area = cv2.contourArea(ct)
            if area < g["min_area_px"] or area > g["max_area_px"] or w <= 0 or h <= 0:
                continue
            aspect = w / h
            fill = area / (w * h)
            if aspect < g["min_aspect"] or aspect > g["max_aspect"] or \
               fill < g["min_fill"] or fill > g["max_fill"]:
                continue
            blobs.append((x, y, w, h, area))
        found[c] = blobs
    return found