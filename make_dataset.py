#!/usr/bin/env python3
"""
make_dataset.py — procedural, deterministic, network-free generator of
labeled hCaptcha-tile-style training images.

60 classes: the original 13 traffic classes plus the 47 families merged from
synth_shapes.py (animals, tools, materials, household/terrain, plus 11 extra
safari/farm/water/household classes) so the offline solver models also cover
reference-affordance grids and point/drag rounds.

Everything is drawn with Pillow: no downloads, no external assets, no
randomness beyond a seeded PRNG, so two runs with the same --seed produce
byte-identical output.

Usage
-----
    pip install Pillow
    python make_dataset.py --per_class 600 --out data
    python make_dataset.py --per_class 3000 --out data     # 39,000 images

Layout
------
    data/<class>/<class>_00000.jpg   ... one folder per class
    data/manifest.jsonl              ... {"image","label","class_id","prompt"}
    data/_preview.jpg                ... contact sheet

Label rules
-----------
    red_light      -> traffic light with the RED lamp always lit
    traffic_light  -> traffic light with the red lamp NEVER lit
                      (dim 3-lamp signal, or yellow/green lit)
    crosswalk      -> white zebra stripes across a road band
"""

import argparse
import json
import math
import os
import random
import sys

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
except ImportError:  # pragma: no cover
    sys.stderr.write("Pillow is required:  pip install Pillow\n")
    raise

# ── classes ───────────────────────────────────────────────────────────────

CLASSES = [
    "bus",
    "car",
    "truck",
    "train",
    "bicycle",
    "motorcycle",
    "boat",
    "airplane",
    "traffic_light",
    "red_light",
    "crosswalk",
    "fire_hydrant",
    "parking_meter",
]

PROMPTS = {
    "bus":           "Please click each image containing a bus",
    "car":           "Please click each image containing a car",
    "truck":         "Please click each image containing a truck",
    "train":         "Please click each image containing a train",
    "bicycle":       "Please click each image containing a bicycle",
    "motorcycle":    "Please click each image containing a motorcycle",
    "boat":          "Please click each image containing a boat",
    "airplane":      "Please click each image containing an airplane",
    "traffic_light": "Please click each image containing a traffic light",
    "red_light":     "Please click each image containing a red light",
    "crosswalk":     "Please click each image containing a crosswalk",
    "fire_hydrant":  "Please click each image containing a fire hydrant",
    "parking_meter": "Please click each image containing a parking meter",
}

# ── small colour helpers ──────────────────────────────────────────────────


def clamp(v, lo=0, hi=255):
    return int(max(lo, min(hi, v)))


def mix(c1, c2, t):
    return tuple(clamp(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def shade(c, f):
    return tuple(clamp(v * f) for v in c)


def jitter(rng, c, amount=14):
    return tuple(clamp(v + rng.randint(-amount, amount)) for v in c)


BODY_COLORS = [
    (196, 60, 55), (44, 82, 158), (232, 174, 44), (232, 232, 236),
    (36, 38, 44), (58, 132, 96), (150, 152, 158), (120, 66, 156),
    (222, 120, 48), (30, 96, 128),
]

# ── scene background ──────────────────────────────────────────────────────

LIGHTING = ["day", "day", "day", "dusk", "night"]


def sky_palette(rng, mood):
    if mood == "day":
        top = jitter(rng, (86, 146, 214))
        bot = jitter(rng, (188, 216, 238))
    elif mood == "dusk":
        top = jitter(rng, (62, 58, 110))
        bot = jitter(rng, (232, 138, 86))
    else:
        top = jitter(rng, (10, 12, 26), 6)
        bot = jitter(rng, (38, 42, 66), 8)
    return top, bot


def ground_palette(rng, mood, kind):
    if kind == "water":
        base = (40, 78, 122)
    elif kind == "grass":
        base = (74, 108, 58)
    else:  # asphalt
        base = (76, 78, 84)
    if mood == "dusk":
        base = mix(base, (120, 70, 50), 0.28)
    elif mood == "night":
        base = shade(base, 0.42)
    return jitter(rng, base, 10)


def draw_scene(d, S, rng, mood, ground_kind, horizon):
    """Sky gradient + ground band + a few ambient details."""
    top, bot = sky_palette(rng, mood)
    for y in range(horizon):
        t = y / max(1, horizon - 1)
        d.line([(0, y), (S, y)], fill=mix(top, bot, t))

    if mood == "night":
        for _ in range(rng.randint(14, 30)):
            x, y = rng.randint(0, S), rng.randint(0, max(1, horizon - 2))
            d.point((x, y), fill=(220, 224, 236))
    elif mood == "day" and rng.random() < 0.65:
        for _ in range(rng.randint(1, 3)):
            cx = rng.randint(0, S)
            cy = rng.randint(2, max(3, horizon // 2))
            r = rng.randint(S // 12, S // 5)
            for k in range(3):
                ox = cx + int(r * 0.7 * (k - 1))
                rr = int(r * (0.7 + 0.3 * (k == 1)))
                d.ellipse([ox - rr, cy - rr // 2, ox + rr, cy + rr // 2],
                          fill=(238, 240, 246))

    g = ground_palette(rng, mood, ground_kind)
    d.rectangle([0, horizon, S, S], fill=g)
    d.line([(0, horizon), (S, horizon)], fill=shade(g, 0.7))

    if ground_kind == "water":
        for _ in range(rng.randint(6, 14)):
            y = rng.randint(horizon + 1, S - 1)
            x = rng.randint(0, S - 8)
            w = rng.randint(5, max(6, S // 4))
            d.line([(x, y), (x + w, y)], fill=shade(g, 1.28))
    elif ground_kind == "road":
        ly = horizon + (S - horizon) // 2
        step = max(6, S // 9)
        for x in range(rng.randint(0, step), S, step * 2):
            d.rectangle([x, ly, x + step, ly + max(1, S // 64)],
                        fill=(216, 208, 168) if mood != "night" else (140, 134, 108))
    else:
        for _ in range(rng.randint(5, 12)):
            x = rng.randint(0, S)
            y = rng.randint(horizon, S - 1)
            d.line([(x, y), (x + rng.randint(-2, 2), y - rng.randint(2, 5))],
                   fill=shade(g, 1.25))


# ── object painters (drawn on a transparent RGBA layer) ───────────────────
#
# Every painter draws inside a w x h box starting at (0, 0) of the layer.


def _wheels(d, x0, y0, w, h, rng, n=2, r=None):
    r = r or max(3, int(h * 0.16))
    ys = y0 + h - r
    xs = [x0 + int(w * 0.20), x0 + int(w * 0.80)] if n == 2 else \
         [x0 + int(w * (0.14 + 0.72 * i / max(1, n - 1))) for i in range(n)]
    for x in xs:
        d.ellipse([x - r, ys - r, x + r, ys + r], fill=(26, 26, 30))
        d.ellipse([x - r // 2, ys - r // 2, x + r // 2, ys + r // 2],
                  fill=(150, 152, 160))


def paint_bus(d, w, h, rng, mood):
    col = jitter(rng, rng.choice([(224, 168, 40), (196, 60, 55), (58, 110, 168),
                                  (232, 232, 236), (58, 132, 96)]))
    body_h = int(h * 0.68)
    top = int(h * 0.10)
    d.rounded_rectangle([0, top, w - 1, top + body_h], radius=max(2, w // 22),
                        fill=col, outline=shade(col, 0.6))
    win = mix((150, 200, 220), (20, 30, 40), 0.25 if mood != "night" else 0.72)
    wy0 = top + int(body_h * 0.14)
    wy1 = top + int(body_h * 0.46)
    n = rng.randint(4, 6)
    pad = max(2, w // 26)
    cw = (w - pad * (n + 1)) / n
    for i in range(n):
        x = pad + i * (cw + pad)
        d.rectangle([x, wy0, x + cw, wy1], fill=win)
    d.rectangle([pad, top + int(body_h * 0.62), w - pad, top + int(body_h * 0.70)],
                fill=shade(col, 0.72))
    _wheels(d, 0, top, w, body_h + int(h * 0.14), rng, n=2)
    if mood == "night":
        d.ellipse([w - max(4, w // 14), top + int(body_h * 0.52),
                   w - 1, top + int(body_h * 0.66)], fill=(255, 240, 190))


def paint_car(d, w, h, rng, mood):
    col = jitter(rng, rng.choice(BODY_COLORS))
    base = int(h * 0.72)
    roof_y = int(h * 0.30)
    d.rounded_rectangle([0, int(h * 0.50), w - 1, base], radius=max(2, h // 10),
                        fill=col, outline=shade(col, 0.6))
    d.polygon([(int(w * 0.24), int(h * 0.52)), (int(w * 0.36), roof_y),
               (int(w * 0.68), roof_y), (int(w * 0.80), int(h * 0.52))],
              fill=shade(col, 0.92), outline=shade(col, 0.55))
    win = mix((160, 205, 225), (18, 26, 36), 0.22 if mood != "night" else 0.7)
    d.polygon([(int(w * 0.30), int(h * 0.50)), (int(w * 0.39), roof_y + 2),
               (int(w * 0.50), roof_y + 2), (int(w * 0.50), int(h * 0.50))], fill=win)
    d.polygon([(int(w * 0.53), int(h * 0.50)), (int(w * 0.53), roof_y + 2),
               (int(w * 0.65), roof_y + 2), (int(w * 0.74), int(h * 0.50))], fill=win)
    _wheels(d, 0, int(h * 0.30), w, int(h * 0.62), rng, n=2)
    lamp = (255, 238, 190) if mood == "night" else (240, 240, 226)
    d.ellipse([w - max(4, w // 12), int(h * 0.56), w - 2, int(h * 0.64)], fill=lamp)
    d.ellipse([2, int(h * 0.56), max(5, w // 12), int(h * 0.64)], fill=(200, 60, 55))


def paint_truck(d, w, h, rng, mood):
    cab = jitter(rng, rng.choice([(196, 60, 55), (44, 82, 158), (36, 38, 44),
                                  (232, 232, 236)]))
    box = jitter(rng, rng.choice([(226, 226, 230), (150, 152, 158), (196, 196, 200)]))
    top = int(h * 0.16)
    base = int(h * 0.74)
    d.rectangle([0, top, int(w * 0.62), base], fill=box, outline=shade(box, 0.62))
    for i in range(1, 4):
        x = int(w * 0.62 * i / 4)
        d.line([(x, top + 2), (x, base - 2)], fill=shade(box, 0.82))
    d.rounded_rectangle([int(w * 0.64), int(h * 0.34), w - 1, base],
                        radius=max(2, w // 26), fill=cab, outline=shade(cab, 0.6))
    win = mix((160, 205, 225), (18, 26, 36), 0.22 if mood != "night" else 0.7)
    d.rectangle([int(w * 0.70), int(h * 0.40), int(w * 0.94), int(h * 0.55)], fill=win)
    _wheels(d, 0, top, w, base - top + int(h * 0.12), rng, n=4)


def paint_train(d, w, h, rng, mood):
    col = jitter(rng, rng.choice([(178, 44, 48), (44, 74, 140), (58, 132, 96),
                                  (200, 200, 206), (40, 42, 50)]))
    top = int(h * 0.18)
    base = int(h * 0.80)
    d.rounded_rectangle([0, top, w - 1, base], radius=max(2, h // 8),
                        fill=col, outline=shade(col, 0.6))
    d.rectangle([0, int(h * 0.62), w - 1, int(h * 0.68)], fill=shade(col, 0.7))
    win = mix((160, 205, 225), (18, 26, 36), 0.2 if mood != "night" else 0.68)
    d.rounded_rectangle([int(w * 0.62), top + int(h * 0.06),
                         int(w * 0.94), int(h * 0.46)],
                        radius=max(1, w // 30), fill=win)
    n = rng.randint(3, 5)
    for i in range(n):
        x = int(w * 0.06 + i * (w * 0.50 / n))
        d.rectangle([x, top + int(h * 0.08), x + int(w * 0.34 / n), int(h * 0.42)],
                    fill=win)
    d.rectangle([0, base, w - 1, base + max(2, int(h * 0.06))], fill=(38, 38, 44))
    for i in range(4):
        x = int(w * (0.12 + 0.24 * i))
        r = max(2, int(h * 0.05))
        d.ellipse([x - r, base + 1, x + r, base + 1 + 2 * r], fill=(26, 26, 30))
    if mood == "night":
        d.ellipse([w - max(5, w // 12), int(h * 0.50), w - 2, int(h * 0.60)],
                  fill=(255, 246, 205))


def paint_bicycle(d, w, h, rng, mood):
    col = jitter(rng, rng.choice([(230, 232, 236), (196, 60, 55), (44, 82, 158),
                                  (36, 38, 44), (58, 132, 96)]))
    lw = max(1, w // 34)
    r = int(min(w, h) * 0.26)
    cy = int(h * 0.70)
    lx, rx = int(w * 0.24), int(w * 0.76)
    for cx in (lx, rx):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(32, 32, 38), width=lw + 1)
        for k in range(6):
            a = math.pi * k / 6
            d.line([(cx - r * math.cos(a), cy - r * math.sin(a)),
                    (cx + r * math.cos(a), cy + r * math.sin(a))],
                   fill=(160, 162, 170), width=1)
    seat = (int(w * 0.38), int(h * 0.36))
    bar = (int(w * 0.66), int(h * 0.34))
    d.line([(lx, cy), seat], fill=col, width=lw)
    d.line([seat, bar], fill=col, width=lw)
    d.line([(lx, cy), bar], fill=col, width=lw)
    d.line([(rx, cy), bar], fill=col, width=lw)
    d.line([(int(w * 0.50), cy), seat], fill=col, width=lw)
    d.line([(bar[0] - int(w * 0.06), bar[1] - int(h * 0.04)),
            (bar[0] + int(w * 0.06), bar[1] - int(h * 0.04))],
           fill=(40, 40, 46), width=lw)
    d.line([(seat[0] - int(w * 0.05), seat[1] - int(h * 0.03)),
            (seat[0] + int(w * 0.04), seat[1] - int(h * 0.03))],
           fill=(30, 30, 34), width=lw + 1)


def paint_motorcycle(d, w, h, rng, mood):
    col = jitter(rng, rng.choice([(36, 38, 44), (196, 60, 55), (44, 82, 158),
                                  (222, 120, 48)]))
    r = int(min(w, h) * 0.22)
    cy = int(h * 0.74)
    lx, rx = int(w * 0.24), int(w * 0.78)
    for cx in (lx, rx):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(26, 26, 30))
        d.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2],
                  fill=(150, 152, 160))
    d.polygon([(lx, cy - r // 2), (int(w * 0.42), int(h * 0.48)),
               (int(w * 0.66), int(h * 0.48)), (rx, cy - r // 2),
               (int(w * 0.56), int(h * 0.62)), (int(w * 0.38), int(h * 0.62))],
              fill=col)
    d.rounded_rectangle([int(w * 0.36), int(h * 0.40), int(w * 0.60), int(h * 0.52)],
                        radius=max(1, w // 28), fill=shade(col, 1.15))
    d.line([(rx, cy - r), (int(w * 0.72), int(h * 0.34))],
           fill=(60, 62, 70), width=max(1, w // 26))
    d.line([(int(w * 0.66), int(h * 0.32)), (int(w * 0.80), int(h * 0.32))],
           fill=(40, 40, 46), width=max(1, w // 30))
    if mood == "night":
        d.ellipse([int(w * 0.78), int(h * 0.36), int(w * 0.90), int(h * 0.46)],
                  fill=(255, 244, 200))


def paint_boat(d, w, h, rng, mood):
    hull = jitter(rng, rng.choice([(232, 232, 238), (44, 74, 140), (176, 66, 58),
                                   (40, 42, 50)]))
    base = int(h * 0.78)
    d.polygon([(0, int(h * 0.58)), (w - 1, int(h * 0.58)),
               (int(w * 0.84), base), (int(w * 0.16), base)],
              fill=hull, outline=shade(hull, 0.62))
    cab = shade(hull, 1.12) if sum(hull) < 400 else shade(hull, 0.86)
    d.rectangle([int(w * 0.32), int(h * 0.38), int(w * 0.62), int(h * 0.58)],
                fill=cab, outline=shade(cab, 0.6))
    win = mix((160, 205, 225), (18, 26, 36), 0.2 if mood != "night" else 0.66)
    d.rectangle([int(w * 0.36), int(h * 0.43), int(w * 0.58), int(h * 0.50)], fill=win)
    if rng.random() < 0.55:
        d.line([(int(w * 0.46), int(h * 0.38)), (int(w * 0.46), int(h * 0.10))],
               fill=(220, 220, 226), width=max(1, w // 40))
        d.polygon([(int(w * 0.48), int(h * 0.12)), (int(w * 0.48), int(h * 0.36)),
                   (int(w * 0.72), int(h * 0.36))], fill=(240, 240, 244))
    d.line([(0, base), (w - 1, base)], fill=shade(hull, 0.5), width=max(1, h // 40))


def paint_airplane(d, w, h, rng, mood):
    col = jitter(rng, rng.choice([(238, 238, 242), (206, 208, 214), (226, 230, 236)]))
    cy = int(h * 0.50)
    fh = max(3, int(h * 0.16))
    d.rounded_rectangle([int(w * 0.06), cy - fh // 2, int(w * 0.88), cy + fh // 2],
                        radius=fh // 2, fill=col, outline=shade(col, 0.66))
    d.polygon([(int(w * 0.88), cy - fh // 2), (w - 1, cy),
               (int(w * 0.88), cy + fh // 2)], fill=shade(col, 0.92))
    accent = jitter(rng, rng.choice([(44, 82, 158), (196, 60, 55), (58, 132, 96)]))
    d.line([(int(w * 0.10), cy + fh // 4), (int(w * 0.88), cy + fh // 4)],
           fill=accent, width=max(1, fh // 4))
    d.polygon([(int(w * 0.44), cy), (int(w * 0.66), cy),
               (int(w * 0.50), cy + int(h * 0.30))], fill=shade(col, 0.80))
    d.polygon([(int(w * 0.44), cy), (int(w * 0.62), cy),
               (int(w * 0.52), cy - int(h * 0.26))], fill=shade(col, 0.95))
    d.polygon([(int(w * 0.06), cy), (int(w * 0.20), cy),
               (int(w * 0.10), cy - int(h * 0.22))], fill=shade(col, 0.88))
    for wx in (int(w * 0.42), int(w * 0.56)):
        d.rounded_rectangle([wx, cy + int(h * 0.08), wx + int(w * 0.10),
                             cy + int(h * 0.17)], radius=2, fill=(90, 92, 100))
    for i in range(6):
        x = int(w * (0.16 + i * 0.05))
        d.rectangle([x, cy - 1, x + 1, cy + 1],
                    fill=(120, 160, 190) if mood != "night" else (240, 230, 160))


def _traffic_light(d, w, h, rng, mood, lit):
    """lit: 'red' | 'yellow' | 'green' | None."""
    pole_w = max(2, w // 8)
    d.rectangle([w // 2 - pole_w // 2, int(h * 0.62), w // 2 + pole_w // 2, h - 1],
                fill=(58, 60, 66))
    bx0, bx1 = int(w * 0.16), int(w * 0.84)
    by0, by1 = int(h * 0.04), int(h * 0.64)
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=max(2, w // 8),
                        fill=(34, 36, 42), outline=(18, 18, 22), width=1)
    r = max(2, int((bx1 - bx0) * 0.30))
    cx = (bx0 + bx1) // 2
    slots = ["red", "yellow", "green"]
    for i, name in enumerate(slots):
        cy = int(by0 + (by1 - by0) * (0.20 + 0.30 * i))
        on = (name == lit)
        base = {"red": (210, 46, 40), "yellow": (232, 186, 46),
                "green": (48, 194, 108)}[name]
        col = base if on else shade(base, 0.20)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col,
                  outline=(16, 16, 20))
        if on:
            glow = mix(base, (255, 255, 255), 0.55)
            d.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2], fill=glow)


def paint_traffic_light(d, w, h, rng, mood):
    # red lamp NEVER lit
    _traffic_light(d, w, h, rng, mood, rng.choice([None, None, "yellow", "green"]))


def paint_red_light(d, w, h, rng, mood):
    _traffic_light(d, w, h, rng, mood, "red")


def paint_crosswalk(d, w, h, rng, mood):
    road = (72, 74, 80) if mood != "night" else (34, 36, 42)
    y0, y1 = int(h * 0.28), int(h * 0.92)
    d.rectangle([0, y0, w - 1, y1], fill=road)
    d.line([(0, y0), (w - 1, y0)], fill=shade(road, 1.35))
    d.line([(0, y1), (w - 1, y1)], fill=shade(road, 0.7))
    n = rng.randint(4, 7)
    stripe = (238, 238, 232) if mood != "night" else (196, 198, 194)
    skew = rng.randint(-w // 12, w // 12)
    gap = w / (n * 1.9)
    sw = gap * rng.uniform(0.85, 1.15)
    for i in range(n):
        x = i * (w / n) + gap * 0.25
        d.polygon([(x, y1), (x + sw, y1),
                   (x + sw + skew, y0 + 1), (x + skew, y0 + 1)], fill=stripe)


def paint_fire_hydrant(d, w, h, rng, mood):
    col = jitter(rng, rng.choice([(198, 44, 40), (206, 62, 48), (216, 176, 46),
                                  (196, 60, 55)]))
    cx = w // 2
    bw = int(w * 0.42)
    top = int(h * 0.22)
    base = int(h * 0.92)
    d.rectangle([cx - int(w * 0.30), base - max(2, int(h * 0.06)),
                 cx + int(w * 0.30), base], fill=shade(col, 0.7))
    d.rounded_rectangle([cx - bw // 2, top, cx + bw // 2, base - int(h * 0.05)],
                        radius=max(2, bw // 4), fill=col, outline=shade(col, 0.62))
    d.ellipse([cx - int(bw * 0.42), top - int(h * 0.10),
               cx + int(bw * 0.42), top + int(h * 0.10)],
              fill=shade(col, 1.1), outline=shade(col, 0.6))
    d.rectangle([cx - int(bw * 0.10), top - int(h * 0.16),
                 cx + int(bw * 0.10), top - int(h * 0.06)], fill=shade(col, 0.8))
    cap = shade(col, 0.78)
    for side in (-1, 1):
        d.rectangle([cx + side * bw // 2 - int(w * 0.10) if side < 0 else cx + bw // 2,
                     int(h * 0.42),
                     cx - bw // 2 if side < 0 else cx + bw // 2 + int(w * 0.10),
                     int(h * 0.56)], fill=cap)
    d.rectangle([cx - bw // 2, int(h * 0.62), cx + bw // 2, int(h * 0.68)],
                fill=shade(col, 0.72))


def paint_parking_meter(d, w, h, rng, mood):
    col = jitter(rng, rng.choice([(90, 92, 100), (56, 58, 66), (140, 142, 150),
                                  (52, 96, 120)]))
    cx = w // 2
    pw = max(2, int(w * 0.14))
    d.rectangle([cx - pw // 2, int(h * 0.40), cx + pw // 2, h - 1], fill=shade(col, 0.8))
    d.rectangle([cx - int(w * 0.22), h - max(2, int(h * 0.05)),
                 cx + int(w * 0.22), h - 1], fill=shade(col, 0.6))
    hw = int(w * 0.44)
    d.rounded_rectangle([cx - hw // 2, int(h * 0.08), cx + hw // 2, int(h * 0.46)],
                        radius=max(2, hw // 4), fill=col, outline=shade(col, 0.6))
    screen = (44, 200, 150) if mood == "night" else (222, 226, 216)
    d.rounded_rectangle([cx - int(hw * 0.32), int(h * 0.16),
                         cx + int(hw * 0.32), int(h * 0.32)],
                        radius=2, fill=screen, outline=(24, 24, 28))
    d.line([(cx - int(hw * 0.18), int(h * 0.37)), (cx + int(hw * 0.18), int(h * 0.37))],
           fill=shade(col, 0.55), width=max(1, h // 48))


PAINTERS = {
    "bus": paint_bus,
    "car": paint_car,
    "truck": paint_truck,
    "train": paint_train,
    "bicycle": paint_bicycle,
    "motorcycle": paint_motorcycle,
    "boat": paint_boat,
    "airplane": paint_airplane,
    "traffic_light": paint_traffic_light,
    "red_light": paint_red_light,
    "crosswalk": paint_crosswalk,
    "fire_hydrant": paint_fire_hydrant,
    "parking_meter": paint_parking_meter,
}

# how the object sits in the scene
GROUND_KIND = {
    "boat": "water",
    "airplane": "sky",
    "bicycle": "road",
    "motorcycle": "road",
    "crosswalk": "road",
    "fire_hydrant": "grass",
    "parking_meter": "grass",
}

# object box as a fraction of the tile (min, max) and aspect ratio w/h
GEOMETRY = {
    "bus":           (0.62, 0.94, 2.05),
    "car":           (0.58, 0.92, 2.30),
    "truck":         (0.62, 0.94, 2.10),
    "train":         (0.66, 0.96, 2.40),
    "bicycle":       (0.56, 0.88, 1.55),
    "motorcycle":    (0.54, 0.86, 1.55),
    "boat":          (0.58, 0.92, 1.75),
    "airplane":      (0.62, 0.96, 2.20),
    "traffic_light": (0.55, 0.90, 0.42),
    "red_light":     (0.55, 0.90, 0.42),
    "crosswalk":     (1.00, 1.00, 1.00),
    "fire_hydrant":  (0.48, 0.84, 0.52),
    "parking_meter": (0.48, 0.86, 0.44),
}

# ── merge the extra families from synth_shapes (13 -> 60 classes) ─────────
#
# synth_shapes supplies animals / tools / materials / household painters with
# the same fn(draw, w, h, rng, mood) contract. Merging here keeps one stable
# class list (order fixed => stable class ids) and auto-fills PROMPTS for any
# class without an explicit one. Seeds stay per-class stable: the generator
# keys randomness on "%d|%d|%d" % (seed, class_id, index) — never hash().
try:
    from synth_shapes import EXTRA_PAINTERS, EXTRA_GEOMETRY, EXTRA_GROUND
except Exception:  # pragma: no cover - synth_shapes is part of the repo
    EXTRA_PAINTERS, EXTRA_GEOMETRY, EXTRA_GROUND = {}, {}, {}

for _name in EXTRA_PAINTERS:
    if _name not in PAINTERS:
        PAINTERS[_name] = EXTRA_PAINTERS[_name]
        CLASSES.append(_name)
GEOMETRY.update(EXTRA_GEOMETRY)
GROUND_KIND.update(EXTRA_GROUND)
for _name in CLASSES:
    if _name not in PROMPTS:
        PROMPTS[_name] = ("Please click each image containing a %s"
                          % _name.replace("_", " "))
del _name


# ── one image ─────────────────────────────────────────────────────────────

def render(label, S, rng):
    SS = S * 3  # supersample, downscaled at the end for clean edges
    mood = rng.choice(LIGHTING)
    kind = GROUND_KIND.get(label, "road")
    ground = "asphalt" if kind == "sky" else kind

    if kind == "sky":
        horizon = rng.randint(int(SS * 0.68), int(SS * 0.86))
    elif kind == "water":
        horizon = rng.randint(int(SS * 0.42), int(SS * 0.56))
    else:
        horizon = rng.randint(int(SS * 0.34), int(SS * 0.52))

    img = Image.new("RGB", (SS, SS), (0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_scene(d, SS, rng, mood, ground, horizon)

    if label == "crosswalk":
        layer = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
        paint_crosswalk(ImageDraw.Draw(layer), SS, SS, rng, mood)
        img.paste(layer, (0, 0), layer)
    else:
        lo, hi, ar = GEOMETRY[label]
        scale = rng.uniform(lo, hi)
        if ar >= 1:
            ow = int(SS * scale)
            oh = int(ow / ar)
        else:
            oh = int(SS * scale)
            ow = int(oh * ar)
        ow = max(12, min(ow, SS - 4))
        oh = max(12, min(oh, SS - 4))

        layer = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
        PAINTERS[label](ImageDraw.Draw(layer), ow, oh, rng, mood)

        if rng.random() < 0.5 and label not in ("traffic_light", "red_light"):
            layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
        ang = rng.uniform(-9, 9)
        layer = layer.rotate(ang, resample=Image.BICUBIC, expand=True)

        lw, lh = layer.size
        if kind == "sky":
            cy = rng.randint(int(SS * 0.10), max(int(SS * 0.11), horizon - lh // 2))
        elif kind == "water":
            cy = int(horizon + (SS - horizon) * rng.uniform(0.05, 0.40)) - lh // 2
        else:
            cy = int(horizon + (SS - horizon) * rng.uniform(0.10, 0.55)) - lh // 2
        cx = int((SS - lw) * rng.uniform(0.02, 0.98))
        cy = max(-lh // 8, min(cy, SS - int(lh * 0.85)))

        # contact shadow
        if kind in ("road", "grass"):
            sh = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sh)
            sy = cy + lh
            sd.ellipse([cx + lw * 0.05, sy - lh * 0.09,
                        cx + lw * 0.95, sy + lh * 0.05],
                       fill=(0, 0, 0, 90))
            sh = sh.filter(ImageFilter.GaussianBlur(SS * 0.012))
            img.paste(sh, (0, 0), sh)

        img.paste(layer, (cx, cy), layer)

    # global lighting wash
    if mood == "night":
        wash = Image.new("RGB", (SS, SS), (12, 16, 40))
        img = Image.blend(img, wash, 0.30)
    elif mood == "dusk":
        wash = Image.new("RGB", (SS, SS), (230, 130, 70))
        img = Image.blend(img, wash, 0.14)

    img = img.resize((S, S), Image.LANCZOS)

    # photometric augmentation
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.18))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.20))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.70, 1.30))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))

    # light grain
    px = img.load()
    for _ in range(int(S * S * 0.02)):
        x, y = rng.randrange(S), rng.randrange(S)
        r, g, b = px[x, y]
        n = rng.randint(-16, 16)
        px[x, y] = (clamp(r + n), clamp(g + n), clamp(b + n))

    return img


def contact_sheet(columns, cell=96):
    """columns: list (one per class) of lists of image paths -> one class per column."""
    cols = max(1, len(columns))
    rows = max(1, max((len(c) for c in columns), default=1))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 22))
    for cx, paths in enumerate(columns):
        for cy, p in enumerate(paths):
            try:
                im = Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS)
            except Exception:
                continue
            sheet.paste(im, (cx * cell, cy * cell))
    return sheet


README = """# Synthetic hCaptcha-tile dataset

Generated by `make_dataset.py` — procedural, deterministic (seeded) and
completely network-free. Every pixel is drawn with Pillow, so the dataset can
be regenerated anywhere with `pip install Pillow`.

## Classes ({n})

{table}

### Label rules

* **red_light** — a traffic light whose **red lamp is always lit**.
* **traffic_light** — a traffic light whose **red lamp is never lit**
  (either a dim 3-lamp signal, or the yellow/green lamp lit).
* **crosswalk** — white zebra stripes painted across a road band.

## Layout

```
{out}/<class>/<class>_00000.jpg   # one folder per class
{out}/manifest.jsonl              # one JSON object per image
{out}/_preview.jpg                # contact sheet
{out}/README.md
```

Each manifest line:

```json
{{"image": "{out}/bus/bus_00000.jpg", "label": "bus", "class_id": 0, "prompt": "Please click each image containing a bus"}}
```

## Regenerate / scale up

```bash
pip install Pillow
python make_dataset.py --per_class 600  --out {out}    # {cur:,} images
python make_dataset.py --per_class 3000 --out {out}    # {big:,} images
```

Useful flags: `--size` (tile px, default 96), `--quality` (JPEG quality),
`--seed` (default 1), `--classes a,b,c` (subset), `--no_preview`.

Seeds are derived from `(seed, class_id, index)`, so a class always renders the
same images regardless of which other classes are generated — you can add a
class or grow `--per_class` without disturbing existing files.
"""


def main():
    ap = argparse.ArgumentParser(description="Procedural hCaptcha-tile dataset generator")
    ap.add_argument("--out", default="data")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--per_class", type=int, default=600)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no_preview", action="store_true")
    a = ap.parse_args()

    names = [c.strip() for c in a.classes.split(",") if c.strip()]
    unknown = [c for c in names if c not in PAINTERS]
    if unknown:
        ap.error("unknown class(es): %s" % ", ".join(unknown))

    os.makedirs(a.out, exist_ok=True)
    manifest_path = os.path.join(a.out, "manifest.jsonl")
    preview_cols = []
    total = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for name in names:
            cid = CLASSES.index(name)          # stable id, never hash()
            cdir = os.path.join(a.out, name)
            os.makedirs(cdir, exist_ok=True)
            picks = []
            for i in range(a.per_class):
                # per-class stable seed: class_id (never hash()) + index
                rng = random.Random("%d|%d|%d" % (a.seed, cid, i))
                img = render(name, a.size, rng)
                rel = os.path.join(a.out, name, "%s_%05d.jpg" % (name, i))
                img.save(rel, "JPEG", quality=a.quality, optimize=True)
                mf.write(json.dumps({
                    "image": rel.replace(os.sep, "/"),
                    "label": name,
                    "class_id": cid,
                    "prompt": PROMPTS.get(name, "Please click each image containing a %s"
                                          % name.replace("_", " ")),
                }) + "\n")
                if i < 3:
                    picks.append(rel)
                total += 1
            preview_cols.append(picks)
            print("  %-14s %d images" % (name, a.per_class))

    table = "\n".join("| %d | `%s` | %s |" % (CLASSES.index(n), n, PROMPTS[n])
                      for n in names)
    table = "| id | class | prompt |\n|---:|---|---|\n" + table
    with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(n=len(names), table=table, out=a.out,
                              cur=len(names) * a.per_class, big=len(names) * 3000))

    if not a.no_preview:
        sheet = contact_sheet(preview_cols, cell=a.size)
        sheet.save(os.path.join(a.out, "_preview.jpg"), "JPEG", quality=90)

    print("\n%d images -> %s" % (total, a.out))
    print("manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
