#!/usr/bin/env python3
"""
make_challenges.py — compose WHOLE hCaptcha challenge rounds with ground truth.

make_dataset.py emits single labelled tiles; the offline PointNet / DragNet
models (train_models.py) and the end-to-end tests (test_solver.py) need full
rounds instead:

  point rounds   — a scene with 3-5 objects and a prompt, either NAMED
                   ("please click on the frog") or RELATIONAL ("please click
                   on the animal who jumps the highest"); label = target's
                   normalised centre (x, y in 0..1, top-left origin).
  drag rounds    — a scene with a punched "slot" silhouette (circle / square
                   / triangle / puzzle) and a matching loose piece carrying
                   the hCaptcha-style "Move" badge; label = (from, to)
                   normalised centres.
  grid rounds    — 9 tiles + the correct 1-based indices, incl. AFFORDANCE
                   rounds where a reference tool heads the prompt and the
                   correct tiles are its materials.

All generators are importable and deterministic per rng:

    img, meta = make_point_round(random.Random("x"), size=96)

so test_solver.py can build HELD-OUT rounds (different seed than the
training corpora) without touching the filesystem.

CLI:
    python make_challenges.py --out data_v2/challenges \
        --n_point 7000 --n_drag 4000 --n_grid 1500 --n_count 500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import make_dataset as md
import hcaptcha_types as hct
import realdata

CLASSES = md.CLASSES                       # all 60, stable order
CID = {name: i for i, name in enumerate(CLASSES)}

# Objects that can be point targets / scene members. crosswalk is a
# full-bleed road texture, not a pointable object.
POINT_CLASSES = [c for c in CLASSES if c != "crosswalk"]

# Classes with a numerical rank -> usable in relational ("largest", ...)
# rounds. Names only (canonical form), from the shared tables.
RANKABLE = sorted(set(hct.SIZE_RANK) | set(hct.JUMP_RANK) |
                  set(hct.SPEED_RANK))

_NAMED_PROMPTS = (
    "Please click on the {name}",
    "Please click on the {name}.",
    "Click on the {name}",
    "Please click on the {name} in the image",
)

_POINT_SUPERLATIVES = (
    ("Please click on the animal who jumps the highest", "JUMP", "max"),
    ("Please click on the animal that jumps the highest", "JUMP", "max"),
    ("Please click on the animal that jumps the lowest", "JUMP", "min"),
    ("Please click on the largest object in the image", "SIZE", "max"),
    ("Please click on the smallest object in the image", "SIZE", "min"),
    ("Please click on the fastest moving object", "SPEED", "max"),
    ("Please click on the slowest moving object", "SPEED", "min"),
    ("Please click on the animal from the coldest place", "TEMP", "min"),
    ("Please click on the animal from the warmest place", "TEMP", "max"),
)

_TABLES = {"JUMP": hct.JUMP_RANK, "SIZE": hct.SIZE_RANK,
           "SPEED": hct.SPEED_RANK, "TEMP": hct.TEMP_RANK}

_DRAG_PROMPTS = (
    "Please drag the element to the place where it fits",
    "Drag the element to the place where it fits best",
    "Please complete the puzzle by dragging the element",
)

_AFFORDANCE_PROMPTS = (
    "Please pick all things you can work on with the item shown in the image",
    "Please pick all the objects you can work on with the item shown",
    "Please click each image you can use the item shown on",
)


# ── shared helpers ────────────────────────────────────────────────────────


def _scene_bg(S, rng):
    """Background for a round: REAL photo crop ~60% of the time (street,
    meadow, beach, desk, asphalt — what live challenges actually show),
    painted scene otherwise. realdata degrades to painted when the corpus
    is absent."""
    if rng.random() < 0.60:
        bg = realdata.real_scene(S, rng)
        if bg is not None:
            return bg
    mood = rng.choice(md.LIGHTING)
    kind = rng.choice(["grass", "road", "water"])
    horizon = rng.randint(int(S * 0.34), int(S * 0.52))
    img = Image.new("RGB", (S, S), (0, 0, 0))
    md.draw_scene(ImageDraw.Draw(img), S, rng, mood, kind, horizon)
    return img.filter(ImageFilter.GaussianBlur(0.6)) \
        if rng.random() < 0.3 else img


def _object_layer(name, w, h, rng):
    """Render one object painter into an RGBA layer (no background)."""
    from PIL import ImageDraw as _D  # local alias
    layer = Image.new("RGBA", (max(4, w), max(4, h)), (0, 0, 0, 0))
    md.PAINTERS[name](_D.Draw(layer), layer.size[0], layer.size[1],
                      rng, "day")
    if rng.random() < 0.5 and name not in ("traffic_light", "red_light",
                                           "clock", "wrench", "screwdriver"):
        layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
    ang = rng.uniform(-14, 14)
    return layer.rotate(ang, resample=Image.BICUBIC, expand=True)


def _paste_object(img, name, box, rng):
    """Paste `name` into img; box=(cx_norm, cy_norm, size_norm). Returns the
    actual pixel (cx, cy, radius) of the pasted object.

    Prefers a REAL photo tile of the object (hCaptcha serves photographs,
    not cartoons) scaled to the requested footprint; falls back to the
    painted layer when the class has no real photos yet."""
    S = img.size[0]
    cx, cy, size = box
    px_side = max(16, int(S * size))
    if rng.random() < 0.55:
        # favourite: real object CUTOUT alpha-composited onto the scene —
        # real pixels, no rectangular photo frame, closest to live rounds
        cut = realdata.cutout_for_class(name, rng)
        if cut is not None:
            w, h = cut.size
            scale = px_side / max(w, h)
            cut = cut.resize((max(8, int(w * scale)), max(8, int(h * scale))),
                             Image.LANCZOS)
            if rng.random() < 0.5:
                cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
            cut = cut.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC,
                             expand=True)
            x0 = int(cx * S - cut.size[0] / 2)
            y0 = int(cy * S - cut.size[1] / 2)
            img.paste(cut, (x0, y0), cut)
            return (cx, cy, size * 0.5)
    if rng.random() < 0.62:
        tile = realdata.real_tile(name, rng, size=px_side)
        if tile is not None:
            x0 = int(cx * S - px_side / 2)
            y0 = int(cy * S - px_side / 2)
            img.paste(tile, (x0, y0))
            return (cx, cy, size * 0.5)
    ar = md.GEOMETRY.get(name, (0.5, 0.8, 1.0))[2]
    oh = max(14, int(S * size))
    ow = max(14, int(oh * ar))
    layer = _object_layer(name, ow, oh, rng)
    px = int(cx * S - layer.size[0] / 2)
    py = int(cy * S - layer.size[1] / 2)
    img.paste(layer, (px, py), layer)
    return (cx, cy, size * 0.5)


def _spread_centers(rng, n, lo=0.17, hi=0.83, min_gap=0.26, tries=60):
    """n mutually separated normalised centers."""
    pts = []
    for _ in range(tries):
        p = (rng.uniform(lo, hi), rng.uniform(lo, hi))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_gap for q in pts):
            pts.append(p)
        if len(pts) == n:
            break
    while len(pts) < n:  # degenerate rng: fall back to a ring
        a = 2 * math.pi * len(pts) / n
        pts.append((0.5 + 0.30 * math.cos(a), 0.5 + 0.30 * math.sin(a)))
    return pts


# ── point rounds ──────────────────────────────────────────────────────────


def make_point_round(rng: random.Random, size: int = 96):
    """(image, meta). meta: prompt, target, target_id, x, y (norm),
    relational flag, objects list."""
    S = size
    n = rng.randint(3, 5)
    relational = rng.random() < 0.5
    if relational:
        prompt, table_name, direction = rng.choice(_POINT_SUPERLATIVES)
        table = _TABLES[table_name]
        # pick objects with UNIQUE table values so the argmax target is
        # unambiguous — dedupe BEFORE pasting so labels match pixels
        pool = [c for c in RANKABLE if c in table]
        if table_name == "TEMP":   # the prompt says "animal" — keep it true
            pool = [c for c in pool if c in hct.ANIMALS]
        rng.shuffle(pool)
        names, seen = [], set()
        for cand in pool:
            if table[cand] not in seen:
                names.append(cand)
                seen.add(table[cand])
            if len(names) == n:
                break
        if len(names) < n:
            return make_point_round(rng, size)  # degenerate rng; resample
    else:
        names = rng.sample(POINT_CLASSES, n)

    centers = _spread_centers(rng, n)
    img = _scene_bg(S, rng)
    objects = []
    for name, c in zip(names, centers):
        scale = rng.uniform(0.26, 0.42)
        x, y, r = _paste_object(img, name, (c[0], c[1], scale), rng)
        objects.append({"name": name, "x": round(x, 4), "y": round(y, 4),
                        "r": round(r, 4)})

    if relational:
        vals = [table[o["name"]] for o in objects]
        pick = max(range(n), key=lambda i: vals[i]) if direction == "max" \
            else min(range(n), key=lambda i: vals[i])
    else:
        pick = rng.randrange(n)
        prompt = rng.choice(_NAMED_PROMPTS).format(
            name=objects[pick]["name"].replace("_", " "))

    tgt = objects[pick]
    meta = {
        "type": "point",
        "prompt": prompt,
        "target": tgt["name"],
        "target_id": CID[tgt["name"]],
        "x": round(tgt["x"], 4),
        "y": round(tgt["y"], 4),
        "relational": relational,
        "objects": objects,
    }
    return img, meta


def make_count_round(rng: random.Random, size: int = 96):
    """Counting round: k separated instances of ONE class on a scene,
    prompt "How many X are in this image?", ground-truth count k."""
    name = rng.choice(POINT_CLASSES)
    k = rng.randint(2, 5)
    img = _scene_bg(size, rng)
    centers = _spread_centers(rng, k, lo=0.14, hi=0.86, min_gap=0.30)
    k = len(centers)
    if k < 2:
        return make_count_round(rng, size)      # degenerate rng; resample
    objects = []
    for c in centers:
        scale = rng.uniform(0.16, 0.26)
        x, y, r = _paste_object(img, name, (c[0], c[1], scale), rng)
        objects.append({"x": round(x, 4), "y": round(y, 4),
                        "r": round(r, 4)})
    prompt = "How many %ss are in this image?" % name.replace("_", " ")
    meta = {
        "type": "count",
        "prompt": prompt,
        "target": name,
        "target_id": CID[name],
        "count": k,
        "objects": objects,
    }
    return img, meta


def make_pattern_round(rng: random.Random, size: int = 96):
    """Pattern-completion drag round: a 3x3 grid of painted icons with ONE
    empty cell and 3 candidates below. Placing the right candidate makes
    every row and column hold three distinct labels (Latin square).

    meta: grid (9 labels, None for the hole), hole (0-8), candidates
    (labels), correct (candidate index), cell_boxes, candidate_boxes
    (normalised rects) for the offline crop-classify test path."""
    animals = [c for c in POINT_CLASSES if c in hct.ANIMALS]
    pool = rng.sample(animals, 3)
    base = [pool[0], pool[1], pool[2],
            pool[1], pool[2], pool[0],
            pool[2], pool[0], pool[1]]
    # random Latin-square-preserving shuffle: permute rows, then columns
    rp = list(range(3))
    rng.shuffle(rp)
    rows = [[base[r * 3 + c] for c in range(3)] for r in rp]
    cp = list(range(3))
    rng.shuffle(cp)
    rows = [[row[c] for c in cp] for row in rows]
    flat = [x for row in rows for x in row]
    hole = rng.randrange(9)
    correct_label = flat[hole]
    flat[hole] = None
    cands = list(pool)
    rng.shuffle(cands)
    correct = cands.index(correct_label)

    # landscape canvas like the real UI (grid on top, candidates below) and
    # ~40px cells: the 64px-input tile classifier labels >=40px painted
    # tiles at ~94% (25px crops only reach ~52%, which defeats the offline
    # pattern logic before it starts)
    W = int(size * 2.0)
    H = int(W * 1.25)
    cell = int(W * 0.21)
    gap = int(W * 0.035)
    gx0 = int(W * 0.05)
    gy0 = int(H * 0.05)
    img = Image.new("RGB", (W, H), (236, 236, 240))
    cell_boxes = []
    for i in range(9):
        r, c = divmod(i, 3)
        x0 = gx0 + c * (cell + gap)
        y0 = gy0 + r * (cell + gap)
        cell_boxes.append({"x": round(x0 / W, 4), "y": round(y0 / H, 4),
                           "w": round(cell / W, 4),
                           "h": round(cell / H, 4)})
        if flat[i] is None:
            d = ImageDraw.Draw(img)
            d.rectangle([x0, y0, x0 + cell, y0 + cell],
                        fill=(250, 250, 252), outline=(168, 170, 180),
                        width=2)
            continue
        tile = md.render(flat[i], cell, rng)
        img.paste(tile, (x0, y0))
    cy0 = gy0 + 3 * (cell + gap)
    cw = cell
    cand_boxes = []
    for i, name in enumerate(cands):
        x0 = gx0 + i * (cw + gap)
        cand_boxes.append({"x": round(x0 / W, 4),
                           "y": round(cy0 / H, 4),
                           "w": round(cw / W, 4),
                           "h": round(cw / H, 4)})
        tile = md.render(name, cw, rng)
        img.paste(tile, (x0, cy0))
    prompt = ("Put one of the animals into the empty spot to complete the "
              "pattern")
    meta = {
        "type": "pattern",
        "prompt": prompt,
        "grid": flat,
        "hole": hole,
        "candidates": cands,
        "correct": correct,
        "cell_boxes": cell_boxes,
        "candidate_boxes": cand_boxes,
    }
    return img, meta


# ── drag rounds ───────────────────────────────────────────────────────────


def _shape_mask(kind, w, h, knob=True):
    """RGBA mask for the piece shape; `puzzle` gets a knob."""
    m = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(m)
    if kind == "circle":
        d.ellipse([2, 2, w - 3, h - 3], fill=(255, 255, 255, 255))
    elif kind == "square":
        d.rounded_rectangle([2, 2, w - 3, h - 3], radius=max(2, w // 10),
                            fill=(255, 255, 255, 255))
    elif kind == "triangle":
        d.polygon([(w / 2, 2), (w - 2, h - 3), (2, h - 3)],
                  fill=(255, 255, 255, 255))
    else:  # puzzle piece: square + knob on top
        d.rectangle([2, h * 0.28, w - 3, h - 3], fill=(255, 255, 255, 255))
        d.ellipse([w * 0.30, 0, w * 0.70, h * 0.42], fill=(255, 255, 255, 255))
    return m


def make_drag_round(rng: random.Random, size: int = 96):
    """(image, meta). meta: fx, fy, tx, ty (normalised centres), shape."""
    S = size
    kind = rng.choice(["circle", "square", "triangle", "puzzle"])
    img = _scene_bg(S, rng)

    pw = int(S * rng.uniform(0.20, 0.28))
    ph = int(pw * (1.15 if kind == "puzzle" else 1.0))
    mask = _shape_mask(kind, pw, ph, knob=True)

    # slot (punched hole) and piece positions, clearly apart
    tx, ty = rng.uniform(0.42, 0.80), rng.uniform(0.28, 0.72)
    while True:
        fx, fy = rng.uniform(0.18, 0.80), rng.uniform(0.24, 0.74)
        if math.hypot(fx - tx, fy - ty) >= 0.34:
            break

    # punched slot: darkened + outlined silhouette of the same shape
    slot_layer = Image.new("RGBA", (pw + 8, ph + 8), (0, 0, 0, 0))
    sd = ImageDraw.Draw(slot_layer)
    big = mask.resize((pw, ph))
    slot_layer.paste((0, 0, 0, 150), (4, 4), big)
    # white outline ring around the hole so it reads as a target
    outline = big.filter(ImageFilter.MaxFilter(5))
    ring = Image.new("RGBA", (pw + 8, ph + 8), (0, 0, 0, 0))
    ring.paste((250, 250, 252, 220), (4, 4), outline)
    ring.paste((0, 0, 0, 0), (4, 4), big)
    img.paste(ring, (int(tx * S - pw / 2) - 4, int(ty * S - ph / 2) - 4), ring)
    img.paste(slot_layer, (int(tx * S - pw / 2) - 4, int(ty * S - ph / 2) - 4),
              slot_layer)

    # the loose piece: vivid fill cut to the same silhouette
    col = tuple(rng.randint(60, 235) for _ in range(3))
    piece = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    fill = Image.new("RGBA", (pw, ph), col + (255,))
    pd = ImageDraw.Draw(fill)
    for _ in range(4):  # a little texture so it reads as a solid object
        gx, gy = rng.randint(0, pw - 4), rng.randint(0, ph - 4)
        pd.ellipse([gx - 3, gy - 3, gx + 3, gy + 3],
                   fill=tuple(min(255, c + 30) for c in col) + (255,))
    piece.paste(fill, (0, 0), mask)
    img.paste(piece, (int(fx * S - pw / 2), int(fy * S - ph / 2)), piece)

    # "Move" badge above the piece (what hCaptcha's hint bubble says)
    try:
        font = ImageFont.load_default(size=max(9, S // 9))
    except TypeError:
        font = ImageFont.load_default()
    label = "Move"
    d = ImageDraw.Draw(img)
    tb = d.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    bx = int(fx * S - tw / 2)
    by = int(fy * S - ph / 2) - th - 8
    by = max(2, by)
    d.rounded_rectangle([bx - 5, by - 3, bx + tw + 5, by + th + 4],
                        radius=4, fill=(250, 250, 252, 235),
                        outline=(60, 60, 66))
    d.text((bx, by), label, font=font, fill=(40, 40, 46))

    meta = {
        "type": "drag",
        "prompt": rng.choice(_DRAG_PROMPTS),
        "shape": kind,
        "fx": round(fx, 4), "fy": round(fy, 4),
        "tx": round(tx, 4), "ty": round(ty, 4),
    }
    return img, meta


# ── grid rounds ───────────────────────────────────────────────────────────

GRID_TILE = 96


def make_grid_round(rng: random.Random, size: int = GRID_TILE):
    """3x3 grid round.

    Returns (grid_image, meta). meta: prompt, tiles (9 class names),
    correct (1-based indices), reference (affordance tool name or None),
    reference_image (an RGB PIL tile or None).
    """
    S = size
    affordance = rng.random() < 0.4
    if affordance:
        reference = rng.choice(sorted(hct.TOOL_AFFORDANCE))
        allowed = sorted(hct.TOOL_AFFORDANCE[reference])
        k = rng.randint(1, min(4, len(allowed)))
        positives = rng.sample(allowed, k)
        negatives_src = [c for c in POINT_CLASSES
                         if c not in hct.TOOL_AFFORDANCE[reference]
                         and c != reference]
        prompt = rng.choice(_AFFORDANCE_PROMPTS)
    else:
        reference = None
        target = rng.choice(POINT_CLASSES)
        positives = [target] * rng.randint(1, 4)
        negatives_src = [c for c in POINT_CLASSES if c != target]
        prompt = md.PROMPTS[target]

    negatives = rng.sample(negatives_src, 9 - len(positives))
    names = positives + negatives
    rng.shuffle(names)
    correct = sorted(i + 1 for i, n in enumerate(names)
                     if n in set(positives))

    gap = max(3, S // 20)
    G = 3 * S + 4 * gap
    grid = Image.new("RGB", (G, G), (52, 54, 60))
    boxes = []
    for i, name in enumerate(names):
        tile = None
        if rng.random() < 0.55:      # real photo tiles where we have them
            tile = realdata.real_tile(name, rng, size=S)
        if tile is None:
            tile = md.render(name, S, rng)
        r, c = divmod(i, 3)
        x = gap + c * (S + gap)
        y = gap + r * (S + gap)
        grid.paste(tile, (x, y))
        boxes.append([x, y, S, S])

    ref_img = None
    if reference:
        ref_img = realdata.real_tile(reference, rng, size=S) \
            if rng.random() < 0.7 else None
        if ref_img is None:
            ref_img = md.render(reference, S, rng)

    meta = {
        "type": "grid",
        "prompt": prompt,
        "tiles": names,
        "correct": correct,
        "reference": reference,
        "tile_boxes": boxes,
        "affordance": affordance,
    }
    if ref_img is not None:
        meta["reference_image"] = ref_img   # kept in-memory; stripped on save
    return grid, meta


# ── file output ───────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Full hCaptcha challenge rounds "
                                 "with ground truth")
    ap.add_argument("--out", default="data_v2/challenges")
    ap.add_argument("--n_point", type=int, default=7000)
    ap.add_argument("--n_drag", type=int, default=4000)
    ap.add_argument("--n_grid", type=int, default=1500)
    ap.add_argument("--n_count", type=int, default=500)
    ap.add_argument("--n_pattern", type=int, default=300)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    manifest_path = os.path.join(a.out, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for sub, count, fn in (("point", a.n_point, make_point_round),
                               ("drag", a.n_drag, make_drag_round),
                               ("grid", a.n_grid, make_grid_round),
                               ("count", a.n_count, make_count_round),
                               ("pattern", a.n_pattern, make_pattern_round)):
            d = os.path.join(a.out, sub)
            os.makedirs(d, exist_ok=True)
            for i in range(count):
                rng = random.Random("%d|%s|%d" % (a.seed, sub, i))
                img, meta = fn(rng, a.size)
                rel = os.path.join(a.out, sub, "%s_%05d.jpg" % (sub, i))
                ref_img = meta.pop("reference_image", None)
                if ref_img is not None:
                    ref_path = os.path.join(
                        a.out, sub, "%s_%05d_ref.jpg" % (sub, i))
                    ref_img.save(ref_path, "JPEG", quality=90, optimize=True)
                    meta["reference_img"] = ref_path.replace(os.sep, "/")
                img.save(rel, "JPEG", quality=90, optimize=True)
                meta["image"] = rel.replace(os.sep, "/")
                mf.write(json.dumps(meta) + "\n")
                if i % 500 == 0:
                    print("  %s %d/%d" % (sub, i, count))
            print("  %-6s %d rounds" % (sub, count))

    print("manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
