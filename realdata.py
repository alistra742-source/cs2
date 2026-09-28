#!/usr/bin/env python3
"""
realdata.py — real-photo corpus for the offline solver models.

The first dataset pass (make_dataset.py + synth_shapes.py) draws every tile
with Pillow — fast, labelled, deterministic, but *drawn*. Real hCaptcha
tiles are photographs. This module turns image-search downloads (workspace
dir ``image-search/``) into a curated real corpus:

    data_real/tiles/<class>/<class>_real_NN.jpg   96x96 real photo tiles
    data_real/val/<class>/...                     held-out real photos
                                                  (never trained on)
    data_real/scenes/*.jpg                        real background scenes
                                                  (streets, grass, beach...)

``--holdout N`` photos per class go to val/ so test_solver can measure the
model against REAL photographs it never saw in training.

Classes with no reliable photo source (currently only ``red_light`` — image
search returns green/yellow signals for "red light" queries, which would
teach the exact confusion the label split exists to prevent) stay synthetic
and are listed in REAL_CLASSES's complement automatically.

Runtime helpers for make_challenges.py:

    real_tile(name, rng, size)  -> PIL square tile or None
    real_scene(size, rng)       -> PIL background photo or None
    real_frac(class_name)       -> 0..1 how much real coverage the class has

Everything degrades to None when data_real/ is absent: the generators then
produce their painted output as before (the corpora are regenerable —
data_real/ is gitignored like data_v2/).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

from PIL import Image, ImageEnhance, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIR = os.path.join(ROOT, "image-search")
REAL_DIR = os.environ.get("SOLVER_REAL_DIR", os.path.join(ROOT, "data_real"))

# query-slug substring -> class. Ordered: first match wins.
# (two fetch rounds per class: a generic query + a context/closeup query)
REAL_MAP = [
    # round 2
    ("dog-closeup", "dog"),
    ("cat-kitten", "cat"),
    ("wild-rabbit", "rabbit"),
    ("horse-head", "horse"),
    ("elephant-zoo", "elephant"),
    ("cow-pasture", "cow"),
    ("colorful-bird", "bird"),
    ("tree-frog", "frog"),
    ("sea-turtle", "turtle"),
    ("garden-snail", "snail"),
    ("kangaroo-joey", "kangaroo"),
    ("claw-hammer", "hammer"),
    ("cordless-power-drill", "drill"),
    ("handsaw-isolated", "saw"),
    ("paint-brush-dipped", "paintbrush"),
    ("adjustable-wrench", "wrench"),
    ("screwdriver-red-handle", "screwdriver"),
    ("stack-of-cut-wood", "wood"),
    ("steel-nails", "nail"),
    ("screws-and-bolts", "bolt"),
    ("green-apple", "apple"),
    ("pizza-slice", "pizza"),
    ("office-desk-table", "table"),
    ("wooden-chair", "chair"),
    ("tea-cup-saucer", "cup"),
    ("stack-of-books", "book"),
    ("alarm-clock", "clock"),
    ("colorful-umbrella", "umbrella"),
    ("pine-tree", "tree"),
    ("yellow-flower", "flower"),
    ("brick-house", "house"),
    ("mountain-range", "mountain"),
    ("hiking-boot", "boot"),
    ("yellow-school-bus", "bus"),
    ("sports-car", "car"),
    ("semi-truck", "truck"),
    ("high-speed-train", "train"),
    ("mountain-bike", "bicycle"),
    ("sport-motorcycle", "motorcycle"),
    ("fishing-boat", "boat"),
    ("jet-airplane", "airplane"),
    ("traffic-light-pole", "traffic_light"),
    ("pedestrian-crossing", "crosswalk"),
    ("red-fire-hydrant", "fire_hydrant"),
    ("coin-parking-meter", "parking_meter"),
    # round 1
    ("hammer-tool", "hammer"),
    ("power-drill", "drill"),
    ("hand-saw", "saw"),
    ("paintbrush-with", "paintbrush"),
    ("wrench-spanner", "wrench"),
    ("screwdriver-tool", "screwdriver"),
    ("dog-photo", "dog"),
    ("cat-photo", "cat"),
    ("rabbit-photo", "rabbit"),
    ("horse-photo", "horse"),
    ("elephant-photo", "elephant"),
    ("cow-photo", "cow"),
    ("bird-photo", "bird"),
    ("frog-photo", "frog"),
    ("turtle-photo", "turtle"),
    ("snail-photo", "snail"),
    ("kangaroo-photo", "kangaroo"),
    ("wooden-planks", "wood"),
    ("metal-nail", "nail"),
    ("metal-wood-screw", "screw"),
    ("hex-bolt", "bolt"),
    ("red-brick-wall", "wall"),
    ("blank-artist-canvas", "canvas"),
    ("sailboat-water", "boat"),
    ("speed-boat", "boat"),
    ("passenger-airplane", "airplane"),
    ("red-apple", "apple"),
    ("whole-pizza", "pizza"),
    ("wooden-dining-table", "table"),
    ("single-chair", "chair"),
    ("coffee-cup", "cup"),
    ("hardcover-book", "book"),
    ("round-wall-clock", "clock"),
    ("open-umbrella", "umbrella"),
    ("single-green-tree", "tree"),
    ("single-flower", "flower"),
    ("suburban-house", "house"),
    ("snowy-mountain", "mountain"),
    ("single-leather-boot", "boot"),
    ("city-bus", "bus"),
    ("sedan-car", "car"),
    ("delivery-truck", "truck"),
    ("train-locomotive", "train"),
    ("bicycle-photo", "bicycle"),
    ("motorcycle-photo", "motorcycle"),
    ("traffic-light-photo", "traffic_light"),
    # "red light lit" queries return mostly green/yellow/amber signals —
    # perfect NEGATIVE evidence for traffic_light, poisonous for red_light
    ("traffic-signal-showing-red-light", "traffic_light"),
    ("zebra-crosswalk", "crosswalk"),
    ("fire-hydrant", "fire_hydrant"),
    ("parking-meter", "parking_meter"),
    # batch 3 (49 -> 60)
    ("zebra-animal", "zebra"),
    ("zebra-standing", "zebra"),
    ("giraffe-animal", "giraffe"),
    ("giraffe-neck", "giraffe"),
    ("lion-animal", "lion"),
    ("male-lion", "lion"),
    ("bear-animal", "bear"),
    ("brown-bear", "bear"),
    ("sheep-animal", "sheep"),
    ("white-sheep", "sheep"),
    ("duck-bird", "duck"),
    ("mallard-duck", "duck"),
    ("tropical-fish", "fish"),
    ("goldfish", "fish"),
    ("butterfly-insect", "butterfly"),
    ("monarch-butterfly", "butterfly"),
    ("banana-fruit", "banana"),
    ("single-banana", "banana"),
    ("acoustic-guitar", "guitar"),
    ("wooden-guitar", "guitar"),
    ("cactus-plant", "cactus"),
    ("saguaro-cactus", "cactus"),
]

SCENE_KEYS = [
    "empty-city-street",
    "green-grass-meadow",
    "sandy-beach",
    "wooden-table-surface",
    "asphalt-road-surface",
    "kitchen-counter",
    "forest-path",
    "concrete-pavement",
    "park-lawn",
    "night-city-street",
]

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _slug_class(filename):
    base = os.path.basename(filename).lower()
    for key, cls in REAL_MAP:
        if key in base:
            return cls
    return None


def _slug_scene(filename):
    base = os.path.basename(filename).lower()
    return any(k in base for k in SCENE_KEYS)


def _to_square_tile(path, size):
    """Validate + center-crop a downloaded photo into a square tile."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if min(w, h) < 64:          # too small to be a useful photo
        return None
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    im = im.crop((x0, y0, x0 + side, y0 + side))
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return im


def organize(src=SEARCH_DIR, out=REAL_DIR, holdout=2, size=96, seed=11):
    """Sort image-search downloads into data_real/ (tiles + val + scenes).

    Rebuilds from scratch: tiles/val/scenes are wiped first so re-runs with
    newly fetched photos produce one consistent corpus (and one consistent
    train/val assignment) instead of double-numbered leftovers."""
    rng = random.Random(seed)
    for sub in ("tiles", "val", "scenes"):
        d = os.path.join(out, sub)
        if os.path.isdir(d):
            import shutil
            shutil.rmtree(d)
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(src, "*" + ext)))
    per_class = {}
    scenes = []
    skipped = 0
    for f in sorted(files):
        cls = _slug_class(f)
        if cls:
            per_class.setdefault(cls, []).append(f)
        elif _slug_scene(f):
            scenes.append(f)
        else:
            skipped += 1
    os.makedirs(out, exist_ok=True)
    stats = {"skipped_unmapped": skipped}
    tiles_dir = os.path.join(out, "tiles")
    val_dir = os.path.join(out, "val")
    for cls, flist in sorted(per_class.items()):
        rng.shuffle(flist)
        hold = flist[:holdout]
        train = flist[holdout:]
        for sub, picked in ((os.path.join(tiles_dir, cls), train),
                            (os.path.join(val_dir, cls), hold)):
            os.makedirs(sub, exist_ok=True)
            n = 0
            for f in picked:
                im = _to_square_tile(f, size)
                if im is None:
                    continue
                im.save(os.path.join(sub, "%s_real_%02d.jpg" % (cls, n)),
                        "JPEG", quality=92)
                n += 1
        stats[cls] = {"train": len(train), "val": len(hold)}
    sc_dir = os.path.join(out, "scenes")
    os.makedirs(sc_dir, exist_ok=True)
    sc_n = 0
    for f in scenes:
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        if min(im.size) < 96:
            continue
        im.save(os.path.join(sc_dir, "scene_%03d.jpg" % sc_n),
                "JPEG", quality=92)
        sc_n += 1
    stats["scenes"] = sc_n
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
    return stats


# ── real object cutouts + composites ─────────────────────────────────────
#
# Many fetched photos are product/stock shots on a near-uniform background.
# Cutting the object OUT and compositing it onto real scene photos teaches
# the classifier background-invariance — the thing plain frame-level
# fine-tuning on a few photos can't give a small CNN.

def estimate_border_bg(im):
    """Median colour + spread of the frame border, or None when the border
    is too textured for a clean cutout."""
    import numpy as np
    a = np.asarray(im, dtype=np.float32)
    b = max(2, min(im.size) // 10)
    border = np.concatenate([
        a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
        a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    med = np.median(border, axis=0)
    dist = np.abs(border - med).sum(1)
    spread = float(np.percentile(dist, 75))
    return med, spread


def cutout_object(path, work=128):
    """RGBA cutout of the object on a uniform-background photo, or None."""
    import numpy as np
    try:
        im = Image.open(path).convert("RGB").resize((work, work),
                                                    Image.LANCZOS)
    except Exception:
        return None
    bg, spread = estimate_border_bg(im)
    if spread > 34:                     # border itself isn't uniform
        return None
    a = np.asarray(im, dtype=np.float32)
    dist = np.abs(a - bg.reshape(1, 1, 3)).sum(2)
    mask = (dist > max(52.0, spread * 2.6)).astype(np.uint8) * 255
    m = Image.fromarray(mask, "L").filter(ImageFilter.MedianFilter(5))
    m = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(5))
    fa = np.asarray(m, dtype=np.float32) / 255.0
    filled = fa.sum()
    if filled < work * work * 0.02:     # nothing found
        return None
    ys, xs = (fa > 0.4).nonzero()
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    fw, fh = (x1 - x0) / work, (y1 - y0) / work
    cover = filled / (work * work)
    if fw < 0.14 or fh < 0.14 or cover > 0.80:   # sliver or frame-filler
        return None
    # background bleed check: the bbox corners of a TRUE cutout are mask
    # background; if they're mostly foreground the "object" is a crop artifact
    corners = [fa[y0, x0], fa[y0, x1], fa[y1, x0], fa[y1, x1]]
    if sum(1 for c in corners if c < 0.5) < 3:
        return None
    rgba = im.convert("RGBA")
    edge = m.filter(ImageFilter.GaussianBlur(1.5))
    rgba.putalpha(edge)
    return rgba.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))


def cutout_for_class(name, rng, max_side=96):
    """Random real-object cutout for the class (None when none cuttables)."""
    files = _class_files(name)
    if not files:
        return None
    key = "cut:" + name
    if key not in _CACHE:
        got = []
        for f in files:
            c = cutout_object(f)
            if c is not None:
                got.append(c)
        _CACHE[key] = got
    got = _CACHE[key]
    return got[rng.randrange(len(got))] if got else None


def composite_for_class(name, rng, size=96):
    """A real object cutout pasted on a real scene photo crop."""
    cut = cutout_for_class(name, rng)
    if cut is None:
        return None
    bg = real_scene(size, rng)
    if bg is None:
        return None
    w, h = cut.size
    scale = rng.uniform(0.4, 0.85) * size / max(w, h)
    nw, nh = max(8, int(w * scale)), max(8, int(h * scale))
    cut = cut.resize((nw, nh), Image.LANCZOS)
    if rng.random() < 0.5:
        cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
    cut = cut.rotate(rng.uniform(-18, 18), resample=Image.BICUBIC,
                     expand=True)
    x0 = rng.randint(0, max(0, size - cut.size[0]))
    y0 = rng.randint(0, max(0, size - cut.size[1]))
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.85, 1.15))
    bg.paste(cut, (x0, y0), cut)
    return bg


def composites(out_tiles, per_class=48, size=96, seed=23):
    """Write composite tiles into <out_tiles>/<class>/comp_NNN.jpg so
    train_models' ordinary tile glob picks them up."""
    import make_dataset as md
    rng0 = random.Random(seed)
    wrote = 0
    report = {}
    for name in md.CLASSES:
        d = os.path.join(out_tiles, name)
        if not os.path.isdir(d):
            continue
        n = 0
        for i in range(per_class):
            rng = random.Random("%d|comp|%s|%d" % (seed, name, i))
            img = composite_for_class(name, rng, size)
            if img is None:
                break
            img.save(os.path.join(d, "comp_%03d.jpg" % i),
                     "JPEG", quality=91)
            n += 1
        report[name] = n
        wrote += n
    print("composite tiles written: %d" % wrote)
    print(json.dumps(report, indent=0, sort_keys=True))
    return wrote


# ── runtime helpers (make_challenges) ─────────────────────────────────────

_CACHE = {}


def real_tiles_dir():
    d = os.path.join(REAL_DIR, "tiles")
    return d if os.path.isdir(d) else None


def _class_files(name):
    if name not in _CACHE:
        d = real_tiles_dir()
        got = sorted(glob.glob(os.path.join(d, name, "*.jpg"))) if d else []
        _CACHE[name] = got
    return _CACHE[name]


def real_frac(name):
    """0..1 — how real-photo-backed this class is (0 = no photos)."""
    return min(1.0, len(_class_files(name)) / 6.0)


def real_tile(name, rng, size=96):
    """A random REAL photo tile for the class, lightly jittered; None when
    the class has no photos."""
    files = _class_files(name)
    if not files:
        return None
    path = files[rng.randrange(len(files))]
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    # vary crop a touch so 3 photos act like 12
    w, h = im.size
    jx = rng.uniform(-0.06, 0.06) * w
    jy = rng.uniform(-0.06, 0.06) * h
    crop = im.crop((int(w * 0.05 + jx), int(h * 0.05 + jy),
                    int(w * 0.95 + jx), int(h * 0.95 + jy))).resize(
                        (size, size), Image.LANCZOS)
    crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.88, 1.12))
    crop = ImageEnhance.Color(crop).enhance(rng.uniform(0.85, 1.20))
    return crop


_SCENES = None


def real_scene(size, rng):
    """A random real background photo crop (size x size), or None."""
    global _SCENES
    if _SCENES is None:
        _SCENES = sorted(glob.glob(os.path.join(REAL_DIR, "scenes", "*.jpg")))
    if not _SCENES:
        return None
    path = _SCENES[rng.randrange(len(_SCENES))]
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    side = min(w, h)
    x0 = rng.randint(0, w - side) if w > side else 0
    y0 = rng.randint(0, h - side) if h > side else 0
    im = im.crop((x0, y0, x0 + side, y0 + side))
    return im.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="organize",
                    choices=["organize", "composites"])
    ap.add_argument("--holdout", type=int, default=2)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--per_class", type=int, default=48)
    ap.add_argument("--tiles", default=os.path.join(ROOT, "data_v2", "tiles"))
    a = ap.parse_args()
    if a.cmd == "composites":
        composites(a.tiles, per_class=a.per_class, size=a.size)
        return
    stats = organize(holdout=a.holdout, size=a.size)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
