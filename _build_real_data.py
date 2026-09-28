#!/usr/bin/env python3
"""
_build_real_data.py — one-shot builder that turns image-search downloads
(workspace dir ``image-search/``) into the committed real-photo dataset
under ``data/<class>/``.

Each downloaded photo is center-cropped to a square and resized to the tile
size (default 96 px), then written as ``data/<class>/<class>_NNNNN.jpg``.
The manifest (``data/manifest.jsonl``) and the class contact sheet
(``data/_preview.jpg``) are regenerated. A ``data/README.md`` is written
describing the 60-class real-photo corpus.

The slug -> class map is explicit (ordered, longest-most-specific first)
so query slugs like ``zebra-crossing-road-stripes`` map to ``crosswalk``
rather than ``zebra``.
"""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIR = os.path.join(ROOT, "image-search")
DATA_DIR = os.path.join(ROOT, "data")
TILE = 96

# Pull the canonical 60-class list straight from the generator so the data
# on disk can never drift from the code.
import make_dataset as md  # noqa: E402

CLASSES = list(md.CLASSES)
PROMPTS = dict(md.PROMPTS)
assert len(CLASSES) == 60, "expected 60 classes, got %d" % len(CLASSES)

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# query-slug substring -> class. Most-specific (longest / ambiguous) keys
# first so they win over shorter prefixes.
SLUG_MAP = [
    # --- street furniture (must beat animal names) ---
    ("zebra-crossing", "crosswalk"),
    ("pedestrian-crossing", "crosswalk"),
    ("traffic-signal-green", "traffic_light"),
    ("traffic-light-pole", "traffic_light"),
    ("traffic-light-red", "red_light"),
    ("red-traffic-light", "red_light"),
    ("red-fire-hydrant", "fire_hydrant"),
    ("fire-hydrant", "fire_hydrant"),
    ("parking-meter", "parking_meter"),
    # --- traffic ---
    ("yellow-school-bus", "bus"),
    ("city-bus", "bus"),
    ("sports-car", "car"),
    ("sedan-car", "car"),
    ("semi-truck", "truck"),
    ("delivery-truck", "truck"),
    ("high-speed-train", "train"),
    ("train-locomotive", "train"),
    ("mountain-bike", "bicycle"),
    ("sport-motorcycle", "motorcycle"),
    ("fishing-boat", "boat"),
    ("sailboat", "boat"),
    ("jet-airplane", "airplane"),
    ("passenger-airplane", "airplane"),
    # --- animals ---
    ("dog-photo", "dog"),
    ("cat-kitten", "cat"),
    ("wild-rabbit", "rabbit"),
    ("brown-horse", "horse"),
    ("horse-head", "horse"),
    ("elephant-zoo", "elephant"),
    ("cow-pasture", "cow"),
    ("colorful-bird", "bird"),
    ("tree-frog", "frog"),
    ("sea-turtle", "turtle"),
    ("garden-snail", "snail"),
    ("kangaroo-joey", "kangaroo"),
    ("zebra-animal", "zebra"),
    ("zebra-standing", "zebra"),
    ("giraffe", "giraffe"),
    ("lion", "lion"),
    ("bear", "bear"),
    ("sheep", "sheep"),
    ("duck", "duck"),
    ("mallard", "duck"),
    ("tropical-fish", "fish"),
    ("goldfish", "fish"),
    ("butterfly", "butterfly"),
    ("monarch", "butterfly"),
    # --- tools ---
    ("claw-hammer", "hammer"),
    ("cordless-power-drill", "drill"),
    ("power-drill", "drill"),
    ("hand-saw", "saw"),
    ("paint-brush", "paintbrush"),
    ("crescent-adjustable-wrench", "wrench"),
    ("adjustable-wrench", "wrench"),
    ("screwdriver", "screwdriver"),
    # --- materials ---
    ("stack-of-cut-wood", "wood"),
    ("wooden-planks", "wood"),
    ("steel-nails", "nail"),
    ("metal-nail", "nail"),
    ("metal-wood-screw", "screw"),
    ("wood-screw", "screw"),
    ("hex-bolt", "bolt"),
    ("red-brick-wall", "wall"),
    ("blank-artist-canvas", "canvas"),
    # --- household / terrain / new objects ---
    ("green-apple", "apple"),
    ("red-apple", "apple"),
    ("pizza-slice", "pizza"),
    ("whole-pizza", "pizza"),
    ("office-desk", "table"),
    ("wooden-dining-table", "table"),
    ("single-wooden-chair", "chair"),
    ("wooden-chair", "chair"),
    ("coffee-cup", "cup"),
    ("tea-mug", "cup"),
    ("stack-of-hardcover-books", "book"),
    ("hardcover-book", "book"),
    ("round-wall-clock", "clock"),
    ("alarm-clock", "clock"),
    ("open-colorful-umbrella", "umbrella"),
    ("colorful-umbrella", "umbrella"),
    ("single-green-pine-tree", "tree"),
    ("pine-tree", "tree"),
    ("single-yellow-flower", "flower"),
    ("yellow-flower", "flower"),
    ("suburban-brick-house", "house"),
    ("brick-house", "house"),
    ("snowy-mountain", "mountain"),
    ("mountain-range", "mountain"),
    ("single-leather-boot", "boot"),
    ("hiking-work-boot", "boot"),
    ("hiking-boot", "boot"),
    ("banana", "banana"),
    ("acoustic-guitar", "guitar"),
    ("wooden-guitar", "guitar"),
    ("cactus", "cactus"),
]


def slug_to_class(filename: str):
    base = os.path.basename(filename).lower()
    for key, cls in SLUG_MAP:
        if key in base:
            return cls
    return None


def to_square_tile(path: str, size: int = TILE):
    """Center-crop to a square, then resize. Returns RGB PIL Image or None."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if min(w, h) < 56:
        return None
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    im = im.crop((x0, y0, x0 + side, y0 + side))
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return im


def main():
    if not os.path.isdir(SEARCH_DIR):
        sys.stderr.write("image-search/ not found; nothing to do\n")
        return 1

    # gather downloads
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(SEARCH_DIR, "*" + ext)))
    per_class = {c: [] for c in CLASSES}
    skipped = []
    for f in sorted(files):
        cls = slug_to_class(f)
        if cls is None:
            skipped.append(f)
            continue
        if cls not in per_class:
            skipped.append(f)
            continue
        im = to_square_tile(f)
        if im is not None:
            per_class[cls].append((f, im))

    # deterministic shuffle within each class so the first N are a mix of
    # both query rounds
    rng = random.Random(7)
    for cls in per_class:
        rng.shuffle(per_class[cls])

    # wipe existing class folders (drawn images and any prior real tiles)
    for name in os.listdir(DATA_DIR):
        p = os.path.join(DATA_DIR, name)
        if os.path.isdir(p):
            shutil.rmtree(p)

    manifest = []
    counts = {}
    for cid, cls in enumerate(CLASSES):
        out_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        items = per_class.get(cls, [])
        n = 0
        for _src, im in items:
            fname = "%s_%05d.jpg" % (cls, n)
            im.save(os.path.join(out_dir, fname), "JPEG", quality=90)
            manifest.append({
                "image": "data/%s/%s" % (cls, fname),
                "label": cls,
                "class_id": cid,
                "prompt": PROMPTS[cls],
            })
            n += 1
        counts[cls] = n

    # manifest
    with open(os.path.join(DATA_DIR, "manifest.jsonl"), "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")

    # contact sheet: one column per class, up to 8 rows
    cell = 96
    rows = 8
    sheet = Image.new("RGB", (len(CLASSES) * cell, rows * cell), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    for cx, cls in enumerate(CLASSES):
        paths = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")))[:rows]
        for cy, p in enumerate(paths):
            try:
                im = Image.open(p).convert("RGB")
                sheet.paste(im, (cx * cell, cy * cell))
            except Exception:
                pass
    sheet.save(os.path.join(DATA_DIR, "_preview.jpg"), "JPEG", quality=82)

    # report
    print("classes: %d" % len(CLASSES))
    print("total real tiles: %d" % sum(counts.values()))
    empty = [c for c in CLASSES if counts[c] == 0]
    if empty:
        print("WARNING: classes with NO usable photos: %s" % ", ".join(empty))
    if skipped:
        print("skipped %d unmapped/unsuitable files" % len(skipped))
    for c in CLASSES:
        print("  %-16s %d" % (c, counts[c]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
