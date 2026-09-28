#!/usr/bin/env python3
"""
hcaptcha_types.py — challenge-family router + semantic knowledge base.

hCaptcha serves five challenge families, not one:

  ============================  =======================================
  image_label_binary            "click each image containing a bus"
  binary + reference image      "pick all things you can work on with
                                 the item shown" (tool -> materials)
  area_select (point)           "click on the animal who jumps highest"
  area_select (bbox)            "draw a box around the cat's head"
  image_drag_drop               "drag the element to the place it fits"
  wooden-block tower (drag)     "move the missing block onto the incomplete tower"
  multiple_choice               "select the most accurate description"
  ============================  =======================================

The old solver assumed every round was a tile grid and answered with tile
indices — only family #1 worked. This module decides WHICH family a round
is (three independent tiers, highest-signal first):

  1. classify_from_payload  — request_type inside hCaptcha's /getcaptcha JSON
  2. classify_from_dom      — visible tiles/canvases/draggables/choices
  3. classify_from_prompt   — wording regexes (last resort)

``classify(payload, dom, prompt)`` tries them in that order.

It also carries the small knowledge base that makes relational prompts
solvable *offline*: superlative tables (size/jump/speed/temperature), a
tool-affordance map ("what can you work on with a drill") and category
sets (animals/edible/wheeled/motorised). ``resolve_semantic`` turns a
prompt + per-tile labels into 1-based tile indices without a vision model.
"""

from __future__ import annotations

import re

# ── families + answer shapes ──────────────────────────────────────────────

BINARY = "image_label_binary"
AREA_POINT = "area_select_point"
AREA_BBOX = "area_select_bbox"
DRAG_DROP = "image_drag_drop"
MULTIPLE_CHOICE = "multiple_choice"
TEXT_ENTRY = "text_entry"
COUNT = "counting"
UNKNOWN = "unknown"

ANSWER_SHAPE = {
    BINARY: "tiles",
    AREA_POINT: "points",
    AREA_BBOX: "bbox",
    DRAG_DROP: "drag",
    MULTIPLE_CHOICE: "choice",
    TEXT_ENTRY: "text",
    COUNT: "count",
    UNKNOWN: "tiles",  # safest fallback: the grid clicker also no-ops cleanly
}


def answer_shape(family: str) -> str:
    return ANSWER_SHAPE.get(family, "tiles")


# ── payload helpers (hCaptcha /getcaptcha JSON) ───────────────────────────

def question_text(payload: dict) -> str:
    """requester_question: {"en": "..."} | str | missing."""
    if not isinstance(payload, dict):
        return ""
    rq = payload.get("requester_question")
    if isinstance(rq, dict):
        for key in ("en", "en-US", "en_US"):
            if rq.get(key):
                return str(rq[key])
        for val in rq.values():
            if isinstance(val, str) and val.strip():
                return val
        return ""
    if isinstance(rq, str):
        return rq
    return ""


def example_urls(payload: dict) -> list:
    """requester_question_example: header reference image URL(s), 0..2."""
    if not isinstance(payload, dict):
        return []
    ex = payload.get("requester_question_example")
    out = []
    if isinstance(ex, str):
        out = [ex]
    elif isinstance(ex, (list, tuple)):
        out = [u for u in ex if isinstance(u, str)]
    elif isinstance(ex, dict):
        for val in ex.values():
            if isinstance(val, str):
                out.append(val)
            elif isinstance(val, (list, tuple)):
                out.extend(u for u in val if isinstance(u, str))
    return [u for u in out if u.startswith(("http", "/", "data:"))]


def task_urls(payload: dict) -> list:
    """tasklist[].datapoint_uri — the tile image URLs in reading order."""
    if not isinstance(payload, dict):
        return []
    tasks = payload.get("tasklist") or payload.get("tasks") or []
    out = []
    for t in tasks if isinstance(tasks, (list, tuple)) else []:
        if isinstance(t, dict) and isinstance(t.get("datapoint_uri"), str):
            out.append(t["datapoint_uri"])
    return out


# ── tier 1: classify from the /getcaptcha payload ────────────────────────

_BBOX_WORD_RE = re.compile(
    r"draw (a|the) (bounding )?(box|rectangle)|bounding box|"
    r"box around|drag a box|draw around", re.I)

# Binary-stage wording inside an image_label_area_select payload. hCaptcha
# serves MIXED rounds under that request_type: a tile-grid binary stage
# ("click each image containing a bus") followed by an area stage ("then
# click on the car"). The payload only carries the stage-1 question, so the
# payload tier must NOT commit to area_select for these — it defers to the
# live DOM/prompt tiers, which classify each stage as it appears.
_BINARY_GRID_Q_RE = re.compile(
    r"click (on )?(each|every|all (the )?) (image|photo|picture|tile)s?|"
    r"select all (the )?(images|photos|pictures|tiles)|"
    r"choose all (the )?(images|photos|pictures|tiles)|"
    r"pick (all|every) (the )?(images?|photos?|pictures?)|"
    r"(select|choose|pick|check|mark) (all )?(the )?items|"
    r"items that (are|have|contain)|"
    r"primarily \w+|made (of|from) |"
    r"find (places|surfaces)|setting down|places? (that are )?safe|"
    r"in the reference|"
    r"find items that|find the (item|odd|matching)|"
    r"select all images with|select the image showing", re.I)


# Counting tasks ("How many X are in this image?") — a photo + numeric
# answer options. Routed from the payload, from counting wording with
# option buttons in the DOM, or from the prompt alone.
_COUNT_WORD_RE = re.compile(
    r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal (number|amount)\b|"
    r"\bselect (the )?(correct )?number\b", re.I)

# Pattern-completion drag rounds ("Put one of the animals into the empty
# spot to complete the pattern"): a 3x3 icon grid with ONE empty cell and
# a row of candidate elements; the right candidate completes every row and
# column. These are image_drag_drop under the hood, but the piece is not a
# punched silhouette — it is chosen by the PATTERN, so the router flags
# them and the round solver applies semantic logic instead of the pure
# geometric DragLocator.
_PATTERN_PHRASE_RE = re.compile(
    r"complete the pattern|finish the pattern|"
    r"empty (spot|space|cell)|blank (spot|space|cell)|"
    r"missing (spot|space|cell)|"
    r"fill (the |in )?(the )?(empty|missing|blank) (spot|space|cell)|"
    r"put (one of )?the .{0,40}? into the (empty|blank|missing) "
    r"(spot|space|cell)|"
    r"to (complete|finish) the (pattern|row|sequence|grid)|"
    r"which (one )?(belongs|goes|fits) in the (empty|blank|missing) "
    r"(spot|space|cell)", re.I)


def is_pattern_prompt(prompt: str) -> bool:
    """True when the prompt is a pattern-completion drag round."""
    return bool(_PATTERN_PHRASE_RE.search(prompt or ""))


# Wooden-block tower drag: "Move the correct missing block segment onto
# the incomplete tower". hCaptcha serves this under image_label_area_select
# (so the payload tier used to commit to a POINT click) even though the
# answer is a Move-badge drag onto the shortest / gapped stack.
_TOWER_PHRASE_RE = re.compile(
    r"missing block|block segment|incomplete tower|"
    r"complete the tower|onto the (incomplete )?tower|"
    r"move the .{0,50}(block|segment|tower)|"
    r"missing segment|onto the stack|"
    r"incomplete (stack|column|pillar|spire|structure)|"
    r"unfinished tower|broken tower|truncated tower|"
    r"stack the (missing )?(block|piece)",
    re.I)


def is_tower_prompt(prompt: str) -> bool:
    """True for the wooden-block 'missing segment onto the tower' drag."""
    return bool(_TOWER_PHRASE_RE.search(prompt or ""))


def _bbox_config(payload: dict) -> bool:
    cfg = payload.get("request_config")
    if not isinstance(cfg, dict):
        return False
    blob = " ".join(str(v) for v in cfg.values()).lower()
    return "bbox" in blob or "bounding" in blob or "rectangle" in blob or \
        "\"box\"" in blob


def classify_from_payload(payload: dict) -> str:
    if not isinstance(payload, dict):
        return UNKNOWN
    rt = str(payload.get("request_type") or payload.get("type") or "").lower()
    if not rt:
        return UNKNOWN
    q = question_text(payload)
    if "drag" in rt or "drop" in rt:
        return DRAG_DROP
    if "multiple" in rt or "choice" in rt:
        return MULTIPLE_CHOICE
    if "count" in rt or "number" in rt:
        return COUNT
    if "area_select" in rt or ("area" in rt and "select" in rt):
        # MIXED rounds (binary grid stage, then an area/drag stage) share
        # this request_type and may mention BOTH stages in one question.
        # Defer those so the live DOM/prompt classify each stage. A payload
        # that is ONLY a tower/pattern drag must NOT commit to a point
        # click — that never grabs the Move piece.
        mixed = bool(_BINARY_GRID_Q_RE.search(q))
        if not mixed and (is_tower_prompt(q) or is_pattern_prompt(q)):
            return DRAG_DROP
        if mixed:
            return UNKNOWN
        if _BBOX_WORD_RE.search(q) or _bbox_config(payload):
            return AREA_BBOX
        return AREA_POINT
    if "binary" in rt:
        return BINARY
    if "text" in rt or "entry" in rt or "ocr" in rt:
        return TEXT_ENTRY
    return UNKNOWN


# ── tier 2: classify from the live DOM (DOM_PROBE_JS facts) ──────────────

# JS evaluated inside the challenge frame by server._probe_challenge_dom.
# Counts what's actually visible; the rules below map counts to a family.
DOM_PROBE_JS = r"""() => {
    const vis = (el) => !!(el) &&
        (el.offsetParent !== null || el.getClientRects().length > 0);
    const q = (sel) => Array.from(document.querySelectorAll(sel));
    const count = (sel) => q(sel).filter(vis).length;
    const tiles = count('div.task-image, [class*="task-image" i]');
    const examples = count(
        '.challenge-example, .prompt-image, .challenge-prompt img, ' +
        '[class*="example" i] img, [class*="example" i] div[style*="background"]');
    const canvases = q('canvas').filter(c => vis(c) && c.width > 50).length;
    const bigImages = q('img, [class*="image" i]')
        .filter(el => vis(el) &&
            (el.clientWidth >= 180 || el.naturalWidth >= 180)).length;
    const draggables = count('[draggable="true"], [class*="drag" i]');
    let moveBadge = false;
    for (const el of q('div, span, p, button, [role="button"]')) {
        const t = ((el.textContent || '') + ' '
            + (el.getAttribute('aria-label') || '')).trim();
        if (!vis(el)) continue;
        if (/^\+?\s*move\s*$/i.test(t) || /\bmove\b/i.test(t) && t.length <= 12) {
            moveBadge = true; break;
        }
    }
    const choices = count(
        '.answer-option, [class*="answer-option" i], [class*="choice" i] button, ' +
        '[class*="multiple" i] [role="button"], .options [role="button"]');
    const inputs = count('input[type="text"], input:not([type]), textarea');
    return {
        tiles, examples, canvases, images: bigImages,
        draggables, move_badge: moveBadge, choices, inputs,
    };
}"""


def classify_from_dom(facts: dict, prompt: str = "") -> str:
    """Map DOM fact counts to a family. Rules ordered by confidence."""
    if not isinstance(facts, dict) or not facts:
        return UNKNOWN

    def n(key):
        try:
            return max(0, int(facts.get(key, 0) or 0))
        except Exception:
            return 0

    tiles = n("tiles")
    # A draggable element / "Move" badge with at most one tile is the
    # drag-drop challenge (the piece is the draggable).
    if (n("draggables") > 0 or facts.get("move_badge")) and tiles <= 1:
        return DRAG_DROP
    # Pattern-completion rounds: a 3x3 grid of tiles PLUS a row of
    # draggable candidates — the DOM shows many tiles, which would
    # otherwise fall through to binary. The wording disambiguates.
    if _PATTERN_PHRASE_RE.search(prompt or "") and (
            n("draggables") > 0 or facts.get("move_badge") or tiles >= 6):
        return DRAG_DROP
    # Wooden-block tower: one canvas, often without a leaf "Move" text
    # node (the badge is "+ Move" / an icon child). Live wording is the
    # signal — do NOT fall through to AREA_POINT.
    if is_tower_prompt(prompt):
        return DRAG_DROP
    # A genuine tile grid is the binary family (hCaptcha grids are 9+ tiles;
    # 4 tolerated for odd layouts).
    if tiles >= 4:
        return BINARY
    # Several option buttons with no grid is multiple choice — unless the
    # wording asks "how many", which is a counting task (numeric options).
    if n("choices") >= 2 and tiles <= 1:
        if _COUNT_WORD_RE.search(prompt or ""):
            return COUNT
        return MULTIPLE_CHOICE
    # A bare text field is a text-entry (OCR) challenge.
    if n("inputs") >= 1 and tiles == 0 and n("canvases") == 0 \
            and n("images") == 0:
        return TEXT_ENTRY
    # One big surface to click on: area_select. Wording splits point vs bbox.
    if tiles == 1 or n("canvases") >= 1 or n("images") >= 1:
        if _BBOX_WORD_RE.search(prompt or ""):
            return AREA_BBOX
        return AREA_POINT
    return UNKNOWN


# ── tier 3: classify from the prompt wording ─────────────────────────────

_PROMPT_RULES = (
    (DRAG_DROP, re.compile(
        r"\bdrag\b|where it fits|place where it belongs|puzzle piece|"
        r"complete the puzzle|missing piece|empty space|matching slot|"
        r"matching outline|move the (piece|element|shape|tile|correct)|"
        r"missing block|block segment|incomplete tower|complete the tower|"
        r"onto the (incomplete )?tower|missing segment|"
        r"complete the pattern|empty (spot|cell)|into the empty "
        r"(spot|space|cell)|fill the (empty|missing|blank) "
        r"(spot|space|cell)|"
        r"\b(rotate|flip|slide|stack|fit|nest|join|attach|align|drop)\b.{0,40}"
        r"(piece|block|segment|tile|shape|part)|"
        r"place the (correct |missing )?(piece|block|tile|shape|part)|"
        r"onto the incomplete|into the (empty|outlined|corresponding) "
        r"(slot|space|gap|area)", re.I)),
    (BINARY, re.compile(
        r"click each image|click (on )?every (image|photo|picture|tile)|"
        r"select all (the )?(images|pictures|photos|tiles)|"
        r"choose all (the )?(images|pictures|photos|tiles)|"
        r"pick (all|every) (the )?(images?|pictures?|photos?|tiles?)|"
        r"check all (the )?(images|pictures|photos|tiles)|"
        r"mark (all|every) (the )?(images|pictures|photos|tiles)|"
        r"click (all )?(the )?images (containing|with)|"
        r"images? (containing|with|of) a|all (the )?squares with|"
        r"(select|choose|pick|check|mark) (all )?(the )?items|"
        r"items that (are|have|contain)|objects that (are|have)|"
        r"primarily \w+|made (of|from) |"
        r"find (places|surfaces)|setting down|places? (that are )?safe|"
        r"find items that|find the (item|odd|matching)|"
        r"select the image showing|"
        r"(two|2|pair).{0,60}(identical|same|matching|duplicate|alike|similar) "
        r"(images|pictures|photos|tiles)|matching pair|"
        r"most similar (images|pictures|photos|tiles)", re.I)),
    (COUNT, _COUNT_WORD_RE),
    (AREA_BBOX, _BBOX_WORD_RE),
    (MULTIPLE_CHOICE, re.compile(
        r"most accurate|best (describes|description)|which (of these|one)|"
        r"(choose|select) the (correct|best|right) (answer|option|description)", re.I)),
    (TEXT_ENTRY, re.compile(
        r"\btype\b|enter the (characters|text|letters)|"
        r"characters (you see|in the image)", re.I)),
    (AREA_POINT, re.compile(
        r"\bclick on\b|\btap on\b|click the (animal|object|item|element|largest|"
        r"smallest|fastest|slowest|highest)|click anywhere on|"
        r"place (a|the) (point|dot)|"
        r"(two|2|pair).{0,80}(identical|same|matching|duplicate|alike|similar) "
        r"(elements?|objects?|items?)|"
        r"most similar (elements?|objects?|items?)", re.I)),
)


def classify_from_prompt(prompt: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return UNKNOWN
    for family, rx in _PROMPT_RULES:
        if rx.search(p):
            return family
    return UNKNOWN


def classify(payload: dict = None, dom: dict = None, prompt: str = "") -> str:
    """All three tiers in order; first non-UNKNOWN wins.

    The live prompt wins for the wooden-block tower: hCaptcha wraps that
    Move-badge drag in ``image_label_area_select``, so a stale mixed
    payload + a single canvas would otherwise commit to a point click.
    """
    if is_tower_prompt(prompt):
        return DRAG_DROP
    if is_setdown_prompt(prompt):
        return BINARY
    for result in (classify_from_payload(payload),
                   classify_from_dom(dom, prompt),
                   classify_from_prompt(prompt)):
        if result != UNKNOWN:
            return result
    return UNKNOWN


# ── knowledge base ────────────────────────────────────────────────────────
#
# Ranks are arbitrary-but-consistent scales. The synthetic challenge
# generator (make_challenges.py) labels relational rounds from THE SAME
# tables, so the offline models and the router stay in lockstep by
# construction.

SIZE_RANK = {
    "nail": 1, "screw": 2, "bolt": 3, "butterfly": 3, "snail": 4, "flower": 5,
    "fish": 5, "apple": 6, "banana": 7, "cup": 7, "frog": 8, "bird": 9,
    "duck": 10, "rabbit": 10, "book": 11, "clock": 12, "turtle": 13,
    "cactus": 14, "boot": 14, "cat": 15, "umbrella": 16, "pizza": 17,
    "dog": 18, "guitar": 19, "chair": 19, "kangaroo": 20, "bicycle": 21,
    "table": 22, "sheep": 23, "motorcycle": 23, "cow": 24, "zebra": 25,
    "car": 25, "lion": 26, "bear": 27, "horse": 26, "boat": 27, "truck": 28,
    "bus": 29, "giraffe": 30, "train": 30, "airplane": 31, "elephant": 32,
    "tree": 33, "house": 34, "mountain": 35,
    # tools / street furniture / surfaces (relational "largest/smallest"
    # rounds over tools or urban objects)
    "screwdriver": 5, "paintbrush": 6, "wrench": 7, "hammer": 8,
    "drill": 9, "saw": 10, "wood": 11, "fire_hydrant": 13,
    "parking_meter": 14, "traffic_light": 15, "red_light": 15,
    "canvas": 17, "crosswalk": 21, "wall": 31,
}

JUMP_RANK = {
    "snail": 1, "fish": 1, "turtle": 2, "elephant": 3, "cow": 4, "sheep": 5,
    "duck": 5, "horse": 6, "bear": 6, "dog": 7, "giraffe": 7, "butterfly": 8,
    "cat": 8, "bird": 8, "rabbit": 9, "zebra": 9, "frog": 10, "lion": 10,
    "kangaroo": 11,
}

SPEED_RANK = {
    "snail": 1, "turtle": 2, "frog": 3, "butterfly": 4, "elephant": 5,
    "cow": 5, "fish": 6, "rabbit": 6, "dog": 7, "kangaroo": 8, "duck": 8,
    "bird": 9, "cat": 9, "horse": 10, "bicycle": 11, "sheep": 11, "motorcycle": 12,
    "bear": 12, "car": 13, "giraffe": 13, "bus": 14, "truck": 15, "zebra": 15,
    "boat": 16, "lion": 16, "train": 17, "airplane": 18,
}

TEMP_RANK = {
    "mountain": 1, "fish": 2, "flower": 3, "apple": 4, "tree": 4, "banana": 5,
    "boot": 5, "house": 5, "umbrella": 5, "cactus": 9, "cup": 8, "pizza": 9,
    # animals by habitat warmth ("coldest/hottest place" rounds)
    "snail": 2, "turtle": 3, "frog": 4, "bear": 5, "sheep": 6, "cow": 6,
    "horse": 6, "rabbit": 6, "duck": 6, "dog": 7, "bird": 7,
    "butterfly": 7, "cat": 8, "kangaroo": 9, "zebra": 9, "giraffe": 9,
    "elephant": 9, "lion": 10,
}

# What you can "work on" with each tool (reference-image affordance grids).
TOOL_AFFORDANCE = {
    "drill": {"wood", "wall", "table", "chair", "house"},
    "hammer": {"nail", "wood", "wall"},
    "saw": {"wood", "tree"},
    "wrench": {"bolt", "bicycle", "car", "truck"},
    "paintbrush": {"wall", "canvas", "house"},
    "screwdriver": {"screw", "wood"},
}

ANIMALS = {"dog", "cat", "rabbit", "horse", "elephant", "cow", "bird",
           "frog", "turtle", "snail", "kangaroo", "zebra", "giraffe", "lion",
           "bear", "sheep", "duck", "fish", "butterfly"}
EDIBLE = {"apple", "pizza", "banana", "fish"}
WHEELED = {"car", "bus", "truck", "bicycle", "motorcycle"}
MOTORISED = {"car", "bus", "truck", "motorcycle", "boat", "airplane", "train"}
TOOLS = set(TOOL_AFFORDANCE)
MATERIALS = {"wood", "nail", "screw", "bolt", "wall", "canvas"}

# Dominant-material / attribute sets for "Select items that are primarily
# metal" and the sibling prompts hCaptcha serves on the same grid family.
# Conservative: only classes whose MAIN subject is that material at tile
# scale. Unknown materials (plastic, glass, fabric, colour, …) stay
# unmapped so resolve_semantic returns None and the vision model answers.
METAL = {
    "nail", "screw", "bolt",
    "hammer", "wrench", "screwdriver", "drill", "saw",
    "fire_hydrant", "parking_meter", "traffic_light", "red_light",
    "bicycle", "motorcycle", "car", "bus", "truck", "train", "airplane",
    "clock",
}
WOODEN = {"wood", "table", "chair", "guitar"}
FURRY = {
    "dog", "cat", "rabbit", "horse", "cow", "kangaroo",
    "zebra", "giraffe", "lion", "bear", "sheep",
}
PLANTS = {"tree", "flower", "cactus"}

# Synonyms -> canonical class name. Keys are lowercase, spaces as "_".
# NB: "red_light" and "traffic_light" are deliberately kept exclusive —
# red_light must never canonicalise to traffic_light (opposite labels).
SYNONYMS = {
    "bus": "bus", "car": "car", "auto": "car", "automobile": "car",
    "truck": "truck", "lorry": "truck", "train": "train",
    "bicycle": "bicycle", "bike": "bicycle", "motorcycle": "motorcycle",
    "motorbike": "motorcycle", "scooter": "motorcycle",
    "boat": "boat", "ship": "boat", "airplane": "airplane",
    "aeroplane": "airplane", "plane": "airplane", "jet": "airplane",
    "traffic_light": "traffic_light", "stoplight": "traffic_light",
    "red_light": "red_light", "crosswalk": "crosswalk",
    "zebra_crossing": "crosswalk",
    "fire_hydrant": "fire_hydrant", "hydrant": "fire_hydrant",
    "parking_meter": "parking_meter",
    "dog": "dog", "puppy": "dog", "doggy": "dog",
    "cat": "cat", "kitten": "cat", "kitty": "cat",
    "rabbit": "rabbit", "bunny": "rabbit", "hare": "rabbit",
    "horse": "horse", "pony": "horse", "elephant": "elephant",
    "cow": "cow", "bird": "bird", "frog": "frog", "toad": "frog",
    "turtle": "turtle", "tortoise": "turtle", "snail": "snail",
    "kangaroo": "kangaroo", "roo": "kangaroo",
    "hammer": "hammer", "drill": "drill", "saw": "saw",
    "paintbrush": "paintbrush", "brush": "paintbrush",
    "wrench": "wrench", "spanner": "wrench", "screwdriver": "screwdriver",
    "wood": "wood", "plank": "wood", "lumber": "wood",
    "nail": "nail", "screw": "screw", "bolt": "bolt",
    "wall": "wall", "wallpaper": "wall", "canvas": "canvas",
    "apple": "apple", "pizza": "pizza", "table": "table",
    "chair": "chair", "cup": "cup", "mug": "cup", "book": "book",
    "clock": "clock", "umbrella": "umbrella", "tree": "tree",
    "flower": "flower", "house": "house", "home": "house",
    "mountain": "mountain", "hill": "mountain", "boot": "boot",
    "shoe": "boot",
    # batch 3 (49 -> 60)
    "zebra": "zebra",
    "giraffe": "giraffe",
    "lion": "lion",
    "bear": "bear", "teddy": "bear",
    "sheep": "sheep", "lamb": "sheep", "ram": "sheep", "ewe": "sheep",
    "duck": "duck", "duckling": "duck", "mallard": "duck",
    "fish": "fish", "goldfish": "fish",
    "butterfly": "butterfly", "monarch": "butterfly",
    "banana": "banana", "plantain": "banana",
    "guitar": "guitar", "acoustic_guitar": "guitar",
    "cactus": "cactus", "cacti": "cactus", "saguaro": "cactus",
    # ── wide coverage aliases ────────────────────────────────────────────
    # These map the long tail of hCaptcha prompt nouns onto the 60 classes
    # the offline models can actually emit, ONLY where the visuals are
    # defensible at tile resolution. Everything else deliberately stays
    # unmapped so the server falls back to the vision model (which reads
    # arbitrary prompt text) instead of trusting a wrong offline answer.
    # aircraft / watercraft
    "seaplane": "airplane", "hydroplane": "airplane",
    "helicopter": "airplane", "copter": "airplane", "aircraft": "airplane",
    "airliner": "airplane",
    "subway": "train", "metro": "train", "tram": "train",
    "streetcar": "train", "locomotive": "train", "monorail": "train",
    "railway": "train", "railcar": "train",
    "sailboat": "boat", "yacht": "boat", "ferry": "boat",
    "canoe": "boat", "kayak": "boat", "rowboat": "boat",
    "speedboat": "boat", "barge": "boat", "catamaran": "boat",
    
    "pickup": "truck", "pickup_truck": "truck", "semi": "truck",
    "semi_truck": "truck", "dump_truck": "truck",
    "garbage_truck": "truck", "fire_truck": "truck",
    "tow_truck": "truck",  "van": "truck",
    "ambulance": "truck",
    "taxi": "car", "police_car": "car", "race_car": "car",
    "racecar": "car", "sedan": "car", "convertible": "car",
    "minivan": "car", 
    "school_bus": "bus", "shuttle": "bus",
    "moped": "motorcycle", 
    # birds (the bird class is a generic bird — fine at tile scale)
    "owl": "bird", "parrot": "bird", "penguin": "bird",
    "chicken": "bird", "rooster": "bird", "hen": "bird",
    "eagle": "bird", "sparrow": "bird", "seagull": "bird",
    "pigeon": "bird", "robin": "bird", "peacock": "bird",
    "flamingo": "bird", "toucan": "bird", "hummingbird": "bird",
    "woodpecker": "bird", "swan": "duck", "goose": "duck",
    # water animals
    "shark": "fish", "dolphin": "fish", "whale": "fish",
    "clownfish": "fish", "salmon": "fish", "trout": "fish",
    "tuna": "fish", 
    # hooved / farm animals
    "deer": "horse", "stag": "horse", "reindeer": "horse",
    "moose": "horse", "elk": "horse", "donkey": "horse",
    "mule": "horse", "camel": "horse",
    "goat": "sheep", "alpaca": "sheep", "llama": "sheep",
    "yak": "cow", "buffalo": "cow", "bison": "cow", "ox": "cow",
    "calf": "cow",
    "panda": "bear", "koala": "bear", "grizzly": "bear",
    "polar_bear": "bear", "geese": "duck",
    "tadpole": "frog",
    # plants / nature
    "palm_tree": "tree", "pine_tree": "tree", "oak": "tree",
    "maple": "tree", "willow": "tree", "birch": "tree",
    "evergreen": "tree", "forest": "tree", "woodland": "tree",
    "jungle": "tree", "rainforest": "tree", "grove": "tree",
    "woods": "tree",
    "rose": "flower", "sunflower": "flower", "tulip": "flower",
    "daisy": "flower", "dandelion": "flower", "lily": "flower",
    "orchid": "flower", "blossom": "flower",
    "volcano": "mountain", "cliff": "mountain", "peak": "mountain",
    "ridge": "mountain", "hill": "mountain",
    # buildings / urban
    "building": "house", "barn": "house", "cabin": "house",
    "cottage": "house", "hut": "house", "shed": "house",
    "apartment": "house", "bungalow": "house", "villa": "house",
    "traffic_signal": "traffic_light", "traffic_lights": "traffic_light",
    "pedestrian_crossing": "crosswalk", "crossing": "crosswalk",
    "zebra_crossing": "crosswalk", "fireplug": "fire_hydrant",
    # furniture / home
    "bench": "chair", "stool": "chair", "sofa": "chair",
    "couch": "chair", "armchair": "chair", "seat": "chair",
    "desk": "table", "counter": "table",
    "pie": "pizza", "quiche": "pizza",
    "teacup": "cup", "drinking_glass": "cup", "bucket": "cup",
    "coffee_mug": "cup", "coffee_cup": "cup", "tea_cup": "cup",
    "nightstand": "table", "night_stand": "table", "dresser": "table",
    "bedside_table": "table", "end_table": "table", "coffee_table": "table",
    "deck": "wood", "floorboard": "wood", "floorboards": "wood",
    "hardwood": "wood", "wooden_deck": "wood", "picnic_table": "table",
    "leaf": "flower", "maple_leaf": "flower", "leaves": "flower",
    "magazine": "book", "notebook": "book",
    "watch": "clock", "alarm_clock": "clock", "wall_clock": "clock",
    "grandfather_clock": "clock", "cuckoo_clock": "clock",
    "stopwatch": "clock", "timer": "clock",
    "nut": "bolt",
}


def canonical(word: str):
    """Map a surface word/phrase to the canonical class name (or None)."""
    if not word:
        return None
    key = re.sub(r"[^a-z]+", "_", word.lower()).strip("_")
    if key in SYNONYMS:
        return SYNONYMS[key]
    # naive singular/plural
    if key.endswith("s") and key[:-1] in SYNONYMS:
        return SYNONYMS[key[:-1]]
    return None


_PHRASES = sorted(SYNONYMS.keys(), key=len, reverse=True)


def extract_target(prompt: str):
    """Pull the canonical noun a prompt is about (longest match wins).

    Matches phrases in the prompt text with the common plural endings
    ("pandas" → panda, "boxes" → box, "buses" → bus)."""
    p = " %s " % re.sub(r"[^a-z ]+", " ", (prompt or "").lower())
    for phrase in _PHRASES:
        stem = " %s" % phrase.replace("_", " ")
        for form in ("%s " % stem, "%ss " % stem, "%ses " % stem):
            if form in p:
                return SYNONYMS[phrase]
    return None


def category_of(name: str):
    c = canonical(name) or name
    for label, s in (("animals", ANIMALS), ("edible", EDIBLE),
                     ("wheeled", WHEELED), ("motorised", MOTORISED),
                     ("tools", TOOLS), ("materials", MATERIALS)):
        if c in s:
            return label
    return None


def resolve_pattern(grid_labels, hole_index, candidates):
    """Pattern completion: which candidate completes the 3x3 grid?

    ``grid_labels``: 9 labels in reading order; ``grid_labels[hole_index]``
    is the empty cell (its value is ignored). ``candidates``: list of
    labels for the candidate elements (reading order). Returns the index
    INTO ``candidates`` that makes every row and every column contain
    three distinct labels (a Latin square), or None when no candidate or
    more than one does. Tries the full Latin-square rule first, then the
    rows-only rule (some hCaptcha patterns only constrain rows).

    This is the semantic core of "put one of the animals into the empty
    spot to complete the pattern": it never trusts a guess — ambiguity
    returns None so the caller falls back to the vision model.
    """
    if not isinstance(grid_labels, (list, tuple)) or len(grid_labels) != 9:
        return None
    if not isinstance(hole_index, int) or not (0 <= hole_index < 9):
        return None
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return None
    winners_full, winners_rows = [], []
    for ci, cand in enumerate(candidates):
        if cand is None:
            continue
        labels = list(grid_labels)
        labels[hole_index] = cand
        rows = [labels[r * 3:(r + 1) * 3] for r in range(3)]
        cols = [[labels[r * 3 + c] for r in range(3)] for c in range(3)]
        if None in labels:
            continue
        rows_ok = all(len(set(r)) == 3 for r in rows)
        if rows_ok and all(len(set(c)) == 3 for c in cols):
            winners_full.append(ci)
        elif rows_ok:
            winners_rows.append(ci)
    winners = winners_full or winners_rows
    return winners[0] if len(winners) == 1 else None


_SUPERLATIVE_RULES = (
    (re.compile(r"jumps?\b.*\b(lowest|least)", re.I), "JUMP", "min"),
    (re.compile(r"jumps?\b.*\b(highest|best)|best jumper|highest jump", re.I),
     "JUMP", "max"),
    (re.compile(r"\b(largest|biggest|greatest)\b", re.I), "SIZE", "max"),
    (re.compile(r"\b(smallest|tiniest|littlest)\b", re.I), "SIZE", "min"),
    (re.compile(r"\b(fastest|quickest|speediest)\b", re.I), "SPEED", "max"),
    (re.compile(r"\b(slowest|most slowly)\b", re.I), "SPEED", "min"),
    (re.compile(r"\b(hottest|warmest)\b", re.I), "TEMP", "max"),
    (re.compile(r"\b(coldest|coolest)\b", re.I), "TEMP", "min"),
)

_TABLES = {"JUMP": JUMP_RANK, "SIZE": SIZE_RANK,
           "SPEED": SPEED_RANK, "TEMP": TEMP_RANK}


def superlative_table(prompt: str):
    """"largest"/"jumps the highest"/"slowest" -> (rank_table, 'max'|'min')."""
    p = prompt or ""
    for rx, table, direction in _SUPERLATIVE_RULES:
        if rx.search(p):
            return _TABLES[table], direction
    return None


# ── offline answer resolution ────────────────────────────────────────────

# Visual-pair rounds: prompts like “Please click on the two elements that are
# identical”, “select the matching pair”, or “choose the two same pictures”.
# The vision model handles the image comparison; this regex lets the offline
# label path solve a clean duplicate-label grid without guessing.
_IDENTICAL_PAIR_PHRASE = re.compile(
    r"\b(two|2|pair|matching pair)\b.{0,80}\b"
    r"(identical|same|matching|match|duplicate|alike|similar)\b|"
    r"\b(identical|same|matching|duplicate|alike|similar)\b.{0,80}\b"
    r"(two|2|pair|elements?|objects?|items?|images?|pictures?|tiles?)\b|"
    r"which (two|2|pair).{0,80}(match|are alike|are similar)|"
    r"most similar (two|2|pair|elements?|objects?|items?|images?|pictures?|tiles?)",
    re.I)

_AFFORDANCE_PHRASE = re.compile(
    r"work on|use (the )?(item|tool|object)|item shown|shown in the "
    r"(image|picture|example)|goes with|used (with|on|together)|"
    r"fit(s)? with|belongs? with|associated with", re.I)

# Live: "Find places safe for setting down the item in the reference"
# Tight on purpose — "place where it fits" is a DRAG prompt, and bare
# "in the reference" matches every reference-image affordance grid.
_SETDOWN_PHRASE = re.compile(
    r"setting down|"
    r"set(?:ting)? (?:it |the (?:item|object) )?down|"
    r"places? (?:that are )?safe|"
    r"safe (?:places?|surfaces?)|"
    r"safe (?:to |for )(?:set|put|place|rest|leave)|"
    r"find places|"
    r"could be (?:safely )?(?:stored|set|put|placed|left)|"
    r"where (?:the )?(?:reference )?(?:item|object) could|"
    r"place to (?:put|set|rest|leave|store) (?:the |this )?(?:item|object|it)|"
    r"surfaces? (?:safe |suitable )?(?:for|to)",
    re.I)

# Horizontal furniture / lumber the 60-class CNN can actually emit.
# house/wall are vertical or whole buildings — not a mug-safe surface.
FLAT_SURFACES = {"table", "chair", "wood"}

_REF_COMPARE_RULES = (
    (re.compile(r"\b(larger|bigger|greater|heavier|taller|wider)\b", re.I),
     "SIZE", "gt"),
    (re.compile(r"\b(smaller|tiniest|shorter|narrower|lighter)\b", re.I),
     "SIZE", "lt"),
    (re.compile(r"\b(faster|quicker|speedier)\b", re.I), "SPEED", "gt"),
    (re.compile(r"\b(slower)\b", re.I), "SPEED", "lt"),
    (re.compile(r"\b(hotter|warmer)\b", re.I), "TEMP", "gt"),
    (re.compile(r"\b(colder|cooler)\b", re.I), "TEMP", "lt"),
)

_EXAMPLE_CAT_PHRASE = re.compile(
    r"same (kind|category|type|group)|similar(ly)?|like the (one|item) shown|"
    r"belong(s|ing)? (with|together)|matching the (item|image|example)", re.I)

_SET_PREDICATES = (
    (re.compile(r"\banimals?\b", re.I), ANIMALS),
    (re.compile(r"\b(food|edible|eat|snack)\w*\b", re.I), EDIBLE),
    (re.compile(r"\bwheels?\b|\bwheeled\b", re.I), WHEELED),
    (re.compile(r"\bmotor\w*\b|\bengine\w*\b", re.I), MOTORISED),
    (re.compile(r"\b(vehicles?|transport(ation)?)\b", re.I),
     WHEELED | MOTORISED),
    (re.compile(r"\btools?\b", re.I), TOOLS),
)

# "Select items that are primarily metal / made of wood / have fur" — the
# live wording hCaptcha uses for attribute grids. Matched BEFORE the
# plain-noun extractor so "primarily wood" does not collapse to the single
# class `wood` and miss wooden chairs/tables/guitars.
_ATTRIBUTE_PROMPT_RE = re.compile(
    r"primarily\s+\w+|"
    r"made (?:of|from)\s+\w+|"
    r"(?:items?|objects?)\s+that\s+(?:are|have|contain)|"
    r"(?:select|choose|pick|check|mark)\s+(all\s+)?(the\s+)?items|"
    r"\b(?:has|have|with)\s+fur\b",
    re.I)

_ATTRIBUTE_SETS = (
    (re.compile(r"\bmetal(lic|s)?\b", re.I), METAL),
    (re.compile(r"\bwood(en|s)?\b|\blumber\b|\bplanks?\b", re.I), WOODEN),
    (re.compile(r"\bfur(ry)?\b|\bhair(y)?\b", re.I), FURRY),
    (re.compile(r"\bplants?\b|\bvegetation\b|\bfoliage\b", re.I), PLANTS),
    (re.compile(r"\banimals?\b", re.I), ANIMALS),
    (re.compile(r"\b(food|edible)\b", re.I), EDIBLE),
    (re.compile(r"\b(vehicles?|transport(ation)?)\b", re.I),
     WHEELED | MOTORISED),
    (re.compile(r"\btools?\b", re.I), TOOLS),
)


def is_attribute_prompt(prompt: str) -> bool:
    """True for material/attribute grids ('select items that are primarily metal')."""
    return bool(_ATTRIBUTE_PROMPT_RE.search(prompt or ""))


def is_setdown_prompt(prompt: str) -> bool:
    """True for 'find places safe for setting down the item in the reference'."""
    return bool(_SETDOWN_PHRASE.search(prompt or ""))


def attribute_members(prompt: str):
    """Class set for a material/attribute prompt, or None if unknown.

    None means either the prompt is not an attribute grid, or it is one
    whose material (plastic, glass, colour, …) is not defensible from the
    60 offline classes — the caller should ask the vision model.
    """
    p = prompt or ""
    if not _ATTRIBUTE_PROMPT_RE.search(p):
        return None
    for rx, members in _ATTRIBUTE_SETS:
        if rx.search(p):
            return members
    return None


def resolve_semantic(prompt: str, tile_labels, example_label=None):
    """Prompt + per-tile canonical labels -> 1-based indices, offline.

    Returns:
      list[int]  understood prompt (possibly legitimately empty — a real
                 zero-match round)
      None       not understood; caller should fall back to the vision model
    """
    p = (prompt or "").strip()
    labels = [canonical(l) or str(l or "").strip().lower()
              for l in (tile_labels or [])]
    example = canonical(example_label) if example_label else None

    if not p and example is None:
        return None

    # 0) exact visual-pair wording on a grid. If the classifier labels reveal
    # exactly one duplicate pair, click that pair. Ambiguous duplicate sets
    # deliberately fall through to the vision model.
    if _IDENTICAL_PAIR_PHRASE.search(p):
        counts = {}
        for lab in labels:
            if lab:
                counts[lab] = counts.get(lab, 0) + 1
        pairs = [lab for lab, n in counts.items() if n == 2]
        if len(pairs) == 1:
            want = pairs[0]
            return [i for i, lab in enumerate(labels, 1) if lab == want]
        # If every matching label count is not a clean single pair, the prompt
        # is understood but needs visual comparison (colour/pose/details), so
        # defer rather than clicking a wrong duplicate category.
        return None

    # 1) superlatives ("click on the animal who jumps the highest")
    sup = superlative_table(p)
    if sup is not None:
        table, direction = sup
        best_i, best_v = None, None
        for i, lab in enumerate(labels, 1):
            v = table.get(lab)
            if v is None:
                continue
            if best_v is None or (v > best_v if direction == "max"
                                  else v < best_v):
                best_i, best_v = i, v
        return [best_i] if best_i is not None else []

    # 1.5) "Find places safe for setting down the item in the reference"
    # — click tiles that are STABLE HORIZONTAL SURFACES (table/chair/wood),
    # not the object itself and not a balloon / leaf / ball. Must run
    # before tool-affordance: "item in the reference" used to miss.
    # Zero surfaces → None so the vision model answers (an empty Verify
    # almost never wins this family).
    if is_setdown_prompt(p):
        skip = {example} if example else set()
        idx = [i for i, lab in enumerate(labels, 1)
               if lab in FLAT_SURFACES and lab not in skip]
        return idx if idx else None

    # 1.6) comparative vs the reference ("larger than the item shown")
    if example:
        for rx, table_name, side in _REF_COMPARE_RULES:
            if rx.search(p) and ("than" in p.lower() or "the reference" in p.lower()
                                 or "item shown" in p.lower()):
                table = _TABLES[table_name]
                ev = table.get(example)
                if ev is None:
                    return None
                if side == "gt":
                    return [i for i, lab in enumerate(labels, 1)
                            if table.get(lab) is not None and table[lab] > ev]
                return [i for i, lab in enumerate(labels, 1)
                        if table.get(lab) is not None and table[lab] < ev]

    # 2) tool affordance ("things you can work on with the item shown")
    if _AFFORDANCE_PHRASE.search(p):
        tool = extract_target(p)
        if tool not in TOOL_AFFORDANCE and example in TOOL_AFFORDANCE:
            tool = example
        if tool in TOOL_AFFORDANCE:
            allowed = TOOL_AFFORDANCE[tool]
            return [i for i, lab in enumerate(labels, 1) if lab in allowed]
        return None  # an affordance phrasing with no known tool: defer

    # 3) same category as the reference example
    if example and _EXAMPLE_CAT_PHRASE.search(p):
        wanted = category_of(example)
        if wanted:
            members = {"animals": ANIMALS, "edible": EDIBLE,
                       "wheeled": WHEELED, "motorised": MOTORISED,
                       "tools": TOOLS, "materials": MATERIALS}[wanted]
            return [i for i, lab in enumerate(labels, 1) if lab in members]

    # 3.5) material / attribute grids ("Select items that are primarily
    # metal", "made of wood", "have fur"). Must run BEFORE the plain-noun
    # extractor: "primarily wood" would otherwise collapse to the single
    # class `wood` and miss wooden chairs/tables. Unknown materials
    # (plastic, glass, colour, …) return None so the vision model answers.
    if is_attribute_prompt(p):
        members = attribute_members(p)
        if members is not None:
            return [i for i, lab in enumerate(labels, 1) if lab in members]
        return None

    # 4) set predicates ("click each image containing an animal")
    for rx, s in _SET_PREDICATES:
        if rx.search(p):
            idx = [i for i, lab in enumerate(labels, 1) if lab in s]
            return idx

    # 5) plain noun target ("each image containing a bus")
    target = extract_target(p)
    if target:
        return [i for i, lab in enumerate(labels, 1) if lab == target]

    return None


# ── geometry helpers shared with the server ───────────────────────────────

def denorm(point, box):
    """Normalised (0..1) point -> page coordinates inside a bounding box.

    ``box`` is a Playwright-style {"x","y","width","height"} dict (or a
    4-tuple). Values outside 0..1 are clamped INTO the box.
    """
    if isinstance(box, dict):
        left = float(box.get("x", 0.0))
        top = float(box.get("y", 0.0))
        w = float(box.get("width", 0.0))
        h = float(box.get("height", 0.0))
    else:
        left, top, w, h = (float(v) for v in box)
    try:
        x, y = float(point[0]), float(point[1])
    except Exception:
        x, y = 0.5, 0.5
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return (left + x * w, top + y * h)


# ── wooden-block tower drag (offline heuristic) ──────────────────────────

def _is_wood_rgb(r, g, b) -> bool:
    """True for warm brown/tan wooden-block pixels, not cream backgrounds.

    Broad on purpose: live hCaptcha towers are *photographs* of pine
    blocks, so highlights go almost white-yellow and shadows go dark.
    """
    try:
        r, g, b = int(r), int(g), int(b)
    except (TypeError, ValueError):
        return False
    if r < 50 or r > 252:
        return False
    if g < 22 or g > 235:
        return False
    if b > 185:
        return False
    if r < g - 8:
        return False
    if (r - b) < 16:
        return False
    mx, mn = max(r, g, b), min(r, g, b)
    if mx > 228 and (mx - mn) < 32:
        return False
    if r > 232 and g > 218 and b > 195:
        return False
    return True


def _is_teal_rgb(r, g, b) -> bool:
    """Cyan / teal Move-badge pixels."""
    try:
        r, g, b = int(r), int(g), int(b)
    except (TypeError, ValueError):
        return False
    return g > 70 and b > 70 and g > r + 16 and b > r + 6


def _rgb_grid(image):
    """Normalise an image to ``(width, height, rows)`` of RGB triples.

    Accepts a list-of-rows (stdlib, used by tests), a PIL Image, or
    PNG/JPEG bytes (production screenshots). Returns None when the
    surface is unreadable or tiny.
    """
    if image is None:
        return None
    if isinstance(image, list) and image and isinstance(image[0], (list, tuple)):
        h = len(image)
        w = len(image[0]) if h else 0
        if w < 8 or h < 8:
            return None
        return w, h, image
    try:
        from PIL import Image as _Im
        import io as _io
        if hasattr(image, "convert") and hasattr(image, "size"):
            im = image.convert("RGB")
        elif isinstance(image, (bytes, bytearray)):
            im = _Im.open(_io.BytesIO(bytes(image))).convert("RGB")
        else:
            return None
        w, h = im.size
        if w < 8 or h < 8:
            return None
        pix = list(im.getdata())
        rows = [pix[y * w:(y + 1) * w] for y in range(h)]
        return w, h, rows
    except Exception:
        return None


def _x_clusters(mask, w, h, minc, min_n):
    """Horizontal runs of columns that have enough mask pixels."""
    xcount = [0] * w
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x]:
                xcount[x] += 1
    clusters = []
    i = 0
    while i < w:
        if xcount[i] >= minc:
            j = i
            sx = n = 0
            while j < w and xcount[j] >= minc:
                sx += j * xcount[j]
                n += xcount[j]
                j += 1
            if n >= min_n:
                clusters.append((i, j - 1, sx / float(max(1, n)), n))
            i = j
        else:
            i += 1
    return clusters


def _measure_stack(mask, x0, x1, cx, n, w, h):
    """Vertical occupancy of one x-cluster → tower / piece metrics."""
    width = x1 - x0 + 1
    ycount = [0] * h
    for y in range(h):
        row = mask[y]
        c = 0
        for x in range(x0, x1 + 1):
            if row[x]:
                c += 1
        ycount[y] = c
    ythresh = max(2, width // 6)
    ys = [y for y in range(h) if ycount[y] >= ythresh]
    if not ys:
        return None
    top, bot = ys[0], ys[-1]
    best_gap = 0
    gap_a = gap_b = None
    y = top
    while y <= bot:
        if ycount[y] < ythresh:
            a = y
            while y <= bot and ycount[y] < ythresh:
                y += 1
            b = y - 1
            glen = b - a + 1
            if glen > best_gap:
                best_gap, gap_a, gap_b = glen, a, b
        else:
            y += 1
    return {
        "cx": cx / float(w),
        "top": top / float(h),
        "bot": bot / float(h),
        "wood_rows": len(ys),
        "span": bot - top + 1,
        "n": n,
        "gap": best_gap,
        "gap_mid": (((gap_a + gap_b) / 2.0) / h
                    if gap_a is not None else None),
    }


def locate_tower_drag(image, piece_hint=None, debug=None):
    """Locate the Move piece and the incomplete wooden tower.

    Live layout: three (sometimes more) vertical wood stacks across the
    left/centre and a 1–2 block ``Move`` segment on the right — often a
    *separate* DOM node outside the photo, so the caller should pass the
    whole challenge iframe (not just the largest canvas) and optionally
    ``piece_hint`` (normalised centre of the Move / draggable control).

    The answer is a drag of that piece onto the shortest stack, or into
    a missing-block gap.

    ``image`` is PNG/JPEG bytes, a PIL Image, or a list-of-rows of
    ``(r, g, b)``. Returns ``{"from": (x, y), "to": (x, y)}`` in
    normalised 0..1 coordinates, or ``None`` when nothing tower-like is
    visible. If ``debug`` is a dict it is filled with counters.

    Does NOT use the punched-slot DragLocator — that geometry is the
    wrong puzzle.
    """
    dbg = debug if isinstance(debug, dict) else {}
    parsed = _rgb_grid(image)
    if parsed is None:
        dbg["reason"] = "unreadable"
        return None
    w, h, rows = parsed
    dbg["size"] = (w, h)

    # Border median ≈ studio backdrop (cream / grey chrome).
    border = []
    step = max(1, w // 40)
    for x in range(0, w, step):
        border.append(rows[0][x][:3])
        border.append(rows[h - 1][x][:3])
    for y in range(0, h, max(1, h // 40)):
        border.append(rows[y][0][:3])
        border.append(rows[y][w - 1][:3])
    br = sorted(p[0] for p in border)[len(border) // 2]
    bg = sorted(p[1] for p in border)[len(border) // 2]
    bb = sorted(p[2] for p in border)[len(border) // 2]
    bg_luma = 0.299 * br + 0.587 * bg + 0.114 * bb

    wood, fg, teal = [], [], []
    wood_n = fg_n = teal_n = 0
    tsx = tsy = 0
    # Ignore the prompt bar / Next button chrome.
    y0 = int(h * 0.10)
    y1 = int(h * 0.92)
    for y in range(h):
        wrow, frow, trow = [], [], []
        in_band = y0 <= y < y1
        for x in range(w):
            pix = rows[y][x]
            r, g, b = pix[0], pix[1], pix[2]
            is_teal = in_band and _is_teal_rgb(r, g, b)
            is_wood = in_band and _is_wood_rgb(r, g, b)
            is_fg = False
            if in_band and not is_teal:
                dist = abs(r - br) + abs(g - bg) + abs(b - bb)
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                if not (r < 40 and g < 40 and b < 40) \
                        and not (b > r + 40 and b > g + 20) \
                        and (is_wood or dist >= 40 or luma < bg_luma - 22):
                    is_fg = True
            wrow.append(is_wood)
            frow.append(is_fg or is_wood)
            trow.append(is_teal)
            if is_wood:
                wood_n += 1
            if is_fg or is_wood:
                fg_n += 1
            if is_teal:
                teal_n += 1
                tsx += x
                tsy += y
        wood.append(wrow)
        fg.append(frow)
        teal.append(trow)
    dbg["wood"] = wood_n
    dbg["fg"] = fg_n
    dbg["teal"] = teal_n

    mask = wood if wood_n >= max(24, (w * h) // 120) else fg
    total = wood_n if mask is wood else fg_n
    dbg["mask"] = "wood" if mask is wood else "fg"
    if total < max(20, (w * h) // 140):
        dbg["reason"] = "too-few-pixels"
        return None

    minc = max(2, h // 30)
    clusters = _x_clusters(mask, w, h, minc, max(12, total // 50))
    dbg["clusters"] = len(clusters)
    if not clusters:
        dbg["reason"] = "no-columns"
        return None

    stacks = []
    for x0, x1, cx, n in clusters:
        st = _measure_stack(mask, x0, x1, cx, n, w, h)
        if st:
            stacks.append(st)
    if not stacks:
        dbg["reason"] = "no-stacks"
        return None

    hint = None
    if piece_hint is not None:
        try:
            hint = (float(piece_hint[0]), float(piece_hint[1]))
        except Exception:
            hint = None
    teal_pt = None
    if teal_n >= 8:
        teal_pt = (tsx / float(teal_n) / w, tsy / float(teal_n) / h)

    stacks.sort(key=lambda s: s["cx"])
    piece = None
    towers = list(stacks)

    # 4+ columns: the rightmost small/short one is the Move piece.
    if len(stacks) >= 4:
        right = stacks[-1]
        others = stacks[:-1]
        tall = max(s["span"] for s in others) or 1
        if right["cx"] >= 0.62 or right["span"] < tall * 0.78:
            piece = (right["cx"], (right["top"] + right["bot"]) / 2.0)
            towers = others
    elif len(stacks) == 3:
        right = stacks[-1]
        others = stacks[:-1]
        tall = max(s["span"] for s in others) or 1
        if right["cx"] >= 0.70 or right["span"] < tall * 0.70:
            piece = (right["cx"], (right["top"] + right["bot"]) / 2.0)
            towers = others

    if piece is None and teal_pt is not None:
        # Badge sits on/above the piece — grab just below it.
        piece = (teal_pt[0], min(0.90, teal_pt[1] + 0.06))
        # Drop the stack that *is* the badge/piece from the tower list.
        towers = [s for s in towers if abs(s["cx"] - teal_pt[0]) > 0.08]

    if piece is None and hint is not None:
        piece = hint
        towers = [s for s in towers if abs(s["cx"] - hint[0]) > 0.08]

    if piece is None:
        # Right-strip leftover (piece painted into the photo).
        split = max(int(w * 0.72), w // 2)
        pc = psx = psy = 0
        for y in range(y0, y1):
            row = mask[y]
            for x in range(split, w):
                if row[x]:
                    pc += 1
                    psx += x
                    psy += y
        if pc >= max(8, ((w - split) * (y1 - y0)) // 240):
            piece = (psx / float(pc) / w, psy / float(pc) / h)
            towers = [s for s in towers if s["cx"] < 0.70]

    if not towers:
        towers = [s for s in stacks if piece is None
                  or abs(s["cx"] - piece[0]) > 0.08]
    if not towers:
        dbg["reason"] = "no-towers"
        return None

    # Prefer a gapped stack; else the one whose peak sits lowest (same
    # baseline → the incomplete tower is the shortest).
    min_block = max(4, int(0.045 * h))
    gapped = [t for t in towers
              if t["gap"] >= min_block and t["gap_mid"] is not None
              and t["top"] < t["gap_mid"] < t["bot"]]
    if gapped:
        target = max(gapped, key=lambda t: t["gap"])
        to = (target["cx"], target["gap_mid"])
        dbg["target"] = "gap"
    else:
        # shortest by peak-y (larger top = shorter stack), then wood_rows
        target = max(towers, key=lambda t: (t["top"], -t["wood_rows"]))
        to = (target["cx"], min(0.92, target["top"] + 0.045))
        dbg["target"] = "shortest"

    if piece is None:
        dbg["reason"] = "no-piece"
        dbg["best"] = {"to": (float(to[0]), float(to[1]))}
        return None
    if abs(piece[0] - to[0]) < 0.04 and abs(piece[1] - to[1]) < 0.04:
        dbg["reason"] = "from-equals-to"
        return None
    dbg["reason"] = "ok"
    dbg["from"] = (round(piece[0], 3), round(piece[1], 3))
    dbg["to"] = (round(to[0], 3), round(to[1], 3))
    return {"from": (float(piece[0]), float(piece[1])),
            "to": (float(to[0]), float(to[1]))}
