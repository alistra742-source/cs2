#!/usr/bin/env python3
"""
tile_classifier.py — runtime wrappers for the three offline solver models
trained by train_models.py (weights in models/<task>.pt + <task>.json).

Everything degrades gracefully: when torch is not installed or the weights
are missing, ``available`` is False and the caller (server.py) falls back to
the vision model. Nothing here touches the network.

    TileClassifier.classify_many(images, with_conf=True)
    TileClassifier.probabilities(images)
    PointLocator.locate(image, target)                    # named target
    PointLocator.scan(image)                              # every class at once
    PointLocator.locate_relational(image, prompt, verifier=TileClassifier())
    DragLocator.locate(image)                             # {"from", "to"}

Coordinates are always normalised 0..1, origin top-left.
"""

from __future__ import annotations

import io
import json
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("SOLVER_MODELS_DIR", os.path.join(_ROOT, "models"))

try:  # torch absence is a supported configuration (vision-model fallback)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from train_models import TileNet, PointNet, DragNet, soft_argmax
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

import hcaptcha_types as hct


def _to_pil(image):
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return image.convert("RGB")


def _prep(image, size):
    im = _to_pil(image)
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return (x - 0.5) / 0.5


class _Base:
    task = ""
    ctor = None

    def __init__(self, models_dir=MODELS_DIR):
        self.available = False
        self.model = None
        self.classes = []
        self.size = 96
        self.width = 16
        if not _TORCH:
            return
        pt = os.path.join(models_dir, "%s.pt" % self.task)
        js = os.path.join(models_dir, "%s.json" % self.task)
        if not (os.path.exists(pt) and os.path.exists(js)):
            return
        try:
            with open(js, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.classes = meta.get("classes", [])
            self.size = int(meta.get("size", 96))
            self.width = int(meta.get("width", 16))
            n = len(self.classes) or 60
            state = torch.load(pt, map_location="cpu")
            # checkpoints predate the background channel: fall back to the
            # plain C-channel head when the saved tensor shape says so
            wants_bg = self.task == "point"
            head_w = "head.weight"
            if head_w in state and state[head_w].shape[0] == n:
                wants_bg = False
            ctor = self._ctor_with_bg(wants_bg) \
                if hasattr(self, "_ctor_with_bg") else self.ctor
            self.model = ctor(n, self.width)
            self.model.load_state_dict(state)
            self.model.eval()
            self.available = True
        except Exception:
            self.model = None
            self.available = False


class TileClassifier(_Base):
    """60-way tile classifier (models/tile.pt)."""

    task = "tile"

    def __init__(self, models_dir=MODELS_DIR):
        self.ctor = lambda n, w: TileNet(n, w)
        super().__init__(models_dir)

    @torch.no_grad() if _TORCH else (lambda f: f)
    def probabilities(self, images):
        """List of {label: prob} dicts, one per image."""
        if not self.available:
            return []
        ims = images if isinstance(images, (list, tuple)) else [images]
        xs = torch.cat([_prep(im, self.size) for im in ims], dim=0)
        probs = F.softmax(self.model(xs), dim=1)  # (B, C)
        out = []
        for row in probs.tolist():
            out.append({self.classes[i]: p for i, p in enumerate(row)})
        return out

    def classify_many(self, images, with_conf=True):
        """[(label, conf)] per image — or [label] when with_conf=False."""
        out = []
        for probs in self.probabilities(images):
            if not probs:
                break
            label = max(probs, key=probs.get)
            out.append((label, probs[label]) if with_conf else label)
        return out


class PointLocator(_Base):
    """One-heatmap-channel-per-class localiser (models/point.pt)."""

    task = "point"

    def __init__(self, models_dir=MODELS_DIR):
        self.ctor = lambda n, w: PointNet(n, w)
        self._ctor_with_bg = lambda wants_bg: (
            lambda n, w: PointNet(n, w, background=wants_bg))
        super().__init__(models_dir)

    def _scores(self, image):
        """One forward pass -> (presence_map, location_map).

        presence: (C, H, W) per-cell softmax ACROSS classes — the classes
        compete for each cell (a raw per-channel peak can't discriminate).
        location: per-class soft-argmax (x, y) of the channel's own spatial
        distribution — sub-cell precision, matching how the net is decoded
        during training (val hit@10% ~0.93 with it, ~0.88 via cell centres).
        """
        if not self.available:
            return None, None
        with torch.no_grad():
            hm = self.model.heatmaps(_prep(image, self.size))  # (1,C[+1],H,W)
            hm = hm.squeeze(0)
            # presence: per-cell softmax across classes AND the background
            # channel — a cell holding nothing salient is claimed by
            # background, which suppresses phantom presences
            presence_map = F.softmax(hm, dim=0)
            if hm.shape[0] > len(self.classes):      # background channel last
                presence_map = presence_map[:-1]
                hm = hm[:-1]
            loc = soft_argmax(hm)  # (C, 2) normalised
            return presence_map, loc

    def scan(self, image):
        """Presence + precise location for every class, sorted by presence.
        Returns [{"label", "presence", "x", "y"}...]."""
        p, loc = self._scores(image)
        if p is None:
            return []
        C, H, W = p.shape
        presence = p.reshape(C, -1).max(dim=1).values
        out = []
        for c in range(C):
            out.append({
                "label": self.classes[c] if c < len(self.classes) else str(c),
                "presence": float(presence[c]),
                "x": float(loc[c][0]),
                "y": float(loc[c][1]),
            })
        out.sort(key=lambda r: -r["presence"])
        return out

    def locate(self, image, target):
        """Named target ("frog") -> (x, y, presence) or None."""
        name = hct.canonical(target) or target
        for row in self.scan(image):
            if row["label"] == name:
                return (row["x"], row["y"], row["presence"])
        return None

    def count(self, image, target, min_peak=0.08, min_sep=0.16,
              weak_gate=0.20, max_n=9, margin=0.04):
        """Counting ("How many X are in this image?") -> int or None.

        The point model is trained on multi-instance count rounds (k
        instances of one class per scene with per-cell background
        competition), so each instance lights its own presence peak. This
        takes the target class's presence map, keeps local maxima above
        ``min_peak``, NMS-clusters them and returns the cluster count.

        It self-gates — and the gates matter, because a count answer is
        graded EXACTLY: any peak touching the image border, an
        over-fragmented map, or a weakest kept peak below ``weak_gate``
        returns None so the caller falls back to the vision model instead
        of answering a wrong number. (Measured on held-out count rounds:
        ~72% answered exactly offline, ~22% gated to vision.)
        """
        if not self.available:
            return None
        name = hct.canonical(target) or target
        if name not in self.classes:
            return None
        cid = self.classes.index(name)
        presence, loc = self._scores(image)
        if presence is None:
            return None
        chan = presence[cid]                     # (H, W) 0..1
        H, W = chan.shape
        peaks = []
        for y in range(H):
            for x in range(W):
                v = float(chan[y, x])
                if v < min_peak:
                    continue
                # local maximum in the 3x3 neighbourhood (padding-safe)
                y0, y1 = max(0, y - 1), min(H, y + 2)
                x0, x1 = max(0, x - 1), min(W, x + 2)
                if v < float(chan[y0:y1, x0:x1].max()):
                    continue
                peaks.append((v, x, y))
        peaks.sort(reverse=True)
        kept = []
        for v, x, y in peaks:
            if all(max(abs(x - kx), abs(y - ky)) >= min_sep * W
                   for _, kx, ky in kept):
                kept.append((v, x, y))
        if not kept:
            return None
        # self-gates: border-touching peaks usually mean a truncated
        # object; a fragmented map (>= max_n clusters) or a weakest kept
        # peak below weak_gate means the scene is too uncertain to answer
        for _, x, y in kept:
            if (x / W) < margin or (x / W) > 1 - margin \
                    or (y / H) < margin or (y / H) > 1 - margin:
                return None
        if len(kept) >= max_n:
            return None
        if kept[-1][0] < weak_gate:
            return None
        return len(kept)

    def locate_relational(self, image, prompt, verifier=None):
        """Superlative prompts ("click the animal who jumps the highest").

        Scores every class in one pass, keeps presence >= 0.30, crops each
        candidate peak and confirms it with the tile classifier (heatmap
        argmax alone sometimes promotes the wrong class), then ranks the
        survivors with the shared superlative table. Returns
        (x, y, label) or None.
        """
        sup = hct.superlative_table(prompt)
        if sup is None:
            target = hct.extract_target(prompt)
            hit = self.locate(image, target) if target else None
            return (hit[0], hit[1], target) if hit else None
        table, direction = sup
        rows = self.scan(image)
        pool = [r for r in rows
                if r["presence"] >= 0.30 and r["label"] in table]
        # NMS: one REAL object hosts several class peaks at nearly the same
        # pixel — keep only the strongest-labelled peak per location cluster,
        # the rest are phantom activations.
        def dist(a, b):
            return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5

        candidates = []
        for row in pool:
            if all(dist(row, kept) > 0.12 for kept in candidates):
                candidates.append(row)
        # tile-classifier pass on each candidate crop, two jobs:
        #   RELABEL — the per-cell winner of the heatmap is the least stable
        #     part of the scene model (a bus patch often peaks as "train");
        #     when the photo-native tile classifier confidently names the
        #     crop something table-ranked, trust its identity
        #   VETO    — only a CONFIDENT contradiction drops the candidate
        #     (scene crops are out-of-tile-distribution; a weak mismatch
        #     means "unsure", not "wrong")
        if verifier is not None and getattr(verifier, "available", False):
            im = _to_pil(image)
            W, H = im.size
            kept = []
            for row in candidates:
                rx = 0.42 * min(W, H) / 2
                cx, cy = row["x"] * W, row["y"] * H
                crop = im.crop((max(0, cx - rx), max(0, cy - rx),
                                min(W, cx + rx), min(H, cy + rx)))
                got = verifier.classify_many([crop])
                if got:
                    lab, conf = got[0]
                    if lab != row["label"] and lab in table \
                            and conf >= 0.55:
                        row = dict(row, label=lab)      # relabel
                    elif lab != row["label"] and conf >= 0.60:
                        continue                        # veto
                kept.append(row)
            candidates = kept
        # relabeling can merge two clusters onto one label — keep the closer
        # match of the two (same physics as the NMS above)
        uniq = {}
        for row in candidates:
            have = uniq.get(row["label"])
            if have is None or row["presence"] > have["presence"]:
                uniq[row["label"]] = row
        candidates = list(uniq.values())
        if not candidates:
            return None
        key = lambda r: table[r["label"]]
        best = max(candidates, key=key) if direction == "max" \
            else min(candidates, key=key)
        return (best["x"], best["y"], best["label"])


class DragLocator(_Base):
    """Piece + slot localiser for drag-drop rounds (models/drag.pt)."""

    task = "drag"

    def __init__(self, models_dir=MODELS_DIR):
        self.ctor = lambda n, w: DragNet(n, w)
        super().__init__(models_dir)

    def locate(self, image):
        """{"from": (x, y), "to": (x, y)} normalised, or None."""
        if not self.available:
            return None
        with torch.no_grad():
            hm = self.model.heatmaps(_prep(image, self.size))  # (1,2,H,W)
            p_from = soft_argmax(hm[:, 0])[0]
            p_to = soft_argmax(hm[:, 1])[0]
        return {"from": (float(p_from[0]), float(p_from[1])),
                "to": (float(p_to[0]), float(p_to[1]))}
