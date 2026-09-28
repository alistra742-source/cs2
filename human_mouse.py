#!/usr/bin/env python3
"""
human_mouse.py — pointer realism for clicking inside the hCaptcha frame.

hCaptcha grades pointer telemetry: a bare ``element.click()`` teleports the
cursor, dwells 0 ms and produces a perfectly straight trail on repeat runs
— an immediate automation tell. Every interaction here instead:

  * glides along a slightly bowed cubic Bezier with ease-in-out timing and
    sub-pixel tremor (``path``),
  * overshoots and corrects like a real hand on long moves (``move``),
  * settles before pressing, holds a human 45-130 ms dwell between
    ``mouse.down()``/``mouse.up()`` (``click``),
  * lands at a gaussian point INSIDE the target box, never dead centre
    (``click_box``),
  * performs real press/move/release drags with micro-adjustments
    (``drag``) — hCaptcha drag challenges ignore synthetic clicks entirely.

All functions are async and take a Playwright ``page``. The pure geometry
(``path``) is deterministic given an `rng` and is unit-tested offline with
a fake mouse.
"""

from __future__ import annotations

import asyncio
import math
import random


def _get_pos(page):
    try:
        return getattr(page, "_hm_pos", None)
    except Exception:
        return None


def _set_pos(page, pos):
    try:
        page._hm_pos = pos
    except Exception:
        pass


def _ease(t: float) -> float:
    """Cubic ease-in-out: accelerate, cruise, decelerate."""
    return 3.0 * t * t - 2.0 * t * t * t


def path(start, end, rng: random.Random = None, steps: int = None):
    """Cubic-Bezier pointer path with bow, ease-in-out and tremor.

    Returns a list of (x, y) float waypoints from ``start`` to ``end``.
    12-60 samples depending on distance; never straight (control points are
    pushed perpendicular to the line of travel by a bow proportional to the
    distance), never teleports (waypoint spacing stays bounded).
    """
    rng = rng or random
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1.0:
        return [(x1, y1)]

    # sample count scales with distance, clamped to a human range
    n = steps or max(12, min(60, int(12 + dist / 14.0)))

    # perpendicular bow (~4-14% of distance, random side, slight asymmetry)
    ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
    px, py = -uy, ux                       # unit perpendicular
    bow = dist * rng.uniform(0.04, 0.14) * rng.choice((-1.0, 1.0))
    a = rng.uniform(0.24, 0.40)            # control point 1 position on the line
    b = rng.uniform(0.60, 0.78)            # control point 2
    c1 = (x0 + (x1 - x0) * a + px * bow,
          y0 + (y1 - y0) * a + py * bow)
    c2 = (x0 + (x1 - x0) * b + px * bow * rng.uniform(0.5, 1.1),
          y0 + (y1 - y0) * b + py * bow * rng.uniform(0.5, 1.1))

    pts = []
    for i in range(n + 1):
        t = _ease(i / n)
        mt = 1.0 - t
        x = mt ** 3 * x0 + 3 * mt * mt * t * c1[0] + \
            3 * mt * t * t * c2[0] + t ** 3 * x1
        y = mt ** 3 * y0 + 3 * mt * mt * t * c1[1] + \
            3 * mt * t * t * c2[1] + t ** 3 * y1
        if 0 < i < n:                      # sub-pixel tremor (not on endpoints)
            x += rng.uniform(-0.45, 0.45)
            y += rng.uniform(-0.45, 0.45)
        pts.append((x, y))
    pts[-1] = (x1, y1)                     # land exactly on target
    return pts


async def _glide(page, start, x, y, rng: random.Random, steps=None):
    for wx, wy in path(start, (x, y), rng, steps=steps):
        await page.mouse.move(wx, wy)
        if rng.random() < 0.10:
            await asyncio.sleep(rng.uniform(0.004, 0.018))
    _set_pos(page, (x, y))


async def move(page, x: float, y: float, rng: random.Random = None):
    """Glide the pointer to (x, y); overshoot-and-correct past ~140 px."""
    rng = rng or random
    start = _get_pos(page)
    if start is None:
        # First move of the session: humans come from somewhere off-target.
        start = (x + rng.uniform(-200, 200), y + rng.uniform(-160, 160))
    dist = math.hypot(x - start[0], y - start[1])
    if dist > 140:
        # overshoot 2-7% past the target, then a short correction glide
        over = rng.uniform(1.02, 1.07)
        ox = start[0] + (x - start[0]) * over
        oy = start[1] + (y - start[1]) * over
        await _glide(page, start, ox, oy, rng)
        await asyncio.sleep(rng.uniform(0.02, 0.06))
        await _glide(page, (ox, oy), x, y, rng,
                     steps=max(6, 12 + int(dist / 30)))
    else:
        await _glide(page, start, x, y, rng)


async def click(page, x: float, y: float, rng: random.Random = None,
                dwell: float = None):
    """move -> settle -> down -> human dwell -> up."""
    rng = rng or random
    await move(page, x, y, rng)
    await asyncio.sleep(rng.uniform(0.04, 0.12))          # settle on target
    await page.mouse.down()
    await asyncio.sleep(dwell if dwell is not None
                       else rng.uniform(0.045, 0.130))     # press dwell
    await page.mouse.up()


async def click_box(page, box, rng: random.Random = None):
    """Click at a gaussian point INSIDE a {"x","y","width","height"} box.

    Real users almost never hit the geometric centre; the landing point is
    a clamped normal around the middle of the box.
    """
    rng = rng or random
    if isinstance(box, dict):
        left = float(box.get("x", 0.0))
        top = float(box.get("y", 0.0))
        w = float(box.get("width", 0.0))
        h = float(box.get("height", 0.0))
    else:
        left, top, w, h = (float(v) for v in box)
    nx = min(0.85, max(0.15, rng.gauss(0.5, 0.16)))
    ny = min(0.85, max(0.15, rng.gauss(0.5, 0.16)))
    await click(page, left + nx * w, top + ny * h, rng)


async def drag(page, start, end, rng: random.Random = None):
    """A REAL drag: press, travel with many samples, adjust, release.

    hCaptcha's drag-drop listens to the full pointer gesture — a synthetic
    click at the destination does nothing. The travel uses extra waypoints,
    then 1-3 small "homing" adjustments before release, like a person
    fitting a puzzle piece into its slot.
    """
    rng = rng or random
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    await move(page, sx, sy, rng)
    await asyncio.sleep(rng.uniform(0.10, 0.28))          # read/aim pause
    await page.mouse.down()
    await asyncio.sleep(rng.uniform(0.08, 0.16))          # grip the piece
    # travel: dense samples so motion sensors see a continuous trail
    dist = math.hypot(ex - sx, ey - sy)
    steps = max(16, min(60, int(dist / 10.0)))
    for wx, wy in path((sx, sy), (ex, ey), rng, steps=steps):
        await page.mouse.move(wx, wy)
        if rng.random() < 0.15:
            await asyncio.sleep(rng.uniform(0.004, 0.016))
    _set_pos(page, (ex, ey))
    # micro-adjustments around the drop point ("does it fit?")
    for _ in range(rng.randint(1, 3)):
        jx = ex + rng.uniform(-4.0, 4.0)
        jy = ey + rng.uniform(-4.0, 4.0)
        await page.mouse.move(jx, jy)
        await asyncio.sleep(rng.uniform(0.02, 0.07))
    await page.mouse.move(ex, ey)
    await asyncio.sleep(rng.uniform(0.06, 0.16))          # hover before drop
    await page.mouse.up()
    _set_pos(page, (ex, ey))
