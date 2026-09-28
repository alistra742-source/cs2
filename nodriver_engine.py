"""nodriver_engine.py — Playwright-compatible async facade backed by nodriver
on a REAL Google Chrome (google-chrome-stable).

The rest of the application imports ``async_playwright`` and ``ENGINE`` from
``browser_engine`` and drives everything through Playwright's async API. This
module re-implements the *subset* of that API the code actually uses on top
of nodriver's CDP primitives, so every caller (server.py workers,
captcha_solver.py, drag_solver.py, live_control.py) is unchanged.

Why nodriver + google-chrome-stable:
  · REAL Chrome — the fingerprint IS a real user's: genuine Chrome JS/TLS/HTTP2
    fingerprints, real fonts, real plugin/NTP surface. No engine-injected
    spoofing to get flagged; nodriver only strips the automation tells
    (navigator.webdriver, CDP hooks, headless UA artifacts).
  · nodriver drives Chrome directly over CDP — no Selenium/WebDriver layer.

Mapping notes (same contract as the previous engines):

  · ``page.evaluate(js)`` accepts both bare JS expressions (``location.href``)
    and arrow/function bodies (``() => {...}``); function bodies are wrapped
    and invoked so they actually run (matching Playwright's semantics).
  · ``page.locator(sel)`` -> ``_Locator`` supports the chained surface used in
    the code: ``.count()``, ``.first``, ``.nth(i)``, ``.filter(visible=True)``,
    ``.click()``, ``.fill()``, ``.press()``, ``.inner_text()``,
    ``.input_value()``, ``.is_visible()``, ``.is_disabled()``,
    ``.bounding_box()``, ``.screenshot()``, ``.content_frame()``,
    ``.element_handle()``.
  · Frames are first-class objects (``page.frames``, ``frame.locator``,
    ``frame.frame_element()``, ``frame.screenshot()``) — an iframe is a
    nodriver IFrame, which is itself a CDP connection, so frame-scoped
    selectors/evaluation run in the frame's own session (no frame switching).
  · ``page.on("request"/"response"/"crash")`` maps to CDP Network / Inspector
    events, so the rqdata + payload capture and crash flag keep working.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import time
from typing import Any, Callable, List, Optional

ENGINE = "nodriver"
CHANNEL = None

try:  # pragma: no cover - raised as a clear error at launch
    import nodriver as uc
    from nodriver import cdp
    from nodriver.core import util as _nd_util
except Exception:  # pragma: no cover
    uc = None
    cdp = None
    _nd_util = None

__all__ = ["async_playwright", "ENGINE", "CHANNEL"]


# ─────────────────────────────────────────────────────────────────────────────
# JS expression helpers
# ─────────────────────────────────────────────────────────────────────────────
_ARROW_RE = re.compile(r"^\s*(async\s+)?(\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")


def _wrap_js(js: Any, arg: Any = None) -> tuple:
    """Normalise a JS string for Runtime.evaluate.

    Playwright evaluates bare expressions directly but runs arrow/function
    bodies, and passes `arg` into the invoked function when supplied. The code
    mixes all of these, so detect function-like strings and invoke them —
    forwarding the argument (e.g. ``([sel, value]) => {...}`` called with a
    two-element list). Returns (expression, await_promise).
    """
    s = (js if isinstance(js, str) else "").strip()
    if not s:
        return "undefined", False
    if _ARROW_RE.match(s) or s.startswith("function") or s.startswith("async function"):
        if arg is None:
            expr = f"({s})()"
        else:
            expr = f"({s})({json.dumps(arg)})"
        return expr, s.startswith("async")
    return s, False


def _wrap_new_document(src: str) -> str:
    """Wrap an init-script string so it runs at document start.

    ``Page.addScriptToEvaluateOnNewDocument`` wants a *statement*; arrow
    function bodies like ``() => {...}`` would otherwise just define (and
    drop) the function. Comments pass through unchanged.
    """
    s = (src or "").strip()
    if not s or s.startswith("//") or s.startswith("/*"):
        return s
    if _ARROW_RE.match(s) or s.startswith("function") or s.startswith("async function"):
        return f"({s})();"
    return s


def _quad_to_box(quad) -> Optional[dict]:
    """CDP content quad ([x1,y1,...x4,y4] floats) -> CSS box dict."""
    try:
        xs = quad[0::2]
        ys = quad[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return {"x": float(x0), "y": float(y0),
                "width": float(x1 - x0), "height": float(y1 - y0)}
    except Exception:
        return None


async def _element_box(el) -> Optional[dict]:
    """Bounding box (page CSS pixels) of a nodriver Element, or None."""
    try:
        pos = await el.get_position()
        if pos is None:
            return None
        return {"x": float(pos.left), "y": float(pos.top),
                "width": float(pos.width), "height": float(pos.height)}
    except Exception:
        return None


async def _element_shot(tab, box: dict) -> bytes:
    """Clipped beyond-viewport capture of an element box (PNG bytes)."""
    try:
        clip = cdp.page.Viewport(x=box["x"], y=box["y"],
                                 width=box["width"], height=box["height"],
                                 scale=1.0)
        data = await tab.send(cdp.page.capture_screenshot(
            format_="png", capture_beyond_viewport=True, clip=clip))
        if data:
            return base64.b64decode(data)
    except Exception:
        pass
    return b""


# ─────────────────────────────────────────────────────────────────────────────
# Mouse
# ─────────────────────────────────────────────────────────────────────────────
_BUTTON_MASK = {"left": 1, "right": 2, "middle": 4}


def mouse_move_points(sx: float, sy: float, x: float, y: float,
                      steps: Optional[int] = None) -> List[tuple]:
    """Waypoints from the current cursor to (x, y).

    Playwright interpolates from the *current* position. The previous
    implementation scaled toward the origin, so a drag with ``steps>1``
    jumped to (0,0) mid-gesture and hCaptcha never saw a real press-move.
    """
    n = int(steps) if steps and int(steps) > 1 else 1
    x, y = float(x), float(y)
    sx, sy = float(sx), float(sy)
    if n <= 1:
        return [(x, y)]
    return [(sx + (x - sx) * i / n, sy + (y - sy) * i / n)
            for i in range(1, n + 1)]


class _Mouse:
    """Playwright page.mouse over CDP Input.dispatchMouseEvent.

    Tracks cursor position *and* pressed buttons so ``move()`` after
    ``down()`` emits ``buttons=1`` mouseMoved events — required for
    hCaptcha drag challenges. ``down()``/``up()`` land where the pointer
    already is, like Playwright.
    """

    def __init__(self, page: "_Page"):
        self._page = page
        self._tab = page._tab
        self._x = 0.0
        self._y = 0.0
        self._buttons = 0

    def _button_name(self, button: str) -> str:
        return button if button in ("left", "right", "middle") else "left"

    def _mask_for(self, button: str) -> int:
        return _BUTTON_MASK.get(self._button_name(button), 1)

    async def _dispatch_move(self, x: float, y: float) -> None:
        held = "left" if (self._buttons & 1) else (
            "right" if (self._buttons & 2) else (
                "middle" if (self._buttons & 4) else "none"))
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mouseMoved", x=float(x), y=float(y),
            button=cdp.input_.MouseButton(held), buttons=int(self._buttons)))

    async def move(self, x: float, y: float, steps: Optional[int] = None) -> None:
        for mx, my in mouse_move_points(self._x, self._y, x, y, steps):
            await self._dispatch_move(mx, my)
            self._x, self._y = float(mx), float(my)
        self._x, self._y = float(x), float(y)

    async def click(self, x: float, y: float, button: str = "left", **kwargs) -> None:
        await self.move(x, y, steps=1)
        await self.down(button)
        await asyncio.sleep(random.uniform(0.03, 0.09))
        await self.up(button)

    async def dblclick(self, x: float, y: float, button: str = "left", **kwargs) -> None:
        await self.move(x, y, steps=1)
        name = self._button_name(button)
        b = cdp.input_.MouseButton(name)
        mask = self._mask_for(name)
        for _ in range(2):
            self._buttons |= mask
            await self._tab.send(cdp.input_.dispatch_mouse_event(
                "mousePressed", x=self._x, y=self._y, button=b,
                buttons=self._buttons, click_count=2))
            await asyncio.sleep(random.uniform(0.02, 0.06))
            self._buttons &= ~mask
            await self._tab.send(cdp.input_.dispatch_mouse_event(
                "mouseReleased", x=self._x, y=self._y, button=b,
                buttons=self._buttons, click_count=2))

    async def down(self, button: str = "left", **kwargs) -> None:
        name = self._button_name(button)
        mask = self._mask_for(name)
        self._buttons |= mask
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mousePressed", x=self._x, y=self._y,
            button=cdp.input_.MouseButton(name),
            buttons=self._buttons, click_count=1))

    async def up(self, button: str = "left", **kwargs) -> None:
        name = self._button_name(button)
        mask = self._mask_for(name)
        self._buttons &= ~mask
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mouseReleased", x=self._x, y=self._y,
            button=cdp.input_.MouseButton(name),
            buttons=self._buttons, click_count=1))

    async def wheel(self, delta_x: float = 0, delta_y: float = 0) -> None:
        await self._tab.send(cdp.input_.dispatch_mouse_event(
            "mouseWheel", x=self._x, y=self._y,
            delta_x=float(delta_x), delta_y=float(delta_y)))


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard
# ─────────────────────────────────────────────────────────────────────────────
_MODIFIER_BITS = {
    "Alt": 1, "AltGr": 1, "Control": 2, "ControlLeft": 2, "ControlRight": 2,
    "Meta": 4, "MetaLeft": 4, "MetaRight": 4, "Shift": 8, "ShiftLeft": 8,
    "ShiftRight": 8,
}
_SPECIAL_KEYS = {
    "Enter": (13, "Enter"), "Tab": (9, "Tab"), "Backspace": (8, "Backspace"),
    "Escape": (27, "Escape"), "Delete": (46, "Delete"), "Home": (36, "Home"),
    "End": (35, "End"), "PageUp": (33, "PageUp"), "PageDown": (34, "PageDown"),
    "ArrowUp": (38, "ArrowUp"), "ArrowDown": (40, "ArrowDown"),
    "ArrowLeft": (37, "ArrowLeft"), "ArrowRight": (39, "ArrowRight"),
    "Space": (32, " "), " ": (32, " "), "Control": (17, "Control"),
    "Shift": (16, "Shift"), "Alt": (18, "Alt"), "Meta": (91, "Meta"),
}
_SHIFTED_CHARS = set("~!@#$%^&*()_+{}|:\"<>?")


def _code_for(name: str) -> str:
    if len(name) == 1 and name.isalpha():
        return "Key" + name.upper()
    if len(name) == 1 and name.isdigit():
        return "Digit" + name
    return name


def _key_parts(key: str):
    mods = 0
    main = key
    if "+" in key:
        parts = key.split("+")
        main = parts[-1].strip()
        for m in parts[:-1]:
            mods |= _MODIFIER_BITS.get(m.strip(), 0)
    return main, mods


async def _dispatch_press(tab, key: str) -> None:
    """Dispatch a single key press (down, optional char, up) with modifiers.

    Chromium inserts a character when EITHER keyDown carries ``text`` OR a
    separate ``char`` event is sent. Emitting both doubles every letter
    (``glass`` -> ``ggllaassss``). Non-Shift modifiers (Ctrl/Alt/Meta) must
    never produce a char at all — otherwise ``Control+a`` types an ``a``
    instead of selecting, the field never clears, and every refill appends
    on top of the previous value (the mangled
    ``shshsh@gggggglaaaaaassssswhiteeee`` form-fill bug).
    """
    main, mods = _key_parts(key)
    if main in _SPECIAL_KEYS:
        vk, name = _SPECIAL_KEYS[main]
        text = None
    elif len(main) == 1:
        ch = main
        if ch.isalpha():
            vk, name = ord(ch.upper()), ch
        elif ch.isdigit():
            vk, name = ord(ch), ch
        else:
            vk, name = ord(ch), ch
        if ch.isupper() or ch in _SHIFTED_CHARS:
            mods |= 8
        # Only emit an insertable char when no non-Shift modifier is held.
        # Control/Alt/Meta shortcuts (Ctrl+A, Ctrl+C, ...) must not type.
        non_shift = mods & ~8
        text = ch if (ch.isprintable() and not non_shift) else None
    else:
        vk, name = ord(main[0]) if main else 0, main
        text = None

    code = _code_for(name)
    # keyDown/keyUp must NOT carry text when a char event follows — Chrome
    # would insert the character twice (once per event).
    down = dict(type_="keyDown", key=name, code=code, windows_virtual_key_code=vk,
                native_virtual_key_code=vk, modifiers=mods)
    await tab.send(cdp.input_.dispatch_key_event(**down))
    if text:
        await tab.send(cdp.input_.dispatch_key_event(
            type_="char", key=name, code=code, windows_virtual_key_code=vk,
            modifiers=mods, text=text, unmodified_text=text))
    up = dict(type_="keyUp", key=name, code=code, windows_virtual_key_code=vk,
              native_virtual_key_code=vk, modifiers=mods)
    await tab.send(cdp.input_.dispatch_key_event(**up))


class _Keyboard:
    """Playwright page.keyboard over CDP Input.dispatchKeyEvent."""

    def __init__(self, page: "_Page"):
        self._page = page
        self._tab = page._tab

    async def press(self, key: str, **kwargs) -> None:
        await _dispatch_press(self._tab, str(key))

    async def type(self, text: str, delay: float = 0, **kwargs) -> None:
        for ch in str(text):
            if ch == "\n":
                await _dispatch_press(self._tab, "Enter")
            elif ch == "\t":
                await _dispatch_press(self._tab, "Tab")
            else:
                await _dispatch_press(self._tab, ch)
            if delay:
                await asyncio.sleep(float(delay))

    async def insert_text(self, text: str, **kwargs) -> None:
        await self._tab.send(cdp.input_.insert_text(str(text)))

    async def down(self, key: str, **kwargs) -> None:
        main, mods = _key_parts(str(key))
        if main in _SPECIAL_KEYS:
            vk, name = _SPECIAL_KEYS[main]
        else:
            vk, name = ord(main[0]) if main else 0, main
        await self._tab.send(cdp.input_.dispatch_key_event(
            type_="keyDown", key=name, code=_code_for(name),
            windows_virtual_key_code=vk, native_virtual_key_code=vk, modifiers=mods))

    async def up(self, key: str, **kwargs) -> None:
        main, mods = _key_parts(str(key))
        if main in _SPECIAL_KEYS:
            vk, name = _SPECIAL_KEYS[main]
        else:
            vk, name = ord(main[0]) if main else 0, main
        await self._tab.send(cdp.input_.dispatch_key_event(
            type_="keyUp", key=name, code=_code_for(name),
            windows_virtual_key_code=vk, native_virtual_key_code=vk, modifiers=mods))


# ─────────────────────────────────────────────────────────────────────────────
# Locator
# ─────────────────────────────────────────────────────────────────────────────
class _Locator:
    """Playwright locator over nodriver Element resolution.

    A locator is a selector (optionally scoped to a frame or to frames
    reached through an iframe selector), an optional index, and optional
    visibility filtering. Resolution happens lazily on each operation.
    """

    def __init__(self, page: "_Page", selector: str, *, frame: Any = None,
                 frame_selector: Optional[str] = None, index: Optional[int] = None,
                 visible_only: bool = False):
        self._page = page
        self._tab = page._tab
        self._selector = selector
        self._frame = frame
        self._frame_selector = frame_selector
        self._index = index
        self._visible_only = visible_only

    # ---- chaining ---------------------------------------------------------
    @property
    def first(self) -> "_Locator":
        return _Locator(self._page, self._selector, frame=self._frame,
                        frame_selector=self._frame_selector, index=0,
                        visible_only=self._visible_only)

    def nth(self, index: int) -> "_Locator":
        return _Locator(self._page, self._selector, frame=self._frame,
                        frame_selector=self._frame_selector, index=int(index),
                        visible_only=self._visible_only)

    def filter(self, visible: bool = False, **kwargs) -> "_Locator":
        return _Locator(self._page, self._selector, frame=self._frame,
                        frame_selector=self._frame_selector, index=self._index,
                        visible_only=visible)

    def locator(self, selector: str) -> "_Locator":
        # descendant locator (rarely used; concat by '>>' is not supported by
        # querySelector, so resolve parent then child under it)
        return _Locator(self._page, selector, frame=self._frame,
                        frame_selector=self._frame_selector)

    async def all(self) -> List["_Locator"]:
        els = await self._resolve_elements()
        return [_Locator(self._page, self._selector, frame=self._frame,
                         frame_selector=self._frame_selector, index=i,
                         visible_only=self._visible_only) for i in range(len(els))]

    # ---- resolution -------------------------------------------------------
    async def _collect(self) -> list:
        if self._frame_selector:
            frames = await self._page._frames_for_selector(self._frame_selector)
            out: list = []
            for fr in frames:
                try:
                    out.extend(await fr._conn().query_selector_all(self._selector))
                except Exception:
                    pass
            return out
        if self._frame is not None:
            try:
                return await self._frame._conn().query_selector_all(self._selector)
            except Exception:
                return []
        return await self._tab.query_selector_all(self._selector)

    async def _resolve_elements(self) -> list:
        els = await self._collect()
        if self._visible_only and els:
            out = []
            for e in els:
                try:
                    if (await e.get_position()) is not None:
                        out.append(e)
                except Exception:
                    pass
            els = out
        return els

    def _idx(self) -> int:
        return 0 if self._index is None else self._index

    async def _pick(self):
        els = await self._resolve_elements()
        if not els:
            return None
        i = self._idx()
        if i >= len(els):
            return None
        return els[i]

    async def _wait_element(self, timeout: Optional[float] = None, retries: int = 200):
        timeout = (timeout or 30000) / 1000.0
        deadline = time.monotonic() + timeout
        while True:
            el = await self._pick()
            if el is not None:
                return el
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.2)

    # ---- operations -------------------------------------------------------
    async def count(self) -> int:
        return len(await self._resolve_elements())

    async def is_visible(self) -> bool:
        el = await self._pick()
        if el is None:
            return False
        try:
            return (await el.get_position()) is not None
        except Exception:
            return False

    async def inner_text(self) -> str:
        el = await self._pick()
        if el is None:
            return ""
        try:
            return str(await el.apply("(e) => e.innerText") or "")
        except Exception:
            return ""

    async def input_value(self) -> str:
        el = await self._pick()
        if el is None:
            return ""
        try:
            return str(await el.apply("(e) => e.value") or "")
        except Exception:
            return ""

    async def get_attribute(self, name: str) -> Optional[str]:
        el = await self._pick()
        if el is None:
            return None
        try:
            return el[name]
        except Exception:
            return None

    async def focus(self, timeout: Optional[float] = None, **kwargs) -> None:
        el = await self._pick()
        if el is not None:
            try:
                await el.focus()
            except Exception:
                pass

    async def is_disabled(self) -> bool:
        el = await self._pick()
        if el is None:
            return True
        try:
            return bool(await el.apply(
                "(e) => e.disabled || e.getAttribute('aria-disabled') === 'true'"))
        except Exception:
            return True

    async def scroll_into_view_if_needed(self, timeout: Optional[float] = None,
                                         **kwargs) -> None:
        el = await self._pick()
        if el is not None:
            try:
                await el.scroll_into_view()
            except Exception:
                pass

    async def dispatch_event(self, event: str, timeout: Optional[float] = None,
                             **kwargs) -> None:
        el = await self._pick()
        if el is None:
            raise TimeoutError(f"locator dispatch_event: not found for '{self._selector}'")
        try:
            await el.apply(
                f"(e) => e.dispatchEvent(new Event('{event}', {{bubbles: true, cancelable: true}}))")
        except Exception:
            pass

    async def select_option(self, index: Optional[int] = None, value: Optional[str] = None,
                            label: Optional[str] = None, timeout: Optional[float] = None,
                            **kwargs) -> None:
        el = await self._pick()
        if el is None:
            raise TimeoutError(f"locator select_option: not found for '{self._selector}'")
        js = "(e) => { let idx = -1; "
        if index is not None:
            js += f"idx = {int(index)}; "
        elif value is not None:
            js += ("for (let i=0;i<(e.options||[]).length;i++)"
                   f"{{ if (String(e.options[i].value) === {json.dumps(str(value))}) {{ idx = i; break; }} }} ")
        elif label is not None:
            js += ("for (let i=0;i<(e.options||[]).length;i++)"
                   f"{{ if (String(e.options[i].text) === {json.dumps(str(label))}) {{ idx = i; break; }} }} ")
        js += ("if (idx >= 0) { e.selectedIndex = idx; "
               "e.dispatchEvent(new Event('input', {bubbles: true})); "
               "e.dispatchEvent(new Event('change', {bubbles: true})); } }")
        await el.apply(js)

    async def bounding_box(self):
        el = await self._pick()
        if el is None:
            return None
        return await _element_box(el)

    async def click(self, timeout: Optional[float] = None, force: bool = False, **kwargs) -> None:
        el = await self._wait_element(timeout)
        if el is None:
            raise TimeoutError(f"locator click: element not found for '{self._selector}'")
        pos = await el.get_position()
        if not pos:
            try:
                await el.scroll_into_view()
            except Exception:
                pass
            await asyncio.sleep(0.15)
            pos = await el.get_position()
        if not pos:
            raise TimeoutError(f"locator click: element not visible for '{self._selector}'")
        # Playwright auto-scrolls the element into view BEFORE clicking —
        # a center below the viewport edge would click empty space. That is
        # exactly how the DOB Year option (its menu opens past the 720px
        # viewport bottom) went unclicked while Month/Day filled fine.
        if not force:
            try:
                await el.scroll_into_view()
                await asyncio.sleep(0.12)
                fresh = await el.get_position()
                if fresh:
                    pos = fresh
            except Exception:
                pass
        x, y = pos.center
        tab = el.tab if el.tab is not None else self._tab
        await tab.send(cdp.input_.dispatch_mouse_event(
            "mousePressed", x=float(x), y=float(y),
            button=cdp.input_.MouseButton("left"), buttons=1, click_count=1))
        await asyncio.sleep(random.uniform(0.04, 0.09))
        await tab.send(cdp.input_.dispatch_mouse_event(
            "mouseReleased", x=float(x), y=float(y),
            button=cdp.input_.MouseButton("left"), buttons=0, click_count=1))

    async def fill(self, text: str, timeout: Optional[float] = None, **kwargs) -> None:
        """Replace the whole input value (Playwright fill semantics).

        Never appends. Clears via the native value setter first so a broken
        Ctrl+A path cannot leave old characters in the field, then inserts
        the new text with a single CDP insertText (trusted input events React
        accepts). Falls back to per-char key events only if insertText fails.
        """
        el = await self._wait_element(timeout)
        if el is None:
            raise TimeoutError(f"locator fill: element not found for '{self._selector}'")
        tab = el.tab if el.tab is not None else self._tab
        try:
            await el.focus()
        except Exception:
            pass
        # Hard-clear the current value on the element itself. Focus + Ctrl+A
        # alone is not enough: if select-all fails (or used to type an "a"),
        # subsequent keystrokes APPEND and produce mangled concatenations
        # like "user@domain.comuser@domain.com" / doubled letters.
        try:
            await el.apply(
                "(e) => {"
                "  try {"
                "    const setter = Object.getOwnPropertyDescriptor("
                "      window.HTMLInputElement.prototype, 'value').set"
                "      || Object.getOwnPropertyDescriptor("
                "      window.HTMLTextAreaElement.prototype, 'value').set;"
                "    if (setter) setter.call(e, '');"
                "    else e.value = '';"
                "    try { const t = e._valueTracker; if (t && t.setValue) t.setValue(''); } catch (err) {}"
                "    e.dispatchEvent(new Event('input', { bubbles: true }));"
                "  } catch (err) { try { e.value = ''; } catch (e2) {} }"
                "}"
            )
        except Exception:
            # Last-resort clear via keyboard (now correctly non-typing Ctrl+A).
            try:
                await _dispatch_press(tab, "Control+a")
                await _dispatch_press(tab, "Backspace")
            except Exception:
                pass
        text = str(text)
        if not text:
            return
        # Single trusted insert — no per-char doubling risk.
        try:
            await tab.send(cdp.input_.insert_text(text))
            return
        except Exception:
            pass
        for ch in text:
            if ch == "\n":
                await _dispatch_press(tab, "Enter")
            else:
                await _dispatch_press(tab, ch)

    async def press(self, key: str, timeout: Optional[float] = None, **kwargs) -> None:
        el = await self._pick()
        if el is not None:
            try:
                await el.focus()
            except Exception:
                pass
        tab = el.tab if (el is not None and el.tab is not None) else self._tab
        await _dispatch_press(tab, str(key))

    async def screenshot(self, timeout: Optional[float] = None, **kwargs) -> bytes:
        el = await self._wait_element(timeout)
        if el is None:
            return b""
        box = await _element_box(el)
        if not box:
            return b""
        tab = el.tab if el.tab is not None else self._tab
        return await _element_shot(tab, box)

    async def content_frame(self) -> Optional["_Frame"]:
        el = await self._pick()
        if el is None:
            return None
        return await self._page._frame_for_element(el)

    async def element_handle(self, timeout: Optional[float] = None):
        el = await self._wait_element(timeout)
        if el is None:
            raise TimeoutError(f"locator element_handle: not found for '{self._selector}'")
        return _ElementHandle(self._page, el)


class _ElementHandle:
    """Minimal Playwright ElementHandle needed by the code (content_frame)."""

    def __init__(self, page: "_Page", el: Any):
        self._page = page
        self._el = el

    def __await__(self):
        # truedriver-era callers may `await` the handle; Playwright handles
        # aren't awaitable, so resolve to ourselves.
        async def _self():
            return self
        return _self().__await__()

    async def content_frame(self) -> Optional["_Frame"]:
        return await self._page._frame_for_element(self._el)


class _FrameLocator:
    """Playwright frame_locator — resolves to the frame(s) behind an iframe
    selector, then scopes further selectors inside those frames."""

    def __init__(self, page: "_Page", frame_selector: str):
        self._page = page
        self._frame_selector = frame_selector

    def locator(self, selector: str) -> "_Locator":
        return _Locator(self._page, selector, frame_selector=self._frame_selector)


class _OwnerLocator:
    """Locator for the <iframe> element that owns a given frame, resolved via
    DOM.getFrameOwner."""

    def __init__(self, page: "_Page", frame: "_Frame"):
        self._page = page
        self._tab = page._tab
        self._frame = frame

    def __await__(self):
        # Playwright Locator/FrameLocator objects are awaitable (they resolve
        # to themselves); callers use both `frame.frame_element()` sync and
        # `await frame.frame_element()`.
        async def _self():
            return self
        return _self().__await__()

    async def bounding_box(self):
        try:
            frame_id = await self._frame._page_frame_id()
            if not frame_id:
                return None
            owner = await _frame_owner_bnode(self._tab, frame_id)
            if not owner:
                return None
            ro = await self._tab.send(cdp.dom.resolve_node(
                backend_node_id=cdp.dom.BackendNodeId(owner)))
            quads = await self._tab.send(cdp.dom.get_content_quads(object_id=ro.object_id))
            if not quads:
                return None
            return _quad_to_box(quads[0])
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Frame
# ─────────────────────────────────────────────────────────────────────────────
class _Frame:
    """One frame (main or iframe). For iframes the underlying object is a
    nodriver IFrame — itself a CDP connection (its own flattened session) —
    so evaluation and selectors run in-frame without any frame switching."""

    def __init__(self, page: "_Page", frame: Any, page_frame_id: Optional[str] = None):
        self._page = page
        self._tab = page._tab
        self._frame = frame          # nodriver IFrame (a Tab) or None for main
        self._frame_id = page_frame_id

    def _conn(self):
        """The CDP connection to send in-frame commands on."""
        return self._frame if self._frame is not None else self._tab

    async def _page_frame_id(self) -> Optional[str]:
        return self._frame_id

    @property
    def url(self) -> str:
        try:
            t = getattr(self._frame, "target", None)
            return (getattr(t, "url", None) or "") if t else ""
        except Exception:
            return ""

    @property
    def name(self) -> str:
        try:
            t = getattr(self._frame, "target", None)
            return (getattr(t, "title", None) or "") if t else ""
        except Exception:
            return ""

    def locator(self, selector: str) -> "_Locator":
        return _Locator(self._page, selector, frame=self)

    async def evaluate(self, js: Any, arg: Any = None) -> Any:
        return await _eval_on(self._conn(), js, arg)

    def frame_element(self) -> "_OwnerLocator":
        return _OwnerLocator(self._page, self)

    async def screenshot(self, timeout: Optional[float] = None, **kwargs) -> bytes:
        fe = _OwnerLocator(self._page, self)
        box = await fe.bounding_box()
        if not box:
            return b""
        return await _element_shot(self._tab, box)


async def _frame_owner_bnode(tab, frame_id: str):
    """DOM.getFrameOwner -> backend node id of the <iframe> element (or None).

    The generated CDP binding wants a FrameId object, not a bare string."""
    try:
        owner, _ = await tab.send(cdp.dom.get_frame_owner(cdp.page.FrameId(str(frame_id))))
        return int(owner) if owner else None
    except Exception:
        return None


async def _eval_on(conn, js: Any, arg: Any = None) -> Any:
    """Runtime.evaluate on any connection (tab or frame) with Playwright
    string semantics (bare expression vs function body, arg forwarded)."""
    expr, await_promise = _wrap_js(js, arg)
    ro, errs = await conn.send(cdp.runtime.evaluate(
        expression=expr, user_gesture=True, await_promise=await_promise,
        return_by_value=True, allow_unsafe_eval_blocked_by_csp=True))
    if errs:
        raise RuntimeError(str(errs))
    return ro.value if ro is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response facades for page.on()
# ─────────────────────────────────────────────────────────────────────────────
class _Request:
    def __init__(self, event: Any):
        self.url = (getattr(getattr(event, "request", None), "url", None)) or ""
        self._post_data = getattr(getattr(event, "request", None), "post_data", None)
        self._post_data_buffer = None

    @property
    def post_data(self):
        return self._post_data

    @property
    def post_data_buffer(self):
        if self._post_data_buffer is None and self._post_data is not None:
            self._post_data_buffer = self._post_data.encode("utf-8", "replace")
        return self._post_data_buffer


class _Response:
    def __init__(self, tab: Any, event: Any):
        self.url = (getattr(event, "response", None) and event.response.url) or ""
        self.status = int((getattr(event, "response", None) and event.response.status) or 0)
        self._tab = tab
        self._request_id = getattr(event, "request_id", None)

    async def json(self):
        body, base64_enc = await self._tab.send(
            cdp.network.get_response_body(self._request_id))
        if base64_enc:
            body = base64.b64decode(body).decode("utf-8", "replace")
        return json.loads(body)


def _call_handler(handler: Callable, *args: Any) -> None:
    """Invoke a possibly-sync / possibly-async user handler, scheduling async
    ones on the running loop without blocking the event callback."""
    try:
        res = handler(*args)
        if asyncio.iscoroutine(res):
            asyncio.create_task(res)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────
class _Page:
    def __init__(self, context: "_Context", tab: Any):
        self.context = context
        self._tab = tab
        self._mouse = _Mouse(self)
        self._keyboard = _Keyboard(self)
        # `page.frames` is a sync property (the code reads it without await),
        # so it serves from a cache that navigation / on-demand lookups
        # refresh (throttled — see _refresh_frames).
        self._frames_cache: List["_Frame"] = []
        self._frames_refreshed_at: float = 0.0

    async def _refresh_frames(self, force: bool = False) -> None:
        # Throttled: the nav-poll loop evaluates every ~150 ms, and a full
        # refresh costs two CDP round-trips (Target.getTargets +
        # Page.getFrameTree) — over a slow residential proxy that triples
        # the poll latency. Frames only change on navigation (force) or on
        # demand; a 2s minimum interval keeps the cache warm cheaply.
        now = time.monotonic()
        if not force and now - self._frames_refreshed_at < 2.0:
            return
        self._frames_refreshed_at = now
        try:
            fobs = await self._tab.get_frames()
            frames = [_Frame(self, f) for f in fobs]
            # Map each iframe's CDP target to its Page-domain frame id so
            # DOM.getFrameOwner works (owner lookup for bounding boxes).
            try:
                tree = await self._tab.send(cdp.page.get_frame_tree())
                by_parent = {}
                for fr in _nd_util.flatten_frame_tree(tree):
                    pid = getattr(fr, "parent_id", None)
                    if pid:
                        by_parent[str(pid)] = getattr(fr, "id_", None)
                for f, wrapped in zip(fobs, frames):
                    t = getattr(f, "target", None)
                    pid = getattr(t, "parent_frame_id", None) if t else None
                    if pid and str(pid) in by_parent:
                        wrapped._frame_id = by_parent[str(pid)]
            except Exception:
                pass
            self._frames_cache = frames
        except Exception:
            pass

    # ---- basic properties / navigation ------------------------------------
    @property
    def url(self) -> str:
        try:
            t = getattr(self._tab, "target", None)
            if t is not None:
                return (getattr(t, "url", None) or "")
        except Exception:
            pass
        return ""

    async def title(self) -> str:
        try:
            return str(await self._eval("document.title") or "")
        except Exception:
            return ""

    async def goto(self, url: str, wait_until: Optional[str] = None,
                   timeout: Optional[float] = None, **kwargs) -> None:
        t = (timeout or 30000) / 1000.0
        try:
            # nodriver's tab.get() has no timeout and waits for nothing;
            # navigate directly and bound ourselves (callers additionally
            # wrap goto in their own hard cap).
            await asyncio.wait_for(
                self._tab.send(cdp.page.navigate(url)), timeout=max(t, 5.0))
        except asyncio.TimeoutError:
            raise
        except Exception:
            raise
        # wait_until="domcontentloaded" (the bot's case) -> readyState
        # "interactive" is enough; default -> "complete". Poll briefly — the
        # caller's form-poll is the real gate, so don't over-wait.
        want_complete = (wait_until or "load") == "load"
        deadline = time.monotonic() + max(t, 5.0)
        while time.monotonic() < deadline:
            try:
                rs = await asyncio.wait_for(
                    self._eval("document.readyState"), timeout=3.0)
            except Exception:
                rs = "loading"
            if rs in ("interactive", "complete") and not (want_complete and rs != "complete"):
                break
            if rs == "complete":
                break
            await asyncio.sleep(0.25)
        await self._refresh_frames(force=True)

    async def reload(self, timeout: Optional[float] = None, **kwargs) -> None:
        try:
            await self._tab.reload()
        except Exception:
            pass
        await self._refresh_frames(force=True)

    async def close(self) -> None:
        try:
            await self._tab.close()
        except Exception:
            pass

    def is_closed(self) -> bool:
        try:
            if getattr(self._tab, "socket", None) is None:
                return True
        except Exception:
            pass
        try:
            if getattr(self._tab, "stopped", False):
                return True
        except Exception:
            pass
        return False

    # ---- evaluation -------------------------------------------------------
    async def _eval(self, js: Any, arg: Any = None) -> Any:
        return await _eval_on(self._tab, js, arg)

    async def evaluate(self, js: Any, arg: Any = None) -> Any:
        out = await self._eval(js, arg)
        await self._refresh_frames()
        return out

    # ---- locating ---------------------------------------------------------
    def locator(self, selector: str) -> "_Locator":
        return _Locator(self, selector)

    def frame_locator(self, selector: str) -> "_FrameLocator":
        return _FrameLocator(self, selector)

    @property
    def frames(self) -> List["_Frame"]:
        return self._frames_cache

    async def _frames_for_selector(self, selector: str) -> list:
        try:
            iframes = await self._tab.query_selector_all(selector)
        except Exception:
            return []
        bnodes = set()
        for e in iframes:
            try:
                bnodes.add(int(e.backend_node_id))
            except Exception:
                pass
        if not bnodes:
            return []
        out = []
        for fr in self._frames_cache:
            frame_id = await fr._page_frame_id()
            if not frame_id:
                continue
            owner = await _frame_owner_bnode(self._tab, frame_id)
            if owner and owner in bnodes:
                out.append(fr._frame)
        return out

    async def _frame_for_element(self, el: Any) -> Optional["_Frame"]:
        try:
            bnode = int(el.backend_node_id)
        except Exception:
            return None
        for fr in self._frames_cache:
            frame_id = await fr._page_frame_id()
            if not frame_id:
                continue
            owner = await _frame_owner_bnode(self._tab, frame_id)
            if owner == bnode:
                return fr
        return None

    # ---- screenshots ------------------------------------------------------
    async def screenshot(self, full_page: bool = False, timeout: Optional[float] = None,
                         clip: Optional[dict] = None, **kwargs) -> bytes:
        """Viewport or full-surface PNG capture.

        ``clip`` ({"x","y","width","height"}) captures an explicit region and
        may extend beyond the viewport — server.capture_page_screenshot uses
        it for budgeted full-page frames with a known full size."""
        t = (timeout or 25000) / 1000.0 if timeout else 25.0
        try:
            if full_page or clip:
                try:
                    args = {"format_": "png", "capture_beyond_viewport": True}
                    if clip:
                        args["clip"] = cdp.page.Viewport(
                            x=float(clip.get("x") or 0),
                            y=float(clip.get("y") or 0),
                            width=float(clip.get("width") or 0),
                            height=float(clip.get("height") or 0),
                            scale=1.0)
                    data = await asyncio.wait_for(
                        self._tab.send(cdp.page.capture_screenshot(**args)),
                        timeout=t)
                    if data:
                        return base64.b64decode(data)
                except Exception:
                    pass
            data = await asyncio.wait_for(
                self._tab.send(cdp.page.capture_screenshot(
                    format_="png", capture_beyond_viewport=False
                )),
                timeout=min(t, 10.0)
            )
            if data:
                return base64.b64decode(data)
        except Exception:
            pass
        return b""

    # ---- mouse / keyboard / events ----------------------------------------
    @property
    def mouse(self) -> "_Mouse":
        return self._mouse

    @property
    def keyboard(self) -> "_Keyboard":
        return self._keyboard

    def on(self, event: str, handler: Callable) -> None:
        """Register a Playwright-style event listener.

        Sync (must stay sync: callers use it without await). The Network
        domain is enabled when the tab is prepared (see _Context._prepare_tab)
        so these events flow; crash detection additionally needs the
        Inspector domain, which is enabled lazily here (the bot attaches the
        crash listener right after page creation, inside the worker loop).
        """
        if event in ("request", "response"):
            if event == "request":
                def _req_h(e, _conn=None):
                    _call_handler(handler, _Request(e))
                self._tab.handlers[cdp.network.RequestWillBeSent].append(_req_h)
            else:
                def _resp_h(e, _conn=None):
                    _call_handler(handler, _Response(self._tab, e))
                self._tab.handlers[cdp.network.ResponseReceived].append(_resp_h)
            try:
                asyncio.get_running_loop().create_task(
                    self._tab.send(cdp.network.enable()))
            except RuntimeError:
                pass
        elif event == "crash":
            # Two sources of the same signal: Inspector.targetCrashed on the
            # target session (needs Inspector.enable) and Target.crashed on
            # the flattened auto-attach path. Register both; the bot's crash
            # handler is idempotent (flag + rate-limited diagnostics).
            def _crash_h(e, _conn=None):
                _call_handler(handler)
            self._tab.handlers[cdp.inspector.TargetCrashed].append(_crash_h)
            self._tab.handlers[cdp.target.TargetCrashed].append(_crash_h)
            try:
                asyncio.get_running_loop().create_task(
                    self._tab.send(cdp.inspector.enable()))
            except RuntimeError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# CDP session (used by DragSolver._cdp_drag)
# ─────────────────────────────────────────────────────────────────────────────
class _CDPSession:
    def __init__(self, tab: Any):
        self._tab = tab

    async def send(self, method: str, params: Optional[dict] = None) -> Any:
        def _gen():
            yield {"method": method, "params": params or {}}
        return await self._tab.send(_gen())


# ─────────────────────────────────────────────────────────────────────────────
# Context / Browser / Playwright
# ─────────────────────────────────────────────────────────────────────────────
class _Context:
    def __init__(self, browser: "_Browser", opts: Optional[dict] = None):
        self._browser = browser
        self._opts = opts or {}
        self._init_scripts: list = []
        self._pages: list = []
        self._used_tabs = set()

    async def add_init_script(self, script: str) -> None:
        self._init_scripts.append(script)

    async def _prepare_tab(self, tab: Any) -> None:
        """Apply the context options nodriver does not take at launch.

        · viewport  -> Emulation.setDeviceMetricsOverride (per tab)
        · color_scheme -> Emulation.setEmulatedMedia
        · ENGLISH is forced (operator request) -> Accept-Language header,
          on top of the navigator.language init script server.py injects.
        """
        vp = self._opts.get("viewport") or {}
        try:
            w = int(vp.get("width") or 1280)
            h = int(vp.get("height") or 720)
            await tab.send(cdp.emulation.set_device_metrics_override(
                width=w, height=h, device_scale_factor=1.0, mobile=False))
        except Exception:
            pass
        scheme = self._opts.get("color_scheme")
        if scheme:
            try:
                await tab.send(cdp.emulation.set_emulated_media(
                    features=[cdp.emulation.MediaFeature(
                        name="prefers-color-scheme", value=str(scheme))]))
            except Exception:
                pass
        try:
            await tab.send(cdp.network.set_extra_http_headers(
                headers=cdp.network.Headers({"Accept-Language": "en-US,en;q=0.9"})))
        except Exception:
            pass
        # nodriver does not auto-enable domains — turn on the ones the bot
        # relies on (DOM queries, page events, frame tree, screenshots).
        for domain in (cdp.page.enable(), cdp.dom.enable(), cdp.network.enable()):
            try:
                await tab.send(domain)
            except Exception:
                pass

    async def new_page(self) -> "_Page":
        tab = await self._browser._new_tab(self)
        await self._prepare_tab(tab)
        for s in self._init_scripts:
            src = _wrap_new_document(s)
            if not src:
                continue
            try:
                await tab.send(cdp.page.add_script_to_evaluate_on_new_document(source=src))
            except Exception:
                pass
        page = _Page(self, tab)
        self._pages.append(page)
        try:
            await page._refresh_frames()
        except Exception:
            pass
        return page

    async def new_cdp_session(self, page: Any) -> "_CDPSession":
        return _CDPSession(page._tab)

    async def close(self) -> None:
        for p in list(self._pages):
            try:
                await p.close()
            except Exception:
                pass
        self._pages.clear()


class _Browser:
    def __init__(self, browser: Any):
        self._browser = browser
        self._contexts: list = []

    @property
    def is_connected(self) -> bool:
        if not self._browser:
            return False
        if getattr(self._browser, "stopped", False):
            return False
        return True

    async def new_context(self, **opts) -> "_Context":
        ctx = _Context(self, opts)
        self._contexts.append(ctx)
        return ctx

    async def _new_tab(self, ctx: "_Context") -> Any:
        if not ctx._used_tabs:
            mt = getattr(self._browser, "main_tab", None)
            if mt is not None:
                try:
                    mt.browser = self._browser
                except Exception:
                    pass
                ctx._used_tabs.add(id(mt))
                return mt
        try:
            tab = await self._browser.get("about:blank", new_tab=True)
            tab.browser = self._browser
            return tab
        except Exception:
            mt = getattr(self._browser, "main_tab", None)
            if mt is not None:
                try:
                    mt.browser = self._browser
                except Exception:
                    pass
                return mt
            raise

    async def close(self) -> None:
        try:
            res = self._browser.stop()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass


def _proxy_args(proxy: Optional[dict]) -> List[str]:
    """Playwright-style proxy dict -> Chrome flags.

    Chrome carries proxy credentials via --proxy-user/--proxy-pass (the
    --proxy-server URL form does not authenticate), so build those from the
    dict server.py produces ({server: scheme://host:port, username, password}).
    """
    if not proxy:
        return []
    args = []
    server = str(proxy.get("server") or "").strip()
    if server:
        args.append(f"--proxy-server={server}")
    if proxy.get("username"):
        args.append(f"--proxy-user={proxy['username']}")
        args.append(f"--proxy-pass={proxy.get('password') or ''}")
    return args


class _Chromium:
    async def launch(self, headless: bool = True, args: Optional[list] = None,
                     proxy: Optional[dict] = None, **kwargs) -> "_Browser":
        if uc is None:
            raise RuntimeError(
                "nodriver is not installed — run: pip install nodriver "
                "(and install google-chrome-stable on the machine)")
        browser_args = list(args or [])
        # Honor the context's ignore_https_errors (the bot always sets it) at
        # the browser level — the CDP equivalent of Playwright's option.
        browser_args.append("--ignore-certificate-errors")
        browser_args.extend(_proxy_args(proxy))
        # google-chrome-stable is on PATH in the container; CHROME_PATH
        # overrides for odd deployments. nodriver auto-detects otherwise.
        exe = os.environ.get("CHROME_PATH") or None
        browser = await uc.start(
            headless=bool(headless),
            browser_args=browser_args,
            sandbox=False,
            browser_executable_path=exe,
        )
        return _Browser(browser)


class _Playwright:
    def __init__(self):
        self._started = False
        self._chromium: Optional["_Chromium"] = None

    async def start(self) -> "_Playwright":
        self._started = True
        return self

    async def stop(self) -> None:
        self._started = False

    @property
    def chromium(self) -> "_Chromium":
        if self._chromium is None:
            self._chromium = _Chromium()
        return self._chromium


def async_playwright() -> "_Playwright":
    return _Playwright()
