"""
live_control.py — live-control helpers for the dashboard's LIVE tab.

These are plain module functions (not methods on DiscordAutomation) so the
dashboard can run them on the bot's asyncio loop without touching the big bot
modules. They drive the SAME ``bot._page`` the bot uses — the operator and the
bot share one real Chrome (nodriver) session, and every action here is a
pure read or input over that page (no second browser is ever launched).

Pointer actions (click + drag) are logged with page-space coordinates and
a challenge-iframe screenshot is captured whenever hCaptcha is showing one.
"""
import asyncio
import base64
import hashlib
import json
import math
import os
import time
from typing import Optional

from browser_engine import ENGINE
from server import (
    NAV_TIMEOUT_MS,
    capture_page_screenshot,
    png_dimensions,
    png_is_complete,
)

VIEWPORT_W = 1920
VIEWPORT_H = 1080
DISCORD_REGISTER_URL = "https://discord.com/register"
_CHALLENGE_GALLERY_CAP = 80
CHALLENGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "challenges")

CHALLENGE_IFRAME_SELECTOR = (
    'iframe[title*="hCaptcha challenge"], '
    'iframe[src*="hcaptcha-challenge"]'
)
_POINTER_LOG_CAP = 24


def parse_xy(action: dict, xkey: str = "x", ykey: str = "y") -> tuple:
    """Read a page-space (x, y) pair from a live-action payload."""
    action = action or {}
    try:
        x = float(action.get(xkey, 0) or 0)
    except (TypeError, ValueError):
        x = 0.0
    try:
        y = float(action.get(ykey, 0) or 0)
    except (TypeError, ValueError):
        y = 0.0
    return x, y


def format_click_log(x: float, y: float, selector: str = "") -> str:
    base = f"click at ({x:.0f}, {y:.0f})"
    sel = str(selector or "").strip()
    return f"{base}  {sel}" if sel else base


def format_drag_log(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f"drag ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f}) "
        f"[dx={x2 - x1:.0f}, dy={y2 - y1:.0f}]"
    )


def is_challenge_frame_url(url: str) -> bool:
    u = (url or "").lower()
    return "hcaptcha-challenge" in u or "frame=challenge" in u


def pointer_entry(kind: str, **fields) -> dict:
    """One logged pointer event (JSON-safe).

    Numeric x/y stay rounded floats. Strings (selector, js) and bools
    (is_input) are kept so the LIVE field can show what was clicked.
    """
    out = {"kind": str(kind), "t": time.strftime("%H:%M:%S"), "ts": time.time()}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                out[key] = round(float(value), 1)
                continue
            except (TypeError, ValueError):
                pass
        text = str(value).strip()
        if text:
            out[key] = text[:500]
    return out


def sanitize_hit(raw) -> dict:
    """Keep only the useful fields from a page.evaluate hit describe."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key in ("selector", "js", "click_js", "tag", "type", "name",
                "text", "src"):
        val = raw.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            out[key] = text[:500]
    if raw.get("is_input") in (True, 1, "1", "true", "True"):
        out["is_input"] = True
    if raw.get("iframe") in (True, 1, "1", "true", "True"):
        out["iframe"] = True
    return out


def format_hit_field(entry: dict) -> str:
    """Text for the LIVE 'what I clicked' field."""
    entry = entry or {}
    kind = str(entry.get("kind") or "click")
    lines = []
    if kind == "drag":
        lines.append(format_drag_log(
            entry.get("x1", 0), entry.get("y1", 0),
            entry.get("x2", 0), entry.get("y2", 0)))
    else:
        lines.append(format_click_log(
            entry.get("x", 0), entry.get("y", 0),
            entry.get("selector") or ""))
    if entry.get("selector"):
        lines.append(f"selector: {entry['selector']}")
    if entry.get("js"):
        lines.append(f"js: {entry['js']}")
    if entry.get("click_js"):
        lines.append(f"click: {entry['click_js']}")
    if entry.get("tag"):
        extra = entry.get("type") or entry.get("name") or ""
        lines.append(f"tag: {entry['tag']}" + (f" {extra}" if extra else ""))
    if entry.get("text"):
        lines.append(f"text: {entry['text']}")
    return "\n".join(lines)


def decode_image_payload(image) -> bytes:
    """PNG bytes from a data-URL, raw base64, or already-decoded bytes."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    text = str(image or "").strip()
    if not text:
        return b""
    if text.startswith("data:image/"):
        text = text.split(",", 1)[-1]
    try:
        return base64.b64decode(text)
    except Exception:
        return b""


def image_src(image) -> str:
    """A browser-usable src: data URL, /challenges/ path, or base64 prefix."""
    text = str(image or "").strip()
    if not text:
        return ""
    if text.startswith("data:image/") or text.startswith("/challenges/"):
        return text
    return "data:image/png;base64," + text


def challenge_file_path(name: str) -> str:
    """Absolute path to a saved challenge PNG, or '' if the name is unsafe."""
    name = os.path.basename(str(name or ""))
    if not name or not name.endswith(".png") or ".." in name:
        return ""
    path = os.path.join(CHALLENGE_DIR, name)
    return path if os.path.isfile(path) else ""


def save_challenge_png(image, question: str = "", kind: str = "challenge") -> dict:
    """Write a complete PNG to disk and never delete it.

    Same-hash images reuse the existing file. Incomplete / half frames
    are rejected so a glitched capture cannot replace a good one.
    """
    raw = decode_image_payload(image)
    if not png_is_complete(raw):
        return {}
    width, height = png_dimensions(raw)
    if width < 40 or height < 40:
        return {}
    digest = hashlib.sha1(raw).hexdigest()[:16]
    try:
        os.makedirs(CHALLENGE_DIR, exist_ok=True)
    except Exception:
        return {}
    existing = ""
    try:
        for fname in os.listdir(CHALLENGE_DIR):
            if digest in fname and fname.endswith(".png"):
                existing = fname
                break
    except Exception:
        existing = ""
    if existing:
        name = existing
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_kind = "".join(ch for ch in str(kind or "challenge") if ch.isalnum()) or "shot"
        name = f"{stamp}-{safe_kind}-{digest}.png"
        path = os.path.join(CHALLENGE_DIR, name)
        try:
            with open(path, "wb") as handle:
                handle.write(raw)
        except Exception:
            return {}
    return {
        "id": digest,
        "file": name,
        "url": f"/challenges/{name}",
        "question": str(question or "").strip()[:300],
        "time": time.strftime("%H:%M:%S"),
        "kind": str(kind or "challenge"),
        "width": width,
        "height": height,
        "bytes": len(raw),
    }


def remember_saved_challenge(bot, rec: dict) -> None:
    """Append a saved-challenge record; never drop previous files."""
    if bot is None or not rec or not rec.get("id"):
        return
    gallery = list(getattr(bot, "_live_challenge_gallery", None) or [])
    if any(item.get("id") == rec["id"] for item in gallery):
        return
    gallery.append(rec)
    if len(gallery) > _CHALLENGE_GALLERY_CAP:
        gallery = gallery[-_CHALLENGE_GALLERY_CAP:]
    try:
        bot._live_challenge_gallery = gallery
    except Exception:
        pass


def format_pointer_dump(items) -> str:
    """Copy-all text: every logged click/drag with selector + js."""
    rows = []
    for entry in list(items or []):
        if not isinstance(entry, dict):
            continue
        block = format_hit_field(entry)
        t = entry.get("t") or ""
        rows.append(f"{t} {block}".strip() if t else block)
    return "\n\n".join(rows)


# elementFromPoint + a short CSS path. Cross-origin iframes only report
# the iframe itself (we cannot read into hCaptcha from the parent).
HIT_AT_JS = r"""([xy]) => {
    const x = Number(xy && xy[0]), y = Number(xy && xy[1]);
    const root = document;
    const el = (root.elementFromPoint && root.elementFromPoint(x, y)) || null;
    if (!el) return JSON.stringify({});
    const cssEscape = (s) => String(s || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    const cssPath = (node) => {
        if (!node || node.nodeType !== 1) return '';
        if (node.id && /^[A-Za-z][\w-]*$/.test(node.id)
                && document.querySelectorAll('#' + node.id).length === 1) {
            return '#' + node.id;
        }
        const parts = [];
        let cur = node;
        while (cur && cur.nodeType === 1 && cur !== document.documentElement
                && parts.length < 6) {
            let part = (cur.tagName || '').toLowerCase();
            if (cur.id && /^[A-Za-z][\w-]*$/.test(cur.id)) {
                parts.unshift(part + '#' + cur.id);
                break;
            }
            const name = cur.getAttribute && cur.getAttribute('name');
            const aria = cur.getAttribute && cur.getAttribute('aria-label');
            const typ = cur.getAttribute && cur.getAttribute('type');
            if (name) part += '[name="' + cssEscape(name) + '"]';
            else if (aria) part += '[aria-label="' + cssEscape(aria).slice(0, 40) + '"]';
            else if (typ) part += '[type="' + cssEscape(typ) + '"]';
            else if (cur.classList && cur.classList.length) {
                const cls = [];
                for (let i = 0; i < cur.classList.length && cls.length < 2; i++) {
                    const c = cur.classList[i];
                    if (c && !c.includes(':') && !c.startsWith('_') && /^[A-Za-z][\w-]*$/.test(c))
                        cls.push(c);
                }
                if (cls.length) part += '.' + cls.join('.');
            }
            const parent = cur.parentElement;
            if (parent) {
                const same = [];
                for (let i = 0; i < parent.children.length; i++) {
                    if (parent.children[i].tagName === cur.tagName) same.push(parent.children[i]);
                }
                if (same.length > 1) part += ':nth-of-type(' + (same.indexOf(cur) + 1) + ')';
            }
            parts.unshift(part);
            cur = parent;
        }
        return parts.join(' > ');
    };
    const sel = cssPath(el);
    const tag = (el.tagName || '').toLowerCase();
    const typ = (el.getAttribute && el.getAttribute('type')) || el.type || '';
    const name = (el.getAttribute && el.getAttribute('name')) || el.name || '';
    const text = String((el.innerText || el.value || el.getAttribute && el.getAttribute('aria-label') || '') || '')
        .replace(/\s+/g, ' ').trim().slice(0, 80);
    const isInput = !!(el.matches && el.matches(
        'input:not([type="hidden"]):not([type="submit"]):not([type="button"]):not([type="checkbox"]):not([type="radio"]), textarea, select, [contenteditable="true"]'));
    const js = sel ? ('document.querySelector(' + JSON.stringify(sel) + ')') : '';
    return JSON.stringify({
        selector: sel,
        js: js,
        click_js: js ? (js + ' && ' + js + '.click()') : '',
        tag: tag,
        type: typ,
        name: name,
        text: text,
        src: (tag === 'iframe' && el.src) ? String(el.src).slice(0, 180) : '',
        is_input: isInput ? 1 : 0,
        iframe: tag === 'iframe' ? 1 : 0
    });
}"""


async def describe_element_at(page, x: float, y: float) -> dict:
    """CSS selector + JS snippet for the node under page-space (x, y)."""
    if page is None:
        return {}
    try:
        raw = await asyncio.wait_for(
            page.evaluate(HIT_AT_JS, [float(x), float(y)]), timeout=3.0)
    except Exception:
        return {}
    return sanitize_hit(raw)


def _record_pointer(bot, entry: dict) -> None:
    if bot is None or not entry:
        return
    log = list(getattr(bot, "_live_pointer_log", None) or [])
    log.append(entry)
    if len(log) > _POINTER_LOG_CAP:
        log = log[-_POINTER_LOG_CAP:]
    try:
        bot._live_pointer_log = log
        bot._live_last_pointer = entry
    except Exception:
        pass


def _live_log(bot, message: str, level: str = "info") -> None:
    if bot is None:
        return
    try:
        bot._log(f"[Live] {message}", level=level)
    except Exception:
        pass


def _push_trainer_pointer(entry: dict) -> None:
    try:
        import trainer
        trainer.trainer_engine.note_pointer(entry)
    except Exception:
        pass


def _push_trainer_shot(image: str, question: str = "") -> None:
    if not image:
        return
    try:
        import trainer
        trainer.trainer_engine.note_live_challenge(image, question)
    except Exception:
        pass


def _attach_pointer_fields(meta: dict, bot) -> dict:
    meta = meta or {}
    meta["last_pointer"] = getattr(bot, "_live_last_pointer", None) or None
    meta["pointer_log"] = list(getattr(bot, "_live_pointer_log", None) or [])
    gallery = list(getattr(bot, "_live_challenge_gallery", None) or [])
    meta["challenge_shots"] = gallery
    latest = getattr(bot, "_live_challenge_b64", "") or ""
    if not latest and gallery:
        latest = gallery[-1].get("url") or ""
    meta["challenge_screenshot"] = latest
    return meta


async def live_meta(bot) -> dict:
    """Cheap metadata read (no screenshot) for the LIVE tab."""
    page = getattr(bot, "_page", None)
    connected = False
    url = ""
    title = ""
    if page is not None:
        try:
            href = await asyncio.wait_for(
                page.evaluate("location.href"), timeout=3.0)
            url = str(href or "") or (page.url or "")
            connected = True
        except Exception:
            try:
                url = page.url or ""
            except Exception:
                url = ""
        try:
            title = str(
                await asyncio.wait_for(page.title(), timeout=3.0) or "")
        except Exception:
            title = ""
    try:
        dsf = float(getattr(bot, "_fingerprint", {}).get("pixel_ratio", 1.0) or 1.0)
    except Exception:
        dsf = 1.0
    meta = {
        "connected": connected,
        "url": url,
        "title": title,
        "viewport_width": VIEWPORT_W,
        "viewport_height": VIEWPORT_H,
        "device_scale_factor": dsf,
        "browser": ENGINE,
        "worker_id": getattr(bot, "worker_id", ""),
    }
    return _attach_pointer_fields(meta, bot)


async def live_screenshot(bot) -> str:
    """Full browser-view PNG -> base64 for the live feed.

    Frames are the whole browser window, with the register form revealed
    when it is out of sight (whole scrollable page only when
    FULLPAGE_SHOTS=1) — see server.capture_page_screenshot for the
    capture policy and the OOM safety net. Failures log the EXACT reason
    so the dashboard's ALL LOGS shows why a frame is missing instead of
    silently sitting on 'waiting for frame'.
    """
    page = getattr(bot, "_page", None)
    if page is None:
        return ""
    last_err = ""
    try:
        shot = await capture_page_screenshot(
            page, fullpage_timeout=10.0, viewport_timeout=6.0, reveal="live")
        if not shot:
            last_err = "empty capture"
        elif not png_is_complete(shot):
            last_err = "incomplete png"
            shot = b""
        else:
            width, height = png_dimensions(shot)
            if width < 200 or height < 200:
                last_err = f"tiny frame {width}x{height}"
                shot = b""
    except Exception as e:
        last_err = str(e)
        shot = b""
    if shot and not last_err:
        b64 = base64.b64encode(shot).decode("utf-8")
        shots = getattr(bot, "_screenshots", None)
        if shots is not None:
            # Same tiny-ring policy as server.capture_screenshot (memory).
            shots.append(b64)
            if len(shots) > 10:
                bot._screenshots = shots[-8:]
        return b64
    try:
        bot._log(f"[Live] screenshot failed: {last_err}", level="warn")
    except Exception:
        pass
    return ""


async def _refresh_frames(page) -> None:
    try:
        refresh = getattr(page, "_refresh_frames", None)
        if callable(refresh):
            await refresh(force=True)
            return
    except Exception:
        pass
    try:
        await page.evaluate("document.readyState")
    except Exception:
        pass


async def _challenge_iframe_present(page) -> bool:
    if page is None:
        return False
    try:
        loc = page.locator(CHALLENGE_IFRAME_SELECTOR)
        if await loc.count() > 0:
            return True
    except Exception:
        pass
    try:
        for frame in list(getattr(page, "frames", None) or []):
            try:
                if is_challenge_frame_url(frame.url or ""):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


async def capture_challenge_screenshot(page) -> str:
    """PNG data-URL of the live hCaptcha challenge iframe, or empty."""
    if page is None:
        return ""
    await _refresh_frames(page)
    shot = b""
    try:
        loc = page.locator(CHALLENGE_IFRAME_SELECTOR)
        if await loc.count() > 0:
            try:
                shot = await loc.first.screenshot(timeout=7000)
            except Exception:
                shot = b""
    except Exception:
        shot = b""
    if not shot:
        try:
            for frame in list(getattr(page, "frames", None) or []):
                try:
                    if not is_challenge_frame_url(getattr(frame, "url", "") or ""):
                        continue
                    shot = await frame.screenshot(timeout=7000)
                except Exception:
                    shot = b""
                if shot:
                    break
        except Exception:
            shot = b""
    if not png_is_complete(shot):
        return ""
    width, height = png_dimensions(shot)
    if width < 40 or height < 40:
        return ""
    return "data:image/png;base64," + base64.b64encode(shot).decode("ascii")


async def _read_challenge_prompt(page) -> str:
    """Best-effort visible instruction from the live challenge frame."""
    if page is None:
        return ""
    try:
        import trainer
    except Exception:
        trainer = None
    try:
        for frame in list(getattr(page, "frames", None) or []):
            try:
                if not is_challenge_frame_url(getattr(frame, "url", "") or ""):
                    continue
                text = ""
                try:
                    text = await frame.locator("body").inner_text()
                except Exception:
                    try:
                        text = str(await frame.evaluate(
                            "() => document.body ? document.body.innerText : ''"
                        ) or "")
                    except Exception:
                        text = ""
                if trainer is not None:
                    question = trainer._question_from_text(text)
                else:
                    question = " ".join(str(text or "").split())[:500]
                if question:
                    return question
            except Exception:
                continue
    except Exception:
        pass
    return ""


async def _note_challenge_shot(bot, page) -> str:
    """Capture the challenge iframe (if any), save it, and keep it."""
    if page is None:
        return getattr(bot, "_live_challenge_b64", "") or ""
    try:
        present = await asyncio.wait_for(
            _challenge_iframe_present(page), timeout=1.5)
    except Exception:
        present = False
    if not present:
        return getattr(bot, "_live_challenge_b64", "") or ""
    try:
        image = await asyncio.wait_for(
            capture_challenge_screenshot(page), timeout=4.0)
    except Exception:
        image = ""
    if not image:
        return getattr(bot, "_live_challenge_b64", "") or ""
    question = ""
    try:
        question = await asyncio.wait_for(
            _read_challenge_prompt(page), timeout=2.0)
    except Exception:
        question = ""
    rec = save_challenge_png(image, question, kind="challenge") or {}
    if rec:
        remember_saved_challenge(bot, rec)
        try:
            bot._live_challenge_file = rec.get("file") or ""
        except Exception:
            pass
    published = rec.get("url") or image
    try:
        bot._live_challenge_b64 = published
    except Exception:
        pass
    # Trainer re-saves from the original data URL (same hash reuses the file).
    _push_trainer_shot(image, question)

    # Companion full-viewport of the same moment. Never deletes the iframe
    # PNG — a second hash just adds another kept file.
    companion = ""
    try:
        last = ""
        shots = getattr(bot, "_screenshots", None)
        if shots:
            last = shots[-1] or ""
        raw = decode_image_payload(last)
        if png_is_complete(raw) and png_dimensions(raw)[0] >= 200:
            companion = last
        else:
            shot = await capture_page_screenshot(
                page, reveal="live",
                fullpage_timeout=8.0, viewport_timeout=5.0)
            if png_is_complete(shot):
                companion = (
                    "data:image/png;base64,"
                    + base64.b64encode(shot).decode("ascii")
                )
    except Exception:
        companion = ""
    if companion:
        brow = save_challenge_png(companion, question, kind="browser") or {}
        remember_saved_challenge(bot, brow)

    digest = rec.get("id") or ""
    if digest and digest != getattr(bot, "_live_challenge_saved_id", ""):
        try:
            bot._live_challenge_saved_id = digest
        except Exception:
            pass
        extra = f" — {question}" if question else ""
        fname = rec.get("file") or ""
        where = f" saved as {fname}" if fname else ""
        _live_log(bot, f"Captured hCaptcha challenge screenshot{where}{extra}.")
    return published


async def get_live_state(bot) -> dict:
    meta = await live_meta(bot)
    shot = ""
    page = getattr(bot, "_page", None)
    if meta["connected"]:
        shot = await live_screenshot(bot)
        if not shot:
            # Never flash a black screen just because one capture failed —
            # keep the last good frame the bot already has.
            try:
                shot = bot.get_latest_screenshot() or ""
            except Exception:
                shot = ""
        try:
            ch = await _note_challenge_shot(bot, page)
            if ch:
                meta["challenge_screenshot"] = ch
        except Exception:
            pass
    meta["screenshot"] = shot
    return _attach_pointer_fields(meta, bot)


def _dead_page(url: str) -> bool:
    """True when the page is a browser error/blank page (proxy tunnel died,
    site unreachable, or navigation never happened) rather than real content.

    Engine-agnostic: Chromium (the engine in use) spells a dead tunnel
    chrome-error://chromewebdata/; about:neterror:: is kept for robustness
    in case a future engine swap lands on Firefox.
    """
    u = (url or "").lower()
    if "chrome-error" in u or "neterror" in u or "err_tunnel" in u or "err_" in u:
        return True
    if u in ("", "about:blank") or u.startswith("about:"):
        return True
    return False


async def _wait_for_content(page, timeout: float = 6.0) -> bool:
    """True once the page has actually painted non-empty text. Discord is a
    SPA, so readyState can be 'complete' while React hasn't rendered yet —
    give it a short grace window before declaring the page blank."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            has = await asyncio.wait_for(page.evaluate(
                "() => { const b = document.body; "
                "return !!(b && (b.innerText || '').trim().length); }"),
                timeout=2.0)
            if has:
                return True
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return False


async def live_navigate(bot, url: str) -> dict:
    page = getattr(bot, "_page", None)
    if page is None:
        meta = await live_meta(bot)
        meta["error"] = "browser not started"
        return meta
    try:
        await page.goto(url, wait_until="domcontentloaded",
                        timeout=NAV_TIMEOUT_MS)
    except Exception as e:
        meta = await live_meta(bot)
        meta["error"] = f"navigation failed: {e}"
        return meta
    meta = await get_live_state(bot)
    # goto() can 'succeed' straight onto a browser error page (chrome-error://
    # on Chromium; about:neterror on other engines) when the proxy CONNECT
    # tunnel is dead — treat that as a navigation failure so the caller can
    # rotate the session.
    if _dead_page(meta.get("url", "")):
        meta["error"] = "site unreachable (proxy tunnel failed)"
        return meta
    # A white screen can also mean navigation never happened (parked on
    # about:blank) or the SPA failed to paint — catch that too.
    if not await _wait_for_content(page):
        meta["error"] = "page rendered blank (no content)"
    return meta


async def _live_click(page, x: float, y: float) -> None:
    x, y = float(x), float(y)
    try:
        await page.mouse.move(x, y, steps=4)
    except Exception:
        pass
    await page.mouse.click(x, y)


async def _live_drag(page, x1: float, y1: float, x2: float, y2: float) -> None:
    """Real press-move-release drag (hCaptcha ignores a click at the drop)."""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    try:
        import human_mouse as hm
        await hm.drag(page, (x1, y1), (x2, y2))
        return
    except Exception:
        pass
    await page.mouse.move(x1, y1)
    await asyncio.sleep(0.08)
    await page.mouse.down()
    dist = math.hypot(x2 - x1, y2 - y1)
    steps = max(12, min(48, int(dist / 10.0) or 12))
    try:
        await page.mouse.move(x2, y2, steps=steps)
    except Exception:
        await page.mouse.move(x2, y2)
    await asyncio.sleep(0.06)
    await page.mouse.up()


async def _live_key(page, key: str) -> None:
    key = str(key or "")
    if not key:
        return
    specials = {"Enter", "Backspace", "Tab", "Escape", "Delete",
                "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                "Home", "End", "PageUp", "PageDown", "Space"}
    if key == " ":
        await page.keyboard.press("Space")
    elif key in specials or len(key) > 1:
        await page.keyboard.press(key)
    else:
        # Printable character: insert as a real text input event so
        # React-controlled fields (Discord included) pick it up.
        await page.keyboard.type(key, delay=0)


async def perform_live_action(page, action: dict) -> dict:
    """Run one pointer/keyboard action. Returns a pointer entry or {error}."""
    action = action or {}
    kind = str(action.get("action", "") or "")
    if kind == "back":
        await page.evaluate("window.history.back()")
        return {"kind": "back"}
    if kind == "forward":
        await page.evaluate("window.history.forward()")
        return {"kind": "forward"}
    if kind == "reload":
        await page.reload(timeout=NAV_TIMEOUT_MS)
        return {"kind": "reload"}
    if kind == "click":
        x, y = parse_xy(action)
        await _live_click(page, x, y)
        rec = pointer_entry("click", x=x, y=y)
        hit = await describe_element_at(page, x, y)
        if hit:
            rec.update(hit)
        return rec
    if kind == "drag":
        x1, y1 = parse_xy(action, "x1", "y1")
        if "x1" not in action and "x" in action:
            x1, y1 = parse_xy(action, "x", "y")
        x2, y2 = parse_xy(action, "x2", "y2")
        if "x2" not in action and action.get("to"):
            x2, y2 = parse_xy(action.get("to") or {}, "x", "y")
        await _live_drag(page, x1, y1, x2, y2)
        return pointer_entry("drag", x1=x1, y1=y1, x2=x2, y2=y2)
    if kind == "mousedown":
        x, y = parse_xy(action)
        await page.mouse.move(x, y)
        await page.mouse.down()
        return pointer_entry("mousedown", x=x, y=y)
    if kind == "mousemove":
        x, y = parse_xy(action)
        await page.mouse.move(x, y)
        return pointer_entry("mousemove", x=x, y=y)
    if kind == "mouseup":
        x, y = parse_xy(action)
        await page.mouse.move(x, y)
        await page.mouse.up()
        return pointer_entry("mouseup", x=x, y=y)
    if kind == "scroll":
        dy = float(action.get("delta_y", action.get("deltaY", 0)) or 0)
        await page.evaluate(f"window.scrollBy(0, {dy})")
        return {"kind": "scroll", "delta_y": dy}
    if kind == "key":
        await _live_key(page, action.get("key", ""))
        return {"kind": "key", "key": str(action.get("key", "") or "")}
    if kind == "type":
        await page.keyboard.type(str(action.get("text", "")), delay=0)
        return {"kind": "type"}
    return {"kind": kind or "unknown", "error": f"unknown action: {kind}"}


async def live_action(bot, action: dict) -> dict:
    page = getattr(bot, "_page", None)
    if page is None:
        meta = await live_meta(bot)
        meta["error"] = "browser not started"
        meta["screenshot"] = ""
        return meta
    action = action or {}
    kind = str(action.get("action", ""))
    err = None
    rec: Optional[dict] = None
    try:
        rec = await perform_live_action(page, action)
        if rec and rec.get("error"):
            err = rec["error"]
    except Exception as e:
        err = f"action {kind} failed: {e}"
        rec = {"kind": kind, "error": err}

    if rec and rec.get("kind") in ("click", "drag", "mousedown", "mouseup"):
        _record_pointer(bot, rec)
        _push_trainer_pointer(rec)
        if rec.get("kind") == "click":
            _live_log(bot, format_click_log(
                rec.get("x", 0), rec.get("y", 0), rec.get("selector") or ""))
            if rec.get("js"):
                _live_log(bot, f"js {rec.get('js')}")
        elif rec.get("kind") == "drag":
            _live_log(bot, format_drag_log(
                rec.get("x1", 0), rec.get("y1", 0),
                rec.get("x2", 0), rec.get("y2", 0)))
        elif rec.get("kind") in ("mousedown", "mouseup"):
            _live_log(bot, f"{rec['kind']} at ({rec.get('x', 0):.0f}, {rec.get('y', 0):.0f})")

    # Visual actions refresh the frame immediately (with a screenshot) so the
    # operator SEES the result of the click; bare-meta responses were blanking
    # the feed to "waiting for frame". Input actions (key/scroll/type) stay
    # fast — the next 1.4s poll refreshes the frame for those.
    visual = kind in ("click", "drag", "mousedown", "mouseup",
                      "back", "forward", "reload")
    if visual:
        st = await get_live_state(bot)
    else:
        st = await live_meta(bot)
    if rec:
        st["last_pointer"] = rec if rec.get("kind") in (
            "click", "drag", "mousedown", "mousemove", "mouseup") else (
            getattr(bot, "_live_last_pointer", None) or rec)
    if err:
        st["error"] = err
    return st
