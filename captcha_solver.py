# language: Python, file: captcha_solver.py, target: local accessibility + DeepSeek
"""Minimal hCaptcha helpers + accessibility challenge path.

Replaces the old offline/CNN solver package. DiscordAutomation imports:
  extract_hcaptcha_sitekey, read_hcaptcha_token
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

from deepseek_bridge import ask_deepseek


async def extract_hcaptcha_sitekey(page) -> str:
    try:
        key = await page.evaluate("""() => {
            const el = document.querySelector('[data-sitekey], iframe[src*="hcaptcha"]');
            if (el && el.getAttribute('data-sitekey')) return el.getAttribute('data-sitekey');
            const ifr = document.querySelector('iframe[src*="hcaptcha"]');
            if (ifr && ifr.src) {
                const m = ifr.src.match(/sitekey=([^&]+)/);
                if (m) return decodeURIComponent(m[1]);
            }
            return '';
        }""")
        return key or ""
    except Exception:
        return ""


async def read_hcaptcha_token(page) -> str:
    try:
        tok = await page.evaluate("""() => {
            const ta = document.querySelector(
                'textarea[name="h-captcha-response"], [name="g-recaptcha-response"], iframe[data-hcaptcha-response]');
            if (ta) {
                if (ta.tagName === 'TEXTAREA' || ta.tagName === 'INPUT') return ta.value || '';
                return ta.getAttribute('data-hcaptcha-response') || '';
            }
            const hidden = document.querySelector('[name="h-captcha-response"]');
            return hidden ? (hidden.value || '') : '';
        }""")
        return (tok or "").strip()
    except Exception:
        return ""


async def open_accessibility_challenge(page, frame, log=None) -> bool:
    """Click the hCaptcha ⋯ menu and choose Accessibility challenge."""
    try:
        # menu button inside challenge frame
        for sel in (
            'button[aria-label*="Menu" i]',
            'button[title*="Menu" i]',
            '[class*="menu" i] button',
            'button[aria-label*="More" i]',
            '#menu-info',
            '.help-button',
            'button:has-text("⋯")',
            'button:has-text("...")',
        ):
            try:
                btn = frame.locator(sel).first
                if await btn.count() > 0:
                    await btn.click(timeout=3000)
                    await asyncio.sleep(0.6)
                    break
            except Exception:
                continue

        # accessibility option in the dropdown (may be in page or frame)
        for root in (frame, page):
            for sel in (
                'text=Accessibility',
                'text=Accessibility Challenge',
                '[role="menuitem"]:has-text("Accessibility")',
                'button:has-text("Accessibility")',
                'a:has-text("Accessibility")',
            ):
                try:
                    item = root.locator(sel).first
                    if await item.count() > 0:
                        await item.click(timeout=3000)
                        if log:
                            log("[Captcha] Accessibility challenge selected")
                        await asyncio.sleep(1.2)
                        return True
                except Exception:
                    continue
        if log:
            log("[Captcha] Accessibility menu item not found", level="warn")
        return False
    except Exception as e:
        if log:
            log(f"[Captcha] Accessibility open failed: {e}", level="warn")
        return False


async def read_accessibility_question(frame) -> str:
    try:
        return await frame.evaluate("""() => {
            const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
            const sels = [
                '.challenge-prompt', '.prompt-text', '#prompt-text',
                '.task-description', '#task-description',
                '[class*="prompt" i]', 'h1', 'h2', 'p', 'label'
            ];
            let best = '';
            for (const sel of sels) {
                for (const el of document.querySelectorAll(sel)) {
                    const t = norm(el.innerText || el.textContent);
                    if (t.length > best.length && t.length < 400) best = t;
                }
            }
            return best;
        }""") or ""
    except Exception:
        return ""


async def solve_accessibility_with_deepseek(page, frame, log=None, deepseek_page=None) -> bool:
    """Full path: open accessibility if needed, read Q, ask DeepSeek, type answer, verify."""
    question = await read_accessibility_question(frame)
    if not question or len(question) < 5:
        opened = await open_accessibility_challenge(page, frame, log=log)
        if opened:
            await asyncio.sleep(1.0)
            question = await read_accessibility_question(frame)
    if not question:
        if log:
            log("[Captcha] No accessibility question text", level="warn")
        return False
    if log:
        log(f"[Captcha] Accessibility Q: {question[:120]}")

    answer = await ask_deepseek(question, page=deepseek_page, log=log)
    if not answer:
        if log:
            log("[Captcha] DeepSeek returned empty answer", level="warn")
        return False
    if log:
        log(f"[Captcha] DeepSeek answer: {answer!r}")

    # type into accessibility input
    try:
        inp = frame.locator('input[type="text"], input:not([type]), textarea').first
        await inp.click(timeout=4000)
        await inp.fill(answer, timeout=4000)
    except Exception as e:
        if log:
            log(f"[Captcha] Could not type answer: {e}", level="warn")
        return False

    # submit
    for sel in (
        'button[type="submit"]',
        '.button-submit',
        'button:has-text("Verify")',
        'button:has-text("Submit")',
        'button:has-text("Next")',
    ):
        try:
            btn = frame.locator(sel).first
            if await btn.count() > 0:
                await btn.click(timeout=4000)
                break
        except Exception:
            continue
    await asyncio.sleep(1.5)
    tok = await read_hcaptcha_token(page)
    return bool(tok)


def extract_rqdata_from_body(body) -> str:
    """Pull enterprise rqdata from a getcaptcha request body if present."""
    try:
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", errors="ignore")
        if not isinstance(body, str):
            body = str(body)
        # form-encoded or json
        import re, json as _json
        m = re.search(r"rqdata=([^&]+)", body)
        if m:
            from urllib.parse import unquote
            return unquote(m.group(1))
        try:
            data = _json.loads(body)
            if isinstance(data, dict) and data.get("rqdata"):
                return str(data["rqdata"])
        except Exception:
            pass
    except Exception:
        pass
    return ""
