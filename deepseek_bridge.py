# language: Python, file: deepseek_bridge.py, target: local PC + chat.deepseek.com
"""DeepSeek chat bridge for hCaptcha accessibility / text challenges.

Two modes:
  1) API  — set DEEPSEEK_API_KEY (OpenAI-compatible https://api.deepseek.com)
  2) UI   — drive an already-open chat.deepseek.com tab (or open one)

Worker calls: answer = await ask_deepseek(question, page=optional_browser_page)
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Optional

import aiohttp

DEEPSEEK_API = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_KEY = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
CHAT_URL = os.environ.get("DEEPSEEK_CHAT_URL", "https://chat.deepseek.com/")

_SYSTEM = (
    "You solve short human-verification questions. "
    "Reply with ONLY the final answer text or number — no explanation, no quotes, no markdown."
)


async def ask_deepseek(question: str, page=None, log=None) -> str:
    """Return plain answer string for the challenge question."""
    q = (question or "").strip()
    if not q:
        return ""
    if DEEPSEEK_KEY:
        ans = await _via_api(q, log=log)
        if ans:
            return ans
    if page is not None:
        ans = await _via_ui(page, q, log=log)
        if ans:
            return ans
    if log:
        log("[DeepSeek] no API key and no page — cannot solve", level="warn")
    return ""


async def _via_api(question: str, log=None) -> str:
    url = f"{DEEPSEEK_API}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ],
        "temperature": 0.1,
        "max_tokens": 64,
    }
    try:
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    if log:
                        log(f"[DeepSeek] API HTTP {resp.status}: {text[:200]}", level="warn")
                    return ""
                data = await resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _clean(content)
    except Exception as e:
        if log:
            log(f"[DeepSeek] API error: {e}", level="warn")
        return ""


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```\w*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    t = t.strip().strip('"').strip("'")
    # first line only — models sometimes add a second sentence
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    return t


async def _via_ui(page, question: str, log=None) -> str:
    """Drive chat.deepseek.com: paste question, wait for reply, return text."""
    try:
        # ensure we're on deepseek chat
        url = (page.url or "") if hasattr(page, "url") else ""
        if "deepseek.com" not in url:
            await page.goto(CHAT_URL, timeout=60000)
            await asyncio.sleep(2.5)

        # input box — deepseek chat uses a textarea / contenteditable
        selectors = [
            "textarea",
            "[contenteditable='true']",
            "div.chat-input textarea",
            "#chat-input",
            "textarea[placeholder]",
        ]
        box = None
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0 and await loc.is_visible():
                    box = loc
                    break
            except Exception:
                continue
        if box is None:
            if log:
                log("[DeepSeek] UI: no input box found", level="warn")
            return ""

        await box.click(timeout=5000)
        await box.fill(question, timeout=8000)
        await asyncio.sleep(0.3)
        # send: Enter or send button
        try:
            await page.keyboard.press("Enter")
        except Exception:
            for sel in ("button[type='submit']", "button:has-text('Send')", "[class*='send' i]"):
                try:
                    btn = page.locator(sel).first
                    if await btn.count() > 0:
                        await btn.click(timeout=3000)
                        break
                except Exception:
                    pass

        # wait for a new assistant message
        answer = await _wait_reply(page, timeout=60.0, log=log)
        return _clean(answer)
    except Exception as e:
        if log:
            log(f"[DeepSeek] UI error: {e}", level="warn")
        return ""


async def _wait_reply(page, timeout: float = 60.0, log=None) -> str:
    """Poll for the latest assistant message bubble."""
    deadline = time.time() + timeout
    last = ""
    stable = 0
    while time.time() < deadline:
        try:
            texts = await page.evaluate("""() => {
                const nodes = document.querySelectorAll(
                    '[class*="message" i], [class*="markdown" i], [class*="answer" i], .ds-markdown, .md-content');
                const out = [];
                for (const n of nodes) {
                    const t = (n.innerText || '').trim();
                    if (t.length > 0 && t.length < 500) out.push(t);
                }
                return out.slice(-6);
            }""")
            if texts:
                cand = texts[-1]
                if cand == last and len(cand) > 0:
                    stable += 1
                    if stable >= 3:
                        return cand
                else:
                    last = cand
                    stable = 0
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return last
