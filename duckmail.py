"""
duckmail.py — fast temp-mail REST client using duckmail.sbs.

Provider: duckmail.sbs → glasswhitehub.com domain (Hydra API).

Discord signup flow:
  1. create_inbox()                  -> random addr@glasswhitehub.com
  2. (Discord registration happens)
  3. wait_for_verification_link()    -> polls inbox, extracts the verify URL
"""

import asyncio
import json
import random
import re
import string
import time
from typing import Callable, Optional

import aiohttp

DUCKMAIL_BASE = "https://api.duckmail.sbs"
# duckmail.sbs delivers to @glasswhitehub.com (user-requested domain)
DUCKMAIL_DOMAIN = "glasswhitehub.com"

# Discord verification emails link to:
#   https://discord.com/register/verify?token=...
#   https://discord.com/verify?token=...
#   https://click.discord.com/ls/click?upn=...      (tracking redirect)
#   https://e.discord.com/...                       (email gateway)
VERIFY_LINK_PATTERNS = [
    r"https?://[^\s\"'<>]+discord\.com/register/verify[^\s\"'<>]*",
    r"https?://[^\s\"'<>]+discord\.com/verify[^\s\"'<>]*",
    r"https?://[^\s\"'<>]*(?:click|e)\.discord\.com[^\s\"'<>]*",
    r"https?://[^\s\"'<>]+discord[^\s\"'<>]*(?:verify|verification)[^\s\"'<>]*",
]

_EMAIL_RANDOM = random.SystemRandom()


def _random_password(length: int = 18) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    pw = [_EMAIL_RANDOM.choice(alphabet) for _ in range(length)]
    return "".join(pw)


class TempMail:
    """Async temp-mail client: duckmail.sbs → glasswhitehub.com.

    Never crashes the caller — every failure is caught, logged and surfaced
    as an empty string / None so the automation can react gracefully.
    """

    def __init__(self, log: Optional[Callable] = None):
        self._log = log or (lambda msg, level="info": None)
        self._session: Optional[aiohttp.ClientSession] = None
        self._provider = ""        # 'duckmail'
        self._address = ""
        self._password = ""
        self._token = ""           # JWT for duckmail

    # ── Public API ────────────────────────────────────────

    @property
    def email(self) -> str:
        return self._address

    @property
    def provider(self) -> str:
        return self._provider or "none"

    async def create_inbox(self, timeout: float = 40.0) -> str:
        """Create a fresh mailbox at @glasswhitehub.com via duckmail.sbs."""
        if await self._create_on(DUCKMAIL_BASE, "duckmail", timeout):
            return self._address
        self._log("[Mail] duckmail.sbs unavailable", level="error")
        return ""

    async def wait_for_verification_link(
        self, keyword: str = "discord",
        timeout: float = 240.0, poll: float = 4.0,
    ) -> Optional[str]:
        """Poll the inbox until a matching message arrives, then return its
        Discord verification URL (or None on timeout)."""
        if not self._token:
            self._log("[Mail] No inbox token — cannot poll", level="warn")
            return None
        deadline = time.time() + timeout
        self._log(f"[Mail] Waiting for {keyword} verification email "
                  f"(up to {int(timeout)}s)...")
        while time.time() < deadline:
            try:
                messages = await self._list_messages()
                for msg in messages:
                    if not self._matches(msg, keyword):
                        continue
                    link = self._extract_verify_link(msg)
                    if link:
                        self._log(f"[Mail] [OK] Verification link found: {link[:90]}...")
                        return link
                    self._log("[Mail] Matching message but no verify link in body "
                              "— keeping polling", level="warn")
            except Exception as e:
                self._log(f"[Mail] poll error: {e}", level="warn")
                # token may have expired — try to refresh once
                if self._token and self._provider:
                    await self._login()
            await asyncio.sleep(poll)
        self._log(f"[Mail] No {keyword} verification email within {int(timeout)}s",
                  level="warn")
        return None

    async def close(self) -> None:
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    # ── duckmail.sbs provider (Hydra API) ─────────────────

    async def _create_on(self, base: str, provider: str,
                         timeout: float) -> bool:
        """Create + authorize a mailbox on duckmail.sbs (Hydra API)."""
        deadline = time.time() + timeout
        domain = DUCKMAIL_DOMAIN

        tries = 0
        while time.time() < deadline:
            tries += 1
            rand = "".join(_EMAIL_RANDOM.choices(
                string.ascii_lowercase + string.digits,
                k=_EMAIL_RANDOM.randint(10, 14)))
            address = f"dm{rand}@{domain}"
            password = _random_password()
            status, data = await self._request(
                base, "POST", "/accounts",
                body={"address": address, "password": password},
                auth=False,
            )
            if 200 <= status < 300:
                self._provider = provider
                self._address = address
                self._password = password
                self._token = ""
                self._log(f"[Mail] [OK] Inbox ready: {address} ({provider})")
                await self._login()
                return True
            if status == 422:
                self._log(f"[Mail] {address} taken — retrying", level="warn")
            else:
                self._log(f"[Mail] {provider} create -> {status} "
                          f"{data.get('message', '')}", level="warn")
            await asyncio.sleep(0.6)
        return False

    async def _login(self) -> bool:
        if not self._provider:
            return False
        base = DUCKMAIL_BASE
        status, data = await self._request(
            base, "POST", "/token",
            body={"address": self._address, "password": self._password},
            auth=False,
        )
        tok = data.get("token", "")
        if isinstance(tok, str) and len(tok) > 20:
            self._token = tok
            return True
        self._log(f"[Mail] {self._provider} login failed ({status})", level="warn")
        return False

    # ── Inbox reads ───────────────────────────────────────

    async def _list_messages(self) -> list:
        if not self._provider:
            return []
        base = DUCKMAIL_BASE
        status, data = await self._request(
            base, "GET", "/messages?itemsPerPage=30")
        if status != 200:
            return []
        if isinstance(data, list):
            return data
        member = data.get("hydra:member", [])
        return member if isinstance(member, list) else []

    async def _fetch_message(self, msg_id: str) -> dict:
        if not self._provider:
            return {}
        base = DUCKMAIL_BASE
        status, data = await self._request(base, "GET", f"/messages/{msg_id}")
        return data if isinstance(data, dict) else {}

    # ── Matching / link extraction ────────────────────────

    def _matches(self, msg: dict, keyword: str) -> bool:
        """True when a message is likely the keyword (e.g. Discord) email."""
        text = self._message_text(msg)
        kw = keyword.lower()
        if kw and kw in text.lower():
            return True
        # Sender heuristic as a second chance
        frm = msg.get("from")
        if isinstance(frm, dict):
            sender = str(frm.get("address", "")).lower()
            if kw and kw in sender:
                return True
        return False

    @staticmethod
    def _message_text(msg: dict) -> str:
        parts: list = []
        for key in ("subject", "introduction", "summary"):
            v = msg.get(key)
            if isinstance(v, str):
                parts.append(v)
        body = msg.get("body")
        if isinstance(body, str):
            parts.append(body)
        elif isinstance(body, dict):
            for k in ("html", "text"):
                v = body.get(k)
                if isinstance(v, str):
                    parts.append(v)
        frm = msg.get("from")
        if isinstance(frm, dict):
            parts.append(str(frm.get("address", "")))

        for to in (msg.get("to") or []):
            if isinstance(to, dict):
                parts.append(str(to.get("address", "")))
            elif isinstance(to, str):
                parts.append(to)
        return "\n".join(parts)

    def _extract_verify_link(self, msg: dict) -> Optional[str]:
        text = self._message_text(msg)
        # 1) Direct URL scan (works for plain-text bodies)
        for pattern in VERIFY_LINK_PATTERNS:
            m = re.search(pattern, text, re.I)
            if m:
                return self._clean_link(m.group(0))
        # 2) HTML anchor hrefs
        for m in re.finditer(r'href=["\']([^"\']+)["\']', text, re.I):
            href = self._clean_link(m.group(1))
            h = href.lower()
            if ("discord.com/verify" in h or "discord.com/register" in h
                    or "click.discord.com" in h or "e.discord.com" in h):
                return href
            if "discord.com" in h and ("verify" in h or "confirm" in h):
                return href
        # 3) Escaped/encoded &amp; variants inside HTML body
        for m in re.finditer(r'href=([^\s>]+)', text, re.I):
            href = self._clean_link(m.group(1).strip("'\"'"))
            h = href.lower()
            if "discord.com" in h and "verify" in h:
                return href
        return None

    @staticmethod
    def _clean_link(url: str) -> str:
        return url.replace("&amp;", "&").rstrip(".,);\"'<>")

    # ── HTTP helper (duckmail.sbs) ──────────────────────────

    async def _request(self, base: str, method: str, path: str,
                       body: Optional[dict] = None,
                       auth: bool = True) -> tuple:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json"}
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            async with self._session.request(
                method, f"{base}{path}", json=body, headers=headers,
                timeout=aiohttp.ClientTimeout(total=25),
            ) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text) if text else {}
                except Exception:
                    data = {}
                return resp.status, data
        except Exception as e:
            self._log(f"[Mail] HTTP {method} {path} error: {e}", level="warn")
            return 0, {}
