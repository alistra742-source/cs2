# language: Python, file: deepseek_bridge.py, target: chat.deepseek.com user token
"""DeepSeek web-chat bridge using your logged-in user token (not platform API key).

How to get the token:
  1. Log in at https://chat.deepseek.com
  2. F12 → Application → Local Storage → https://chat.deepseek.com
  3. Key `userToken` → copy the JSON `value` field
     OR Network tab → any /api/v0/ request → Request Headers → authorization
        (Bearer xxxxx — paste with or without the Bearer prefix)

Env:
  DEEPSEEK_USER_TOKEN   required — the bearer token string
  DEEPSEEK_SESSION_ID   optional — reuse a chat session id
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Optional

import aiohttp

BASE = "https://chat.deepseek.com/api/v0"
CHAT_ORIGIN = "https://chat.deepseek.com"

_SYSTEM = (
    "You solve short human-verification questions. "
    "Reply with ONLY the final answer text or number — no explanation, no quotes, no markdown."
)


def _token() -> str:
    raw = (os.environ.get("DEEPSEEK_USER_TOKEN") or os.environ.get("DEEPSEEK_TOKEN") or "").strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    return raw


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Origin": CHAT_ORIGIN,
        "Referer": f"{CHAT_ORIGIN}/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/132.0.0.0 Safari/537.36"
        ),
        "X-Client-Platform": "web",
        "X-Client-Locale": "en_US",
        "X-Client-Version": "1.3.0",
    }


async def ask_deepseek(question: str, page=None, log=None) -> str:
    """Answer a short challenge question via chat.deepseek.com user token."""
    q = (question or "").strip()
    if not q:
        return ""
    token = _token()
    if not token:
        if log:
            log("[DeepSeek] set DEEPSEEK_USER_TOKEN (from chat.deepseek.com localStorage userToken)", level="warn")
        return ""
    prompt = f"{_SYSTEM}\n\nQuestion: {q}"
    try:
        answer = await _chat_completion(token, prompt, log=log)
        return _clean(answer)
    except Exception as e:
        if log:
            log(f"[DeepSeek] error: {e}", level="warn")
        return ""


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```\w*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    t = t.strip().strip('"').strip("'")
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    return t


async def _chat_completion(token: str, prompt: str, log=None) -> str:
    headers = _headers(token)
    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        session_id = (os.environ.get("DEEPSEEK_SESSION_ID") or "").strip()
        if not session_id:
            session_id = await _create_session(session, headers, log=log)
            if not session_id:
                raise RuntimeError("could not create chat session (token expired?)")

        body = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": False,
            "search_enabled": False,
        }
        url = f"{BASE}/chat/completion"
        async with session.post(url, headers=headers, json=body) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"completion HTTP {resp.status}: {text[:300]}")
            if "text/event-stream" in ctype or "stream" in ctype:
                return await _read_sse(resp, log=log)
            # some builds return JSON envelope
            data = await resp.json(content_type=None)
            return _extract_json_answer(data)


async def _create_session(session: aiohttp.ClientSession, headers: dict, log=None) -> str:
    url = f"{BASE}/chat_session/create"
    body = {"character_id": None}
    async with session.post(url, headers=headers, json=body) as resp:
        if resp.status != 200:
            text = await resp.text()
            if log:
                log(f"[DeepSeek] create session HTTP {resp.status}: {text[:200]}", level="warn")
            return ""
        data = await resp.json(content_type=None)
    # envelope variants
    biz = data.get("data", {})
    if isinstance(biz, dict):
        biz = biz.get("biz_data", biz)
    sid = ""
    if isinstance(biz, dict):
        sid = biz.get("id") or biz.get("chat_session_id") or ""
        if not sid and isinstance(biz.get("chat_session"), dict):
            sid = biz["chat_session"].get("id", "")
    if not sid:
        sid = data.get("id") or ""
    if log and sid:
        log(f"[DeepSeek] session {sid}")
    return str(sid) if sid else ""


async def _read_sse(resp: aiohttp.ClientResponse, log=None) -> str:
    """Accumulate assistant content from DeepSeek SSE (diff / JSON lines)."""
    chunks: list[str] = []
    async for raw in resp.content:
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            obj = json.loads(line)
        except Exception:
            # plain text fragment
            if line and not line.startswith("{"):
                chunks.append(line)
            continue
        piece = _piece_from_event(obj)
        if piece:
            chunks.append(piece)
    return "".join(chunks)


def _piece_from_event(obj: dict) -> str:
    """Pull text delta from known DeepSeek SSE shapes."""
    if not isinstance(obj, dict):
        return ""
    # direct content
    for key in ("content", "text", "delta", "response"):
        v = obj.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            for k2 in ("content", "text"):
                if isinstance(v.get(k2), str) and v[k2]:
                    return v[k2]
    # choices[0].delta.content (openai-ish)
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0].get("delta") or choices[0].get("message") or {}
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return delta["content"]
    # nested p / v patch style
    if obj.get("p") == "response" or obj.get("type") == "response":
        v = obj.get("v") or obj.get("value")
        if isinstance(v, str):
            return v
    # biz_data path
    data = obj.get("data") or {}
    if isinstance(data, dict):
        biz = data.get("biz_data") or data
        if isinstance(biz, dict):
            for key in ("content", "text", "answer"):
                if isinstance(biz.get(key), str) and biz[key]:
                    return biz[key]
    return ""


def _extract_json_answer(data: dict) -> str:
    if not isinstance(data, dict):
        return str(data or "")
    for key in ("content", "text", "answer", "response"):
        if isinstance(data.get(key), str):
            return data[key]
    biz = data.get("data") or {}
    if isinstance(biz, dict):
        biz = biz.get("biz_data") or biz
        if isinstance(biz, dict):
            for key in ("content", "text", "answer"):
                if isinstance(biz.get(key), str):
                    return biz[key]
    return ""
