#!/usr/bin/env python3
"""Authenticated gateway in front of a small local Ollama vision model.

Two API surfaces:

1. Ollama-compatible (what the captcha bot's vision_solver speaks):
     GET  /api/tags  -> proxied to local Ollama (model list)
     POST /api/chat  -> proxied to local Ollama, with the model field
                        FORCED to this service's configured model
                        (fixed-model server-side, as before)
   Both require the API key. This is a plain authenticated proxy to the
   local Ollama — the bot's prompts are whatever the operator's own
   automation sends.

2. Legacy safe endpoint (lawful captioning/OCR/visual Q&A only):
     GET  /          -> health/info (open)
     POST /v1/analyze-> fixed-model analysis; REFUSES CAPTCHA /
                        security-challenge prompts (_BLOCKED_TASK).

Raw Ollama is only bound to 127.0.0.1 inside the container; nothing
unauthenticated reaches the model.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import re
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = (
    os.environ.get("OLLAMA_MODEL")
    or os.environ.get("MODEL")
    or os.environ.get("DEFAULT_OLLAMA_MODEL")
    or "ahmadwaqar/smolvlm2-256m-video:q8_0"
).strip()
# Normalize accidental whitespace introduced while copying a Railway secret.
# The client strips its environment value too; doing the same here prevents an
# invisible trailing newline from turning two otherwise-identical keys into a
# permanent HTTP 401. Never log this value.
API_KEY = os.environ.get("VISION_API_KEY", "").strip()
# The external/public URL clients use to reach this gateway.  Set by Railway
# as RAILWAY_PUBLIC_DOMAIN, or override manually with VISION_API_BASE.
VISION_API_BASE = (
    os.environ.get("VISION_API_BASE")
    or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    or ""
).rstrip("/")
# SmolVLM2 yes/no is a few seconds. 180s pins the worker after the bot
# has already moved on (the old 9-image JSON path 504'd here).
REQUEST_TIMEOUT = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "45"))
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "8"))
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "4000"))
# Base64 expands data by roughly 4/3. Leave room for JSON and multiple images.
MAX_REQUEST_BYTES = int(
    os.environ.get(
        "MAX_REQUEST_BYTES",
        str((MAX_IMAGE_BYTES * MAX_IMAGES * 4 // 3) + 1024 * 1024),
    )
)

_BLOCKED_TASK = re.compile(
    r"(?:"
    r"\b(?:hcaptcha|recaptcha|captcha|turnstile)\b|"
    r"\bsecurity\s+challenge\b|"
    r"\bhuman\s+verification\b|"
    r"\bverification\s+(?:puzzle|grid|challenge)\b|"
    r"\bselect\s+all\s+(?:images|squares|tiles)\b|"
    r"\b(?:solve|bypass|defeat)\b.{0,30}\b(?:challenge|verification)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_PROMPT = (
    "You are a compact vision assistant for lawful image captioning, OCR, "
    "document analysis, and visual question answering. Be factual, say when "
    "an image is unclear, and do not claim certainty you do not have. Do not "
    "solve CAPTCHAs, human-verification puzzles, or security challenges."
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _ollama(path: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> dict[str, Any]:
    data = None if payload is None else _json_bytes(payload)
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _decode_image(raw: Any) -> str:
    """Validate one base64 image and return normalized base64 for Ollama."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Each image must be a non-empty base64 string.")
    value = raw.strip()
    if value.startswith("data:"):
        try:
            header, value = value.split(",", 1)
        except ValueError as exc:
            raise ValueError("Malformed image data URI.") from exc
        if ";base64" not in header.lower():
            raise ValueError("Image data URIs must use base64 encoding.")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image is not valid base64.") from exc
    if not decoded:
        raise ValueError("Image data is empty.")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError(f"Each image must be at most {MAX_IMAGE_BYTES} bytes.")
    return base64.b64encode(decoded).decode("ascii")


class VisionHandler(BaseHTTPRequestHandler):
    server_version = "SafeVisionGateway/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Never log request bodies, prompts, images, or authorization headers.
        print(f"[vision-service] {self.address_string()} - {format % args}", flush=True)

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = _json_bytes(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        if not API_KEY:
            return False
        authorization = self.headers.get("Authorization", "")
        bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        provided = bearer or self.headers.get("X-API-Key", "")
        return bool(provided) and hmac.compare_digest(provided, API_KEY)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            raise ValueError("Content-Length is required.")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes.")
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid UTF-8 JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError("Request JSON must be an object.")
        return value

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/":
            info = {
                "service": "small-vision-ai",
                "model": OLLAMA_MODEL,
                "endpoint": "POST /v1/analyze",
                "ollama_compatible": ["GET /api/tags", "POST /api/chat"],
                "authentication": "Bearer token or X-API-Key",
            }
            if VISION_API_BASE:
                info["base_url"] = VISION_API_BASE
            self._send(HTTPStatus.OK, info)
            return
        if self.path == "/api/tags":
            # Ollama-compatible model list (the bot's vision_solver probes
            # this to confirm the endpoint is up). Authenticated proxy to
            # local Ollama.
            if not self._authorized():
                self._send(HTTPStatus.UNAUTHORIZED, {"error": "Invalid or missing API key."})
                return
            try:
                self._send(HTTPStatus.OK, _ollama("/api/tags"))
            except urllib.error.HTTPError as exc:
                print(f"[vision-service] Ollama /api/tags returned HTTP {exc.code}", flush=True)
                self._send(HTTPStatus.BAD_GATEWAY, {"error": "The model server rejected the request."})
            except (urllib.error.URLError, TimeoutError):
                self._send(HTTPStatus.GATEWAY_TIMEOUT, {"error": "The model server is unavailable."})
            except Exception as exc:
                print(f"[vision-service] /api/tags error: {type(exc).__name__}", flush=True)
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Internal server error."})
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/api/chat":
            # Ollama-compatible chat: the bot's vision_solver posts native
            # Ollama payloads (system+user messages, base64 images,
            # format:"json"). Forward to local Ollama with the model field
            # FORCED to this service's configured model (fixed-model
            # server-side), and return Ollama's response verbatim.
            if not self._authorized():
                self._send(HTTPStatus.UNAUTHORIZED, {"error": "Invalid or missing API key."})
                return
            try:
                payload = self._read_json()
                messages = payload.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty array.")
                payload["model"] = OLLAMA_MODEL
                self._send(
                    HTTPStatus.OK,
                    _ollama("/api/chat", payload=payload, timeout=REQUEST_TIMEOUT),
                )
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except urllib.error.HTTPError as exc:
                print(f"[vision-service] Ollama /api/chat returned HTTP {exc.code}", flush=True)
                self._send(HTTPStatus.BAD_GATEWAY, {"error": "The model rejected the request."})
            except (urllib.error.URLError, TimeoutError):
                self._send(HTTPStatus.GATEWAY_TIMEOUT, {"error": "The model timed out or is unavailable."})
            except Exception as exc:
                print(f"[vision-service] /api/chat error: {type(exc).__name__}", flush=True)
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Internal server error."})
            return
        if self.path != "/v1/analyze":
            self._send(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        if not self._authorized():
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "Invalid or missing API key."})
            return

        try:
            body = self._read_json()
            prompt = body.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string.")
            prompt = prompt.strip()
            if len(prompt) > MAX_PROMPT_CHARS:
                raise ValueError(f"prompt must be at most {MAX_PROMPT_CHARS} characters.")
            if _BLOCKED_TASK.search(prompt):
                self._send(
                    HTTPStatus.FORBIDDEN,
                    {"error": "CAPTCHA and security-challenge solving is not supported."},
                )
                return

            raw_images = body.get("images")
            if raw_images is None and body.get("image") is not None:
                raw_images = [body.get("image")]
            if not isinstance(raw_images, list) or not raw_images:
                raise ValueError("Provide image or a non-empty images array.")
            if len(raw_images) > MAX_IMAGES:
                raise ValueError(f"At most {MAX_IMAGES} images are allowed per request.")
            images = [_decode_image(image) for image in raw_images]

            max_tokens = body.get("max_tokens", 384)
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
                raise ValueError("max_tokens must be an integer.")
            if max_tokens < 1 or max_tokens > 768:
                raise ValueError("max_tokens must be between 1 and 768.")

            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt, "images": images},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                },
                "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
            }
            result = _ollama("/api/chat", payload=payload, timeout=REQUEST_TIMEOUT)
            answer = (result.get("message") or {}).get("content", "")
            self._send(
                HTTPStatus.OK,
                {
                    "model": OLLAMA_MODEL,
                    "response": answer,
                    "done": bool(result.get("done", True)),
                },
            )
        except ValueError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except urllib.error.HTTPError as exc:
            print(f"[vision-service] Ollama returned HTTP {exc.code}", flush=True)
            self._send(HTTPStatus.BAD_GATEWAY, {"error": "The model rejected the request."})
        except (urllib.error.URLError, TimeoutError):
            self._send(HTTPStatus.GATEWAY_TIMEOUT, {"error": "The model timed out or is unavailable."})
        except Exception as exc:
            print(f"[vision-service] Unexpected error: {type(exc).__name__}", flush=True)
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Internal server error."})


class VisionServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("VISION_API_KEY is required.")
    url_part = f" ({VISION_API_BASE})" if VISION_API_BASE else ""
    print(f"[vision-service] Gateway ready on {HOST}:{PORT}{url_part} using {OLLAMA_MODEL}", flush=True)
    VisionServer((HOST, PORT), VisionHandler).serve_forever()
