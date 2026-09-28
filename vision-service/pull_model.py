#!/usr/bin/env python3
"""Download a public Ollama registry model into Ollama's local model store.

Railway's runtime network currently returns EOF to `ollama pull` while fetching
some registry manifests. This downloader uses the registry's plain OCI HTTP
endpoints, verifies every blob by size and SHA-256, and writes the same on-disk
layout Ollama uses. Partial blob downloads are resumed on the next attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REGISTRY = "registry.ollama.ai"
MODEL_RE = re.compile(
    r"^(?:(?P<namespace>[a-z0-9][a-z0-9._-]*)/)?"
    r"(?P<name>[a-z0-9][a-z0-9._-]*)"
    r"(?::(?P<tag>[A-Za-z0-9][A-Za-z0-9._-]*))?$"
)
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 12


def _request(url: str, *, range_start: int | None = None):
    headers = {
        "User-Agent": "Railway-Ollama-Model-Installer/1.0",
        "Connection": "close",
    }
    if range_start:
        headers["Range"] = f"bytes={range_start}-"
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=90
    )


def _fetch_manifest(url: str) -> bytes:
    error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with _request(url) as response:
                raw = response.read()
            value = json.loads(raw)
            if not isinstance(value, dict) or not isinstance(value.get("layers"), list):
                raise ValueError("Registry returned an invalid model manifest.")
            return raw
        except Exception as exc:  # transient CDN/network failures are retried
            error = exc
            print(f"[model-download] Manifest attempt {attempt}/{MAX_ATTEMPTS} failed: {type(exc).__name__}", flush=True)
            time.sleep(min(attempt * 2, 15))
    raise RuntimeError(f"Could not download model manifest: {type(error).__name__}")


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _download_blob(base_url: str, descriptor: dict[str, Any], blobs_dir: Path) -> None:
    digest_value = descriptor.get("digest", "")
    expected_size = descriptor.get("size")
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", str(digest_value))
    if not match or not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError("Manifest contains an invalid blob descriptor.")

    expected_hash = match.group(1)
    destination = blobs_dir / f"sha256-{expected_hash}"
    partial = blobs_dir / f"sha256-{expected_hash}-partial"

    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"[model-download] Cached blob {expected_hash[:12]} ({expected_size} bytes)", flush=True)
        return
    destination.unlink(missing_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        current_size = partial.stat().st_size if partial.exists() else 0
        if current_size > expected_size:
            partial.unlink()
            current_size = 0
        if current_size == expected_size:
            break

        url = f"{base_url}/blobs/{digest_value}"
        try:
            with _request(url, range_start=current_size) as response:
                status = getattr(response, "status", response.getcode())
                append = current_size > 0 and status == 206
                if current_size > 0 and not append:
                    current_size = 0
                mode = "ab" if append else "wb"
                downloaded = current_size
                next_report = downloaded + 20 * 1024 * 1024
                with partial.open(mode) as stream:
                    while chunk := response.read(CHUNK_SIZE):
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report or downloaded == expected_size:
                            percent = min(100, int(downloaded * 100 / max(expected_size, 1)))
                            print(
                                f"[model-download] {expected_hash[:12]} {percent}% "
                                f"({downloaded}/{expected_size} bytes)",
                                flush=True,
                            )
                            next_report = downloaded + 20 * 1024 * 1024
        except Exception as exc:
            print(
                f"[model-download] Blob {expected_hash[:12]} attempt "
                f"{attempt}/{MAX_ATTEMPTS} interrupted: {type(exc).__name__}",
                flush=True,
            )
            time.sleep(min(attempt * 2, 15))
            continue

        if partial.stat().st_size == expected_size:
            break
        print(
            f"[model-download] Blob {expected_hash[:12]} incomplete; resuming.",
            flush=True,
        )

    if not partial.exists() or partial.stat().st_size != expected_size:
        actual = partial.stat().st_size if partial.exists() else 0
        raise RuntimeError(
            f"Blob {expected_hash[:12]} incomplete after retries "
            f"({actual}/{expected_size} bytes)."
        )
    actual_hash = _digest_file(partial)
    if actual_hash != expected_hash:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Blob {expected_hash[:12]} failed SHA-256 verification.")
    os.replace(partial, destination)
    print(f"[model-download] Verified blob {expected_hash[:12]}", flush=True)


def install(model_name: str) -> None:
    match = MODEL_RE.fullmatch(model_name.strip())
    if not match:
        raise ValueError(
            "Fallback downloader supports Ollama registry names such as "
            "namespace/model:tag."
        )
    namespace = match.group("namespace") or "library"
    name = match.group("name")
    tag = match.group("tag") or "latest"

    models_dir = Path(os.environ.get("OLLAMA_MODELS", "/data/ollama"))
    blobs_dir = models_dir / "blobs"
    manifest_dir = models_dir / "manifests" / REGISTRY / namespace / name
    blobs_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://{REGISTRY}/v2/{namespace}/{name}"
    manifest_bytes = _fetch_manifest(f"{base_url}/manifests/{tag}")
    manifest = json.loads(manifest_bytes)
    descriptors = [manifest.get("config")] + manifest.get("layers", [])
    if not all(isinstance(item, dict) for item in descriptors):
        raise ValueError("Manifest is missing required model descriptors.")

    total = sum(int(item.get("size", 0)) for item in descriptors)
    print(
        f"[model-download] Installing {namespace}/{name}:{tag} "
        f"({total} bytes)",
        flush=True,
    )
    for descriptor in descriptors:
        _download_blob(base_url, descriptor, blobs_dir)

    manifest_path = manifest_dir / tag
    temporary_manifest = manifest_path.with_name(f".{tag}.tmp")
    temporary_manifest.write_bytes(manifest_bytes)
    os.replace(temporary_manifest, manifest_path)
    print(f"[model-download] Installed {namespace}/{name}:{tag}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: pull_model.py namespace/model:tag")
    try:
        install(sys.argv[1])
    except Exception as exc:
        print(f"[model-download] Fatal: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
