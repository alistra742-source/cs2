import asyncio
import base64
import json
import os
import random
import re
import threading
import time
from typing import Dict, List, Optional

import aiohttp
import requests as _requests

from flask import Flask, jsonify, request, Response, send_file

# Database removed - no token persistence

try:
    from proxies import (pool as proxy_pool, configured as _proxies_configured,
                         proxy_files, proxy_files_signature, used_store)
    _proxies_available = True
except ImportError:
    proxy_pool = None
    used_store = None
    _proxies_configured = lambda: False
    _proxies_available = False
    print("[app] proxies.py not found - direct connections only", flush=True)

# "force use the proxies no matter what" — when residential sessions are
# configured (proxies.txt in the repo, or VAULTPROXY_* env) the workers
# NEVER fall back to TOR. Set PROXY_MODE=force to force even without a file.
PROXY_FORCE = (
    (os.environ.get("PROXY_MODE") or "").strip().lower()
    in ("force", "1", "true", "yes")
    or _proxies_configured()
)

# Fall back to TOR (socks5://127.0.0.1:9050) when the proxy pool is
# exhausted or every session is dead (e.g. vaultproxies at 0.00 GB quota).
# Disable with TOR_FALLBACK=0.
TOR_FALLBACK = (os.environ.get("TOR_FALLBACK") or "").strip().lower() not in ("0", "false", "no", "off")

from server import DiscordAutomation, _tor_check, ENGINE
import live_control
import live_ui
import trainer

# ── Global state (Flask thread + asyncio thread) ──

_loop: Optional[asyncio.AbstractEventLoop] = None
_running = False
_start_time = 0.0

# worker_id -> worker state
_workers: Dict[str, dict] = {}
# Active worker tasks are retained so Stop can cancel in-flight browser work
# immediately instead of waiting for the current navigation or retry to end.
_worker_tasks: Dict[str, asyncio.Task] = {}
WORKER_COUNT = 1

# Railway's 1 GB container can OOM when a browser renderer and a large proxy
# TLS sweep start together. Low-memory mode is on by default; a single worker
# continues to probe each selected proxy, while bulk validation waits until the
# browser is idle.
LOW_MEMORY_MODE = (os.environ.get("LOW_MEMORY_MODE") or "1").strip().lower() not in ("0", "false", "no", "off")
LOW_MEMORY_SWEEP_DELAY_S = max(15.0, float(os.environ.get("LOW_MEMORY_SWEEP_DELAY_S", "60")))
LOW_MEMORY_SWEEP_CONCURRENCY = max(1, min(8, int(os.environ.get("LOW_MEMORY_SWEEP_CONCURRENCY", "4"))))
WORKER_IDS = [f"B{i+1}" for i in range(WORKER_COUNT)]

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# duckmail.sbs delivers inboxes on the Discord-friendly domain
# @glasswhitehub.com. Extra inbox domains can be listed in config
# (mail_domains); a domain is burned at runtime when a signup ends in phone
# verification so it is never reused.
DEFAULT_MAIL_DOMAIN = "glasswhitehub.com"

DEFAULT_CONFIG = {
    "headless": True,
    "web_port": 8080,
    "camera_interval": 3,
    "worker_count": WORKER_COUNT,
    "mail_domains": ["glasswhitehub.com"],
    "custom_email": "",
}

def load_config(path: str = _config_path) -> dict:
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                saved = json.load(f)
                for key, value in DEFAULT_CONFIG.items():
                    if key in saved:
                        config[key] = saved[key]
        except Exception:
            pass
    # This deployment is intentionally single-worker. Ignore any legacy
    # value saved in config.json so status and runtime always agree.
    config["worker_count"] = WORKER_COUNT
    config["web_port"] = int(os.environ.get("PORT", config.get("web_port", 8080)))
    return config

def save_config(config: dict, path: str = _config_path) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass

# Shared ring buffer so app-level lines (proxy stats, worker outcomes) reach
# the web terminal, not just stdout.
_APP_LOGS: list = []

# Domains that got burned at runtime (phone verification on signup). Seeded
# from config.json and persisted back so burned domains stay burned.
# Already-tried domains (mikerossy.com, blobers.it.com, vibify.cc, vibeify.cc)
# are burned up front so a stale config.json can never revive them.
_BURNED_DOMAINS: set = {
    "mikerossy.com", "blobers.it.com", "vibify.cc", "vibeify.cc",
}

def _load_burned() -> None:
    global _BURNED_DOMAINS
    _BURNED_DOMAINS = {"mikerossy.com", "blobers.it.com", "vibify.cc", "vibeify.cc"}
    try:
        cfg = load_config()
        _BURNED_DOMAINS.update(
            str(d).strip().lower() for d in (cfg.get("burned_domains") or []))
    except Exception:
        pass

def _burn_domain(domain: str) -> None:
    """Permanently remove a domain from the pool after a phone-verification hit."""
    d = (domain or "").strip().lower()
    if not d:
        return
    _BURNED_DOMAINS.add(d)
    try:
        cfg = load_config()
        burned = [str(x).strip().lower() for x in (cfg.get("burned_domains") or [])]
        if d not in burned:
            burned.append(d)
        cfg["burned_domains"] = burned
        mail = [str(x).strip().lower() for x in (cfg.get("mail_domains") or [])]
        if d in mail:
            mail.remove(d)
        cfg["mail_domains"] = mail
        save_config(cfg)
    except Exception:
        pass

def _pick_domain(cfg: dict) -> str:
    """Pick a fresh, non-burned inbox domain from the configured list (falls
    back to duckmail's default @glasswhitehub.com)."""
    pools = [
        [str(x).strip().lower() for x in (cfg.get("mail_domains") or []) if str(x).strip()],
        [DEFAULT_MAIL_DOMAIN],
    ]
    for pool in pools:
        fresh = [d for d in pool if d not in _BURNED_DOMAINS]
        if fresh:
            return random.choice(fresh)
    return DEFAULT_MAIL_DOMAIN

# App-level logs: proxy sweeps, AI warm-up, worker chatter etc. only appear
# in the ALL logs (LOG_LEVEL=all). Warnings / errors always print.
_APP_LOG_ALL = os.environ.get("LOG_LEVEL", "").strip().lower() \
    in ("all", "debug", "verbose")

def _log(msg: str, level: str = "info"):
    # Store EVERYTHING so the dashboard's ALL LOGS toggle can show it; only
    # print warnings/errors to the console (and everything with LOG_LEVEL=all).
    essential = level in ("warn", "error")
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "timestamp": time.time(),
        "level": level,
        "essential": essential,
        "message": msg,
    }
    _APP_LOGS.append(entry)
    if len(_APP_LOGS) > 400:
        _APP_LOGS[:] = _APP_LOGS[-400:]
    if _APP_LOG_ALL or essential:
        print(f"[{level.upper()}] {msg}", flush=True)

# ── Worker management (runs in the asyncio thread) ──

def _init_worker(wid: str) -> dict:
    return {
        "id": wid,
        "bot": None,
        "status": "idle",          # idle | starting | running | live | demo | done | error
        "step": "",
        "email": "",
        "username": "",
        "password": "",
        "token": "",
        "proxy": "",
        "started_at": 0,
        "finished_at": 0,
        "screenshots": 0,
        "last_shot_b64": "",
        "launching": False,
        "capture_task": None,
    }

async def _worker_capture_loop(wid: str, cfg: dict, stagger: int) -> None:
    """Capture screenshots for this worker, staggered so browsers don't all
    upload at the same time: B1 immediately, B2 after 2s, B3 after 4s..."""
    bot: DiscordAutomation = _workers[wid]["bot"]
    base = max(1, int(cfg.get("camera_interval", 3)))
    # One image every base seconds across ALL browsers: each worker waits
    # base * len(WORKER_IDS) between its own uploads, staggered base apart.
    interval = base * len(WORKER_IDS)
    await asyncio.sleep(stagger)
    while _running and bot is not None and bot._page is not None:
        # Screenshot encoding can briefly allocate a full viewport buffer. Do
        # not compete with the most memory-sensitive stage: initial React
        # navigation and hydration. The Live modal resumes 3-second frames as
        # soon as the register page has rendered.
        if LOW_MEMORY_MODE and not bool(getattr(bot, "_nav_ok", False)):
            await asyncio.sleep(interval)
            continue
        try:
            shot = await asyncio.wait_for(bot.capture_screenshot(), timeout=25)
            if shot:
                _workers[wid]["last_shot_b64"] = shot
                _workers[wid]["screenshots"] += 1
        except Exception:
            pass
        await asyncio.sleep(interval)

async def _next_proxy(force: bool = False):
    """Grab the least-recently-used proxy session, or None (auto mode only —
    the caller may fall back to TOR). In force mode the pool is refreshed on
    demand so proxies are ALWAYS used and the bot never silently goes TOR."""
    if not (_proxies_available and proxy_pool is not None):
        return None
    if proxy_pool.count == 0:
        try:
            await proxy_pool.refresh()
        except Exception:
            pass
    if proxy_pool.count > 0:
        return proxy_pool.take()
    return None

async def _probe_gated_proxy(wid: str, bot, tries: int = 2):
    """Draw the next session and only accept it if it probes live against
    discord.com (the same gate the worker loop uses). Dead sessions are
    blacklisted as they're found, so an expired proxies.txt can't trap the
    LIVE tab in a browser-error loop. Returns a proven-live proxy or None."""
    for _ in range(tries):
        try:
            proxy = await _next_proxy(force=PROXY_FORCE)
        except Exception:
            proxy = None
        if proxy is None:
            return None
        if (bot.proxy or {}).get("key") == proxy.get("key"):
            return proxy  # already the session the browser is on
        if proxy_pool is None:
            return proxy
        try:
            live = await asyncio.wait_for(proxy_pool.probe(proxy), timeout=5.0)
        except Exception:
            live = False
        if live:
            return proxy
        _log(f"[{wid}] [Live] Probe failed - dead session, blacklisting {proxy.get('key','?')[:44]}", level="info")
        try:
            proxy_pool.release(proxy, ok=False)
        except Exception:
            pass
    return None

def _proxy_stats_line(wid: str) -> None:
    """Log live proxy usage counters (used / working / failed) for the terminal."""
    try:
        s = proxy_pool.stats() if proxy_pool is not None else {}
    except Exception:
        return
    _log(
        f"[{wid}] [Proxy] Used {s.get('used', 0)} sessions, "
        f"Working {s.get('working', 0)}, Failed {s.get('failed', 0)}"
    )

async def _run_worker(wid: str, cfg: dict, proxy=None) -> None:
    """Worker loop: one proxy session per signup attempt, rotating on failure.
    Falls back to proxy pool + TOR as needed."""
    state = _workers[wid]
    state["status"] = "starting"
    state["started_at"] = time.time()
    max_tries = 30 if PROXY_FORCE else 12

    # ── Reuse a browser parked on Discord by a previous run ──
    # Stop leaves the browser ALIVE on discord.com; Start picks it up here
    # instead of cold-launching Brave + CDP (the slow part). A dead parked
    # browser (circuit dropped, browser crashed) is closed and relaunched.
    bot = state.get("bot")
    if bot is not None:
        try:
            bot._stopped.clear()
        except Exception:
            pass
        if not await bot.is_alive():
            _log(f"[{wid}] Parked browser died while stopped - launching fresh", level="warn")
            try:
                await bot.close()
            except Exception:
                pass
            bot = None
            state["bot"] = None
        else:
            _log(f"[{wid}] Reusing parked browser (already on Discord) - skipping cold launch")

    consecutive_tunnel_fails = 0  # fast-fail after consecutive dead connections
    backoff = 0.3  # seconds between attempts; doubles after dead-session failures
    tor_fallback = False  # flipped to True when the proxy pool proves dead → TOR

    for attempt in range(max_tries):
        if not _running:
            state["status"] = "stopped"
            # PARK the browser on Discord — the next Start reuses it
            # (is_alive() gates the reuse; a dead one gets relaunched).
            return

        # Re-read the config on every attempt so a custom email / headless
        # change made in the dashboard mid-run is picked up on the very next
        # attempt (a stale cfg would keep using the old email forever).
        try:
            cfg = load_config()
        except Exception:
            pass

        # ── Pick a session for this attempt ──
        if proxy is None and not tor_fallback:
            proxy = await _next_proxy(force=PROXY_FORCE)
        if proxy is None and not tor_fallback:
            if TOR_FALLBACK and _tor_check():
                tor_fallback = True
                _log(f"[{wid}] [Proxy] No usable proxy sessions — falling back to TOR (socks5://127.0.0.1:9050)", level="warn")
            else:
                _log(f"[{wid}] [Proxy] No proxy sessions (forced mode) — refreshing and waiting...", level="warn")
                state["proxy"] = "waiting-for-proxy"
                await asyncio.sleep(5)
                continue
        state["proxy"] = proxy.get("key", "tor") if proxy else "tor"

        label = state["proxy"]

        # ── Fast liveness probe BEFORE launching a browser ──
        # A dead session costs ~10s+ when we only discover it after the
        # browser boots and the goto times out. Probing first (3s cap, plain
        # HTTP round-trip through the session) blacklists dead sessions in
        # seconds and skips the browser launch entirely for them. Skipped for
        # the same session being reused (already proven live this round).
        if (proxy and proxy.get("host")
                and (bot is None or (bot.proxy or {}).get("key") != proxy.get("key"))
                and proxy_pool is not None):
            try:
                probe_ok = await proxy_pool.probe(proxy)
            except Exception:
                probe_ok = False
            if not probe_ok:
                _log(f"[{wid}] [Proxy] Probe failed — session dead, blacklisting {proxy.get('key','?')[:44]}...", level="info")
                proxy_pool.release(proxy, ok=False)
                proxy = None
                consecutive_tunnel_fails += 1
                backoff = min(backoff * 2, 8)
                if consecutive_tunnel_fails >= 4:
                    if TOR_FALLBACK and _tor_check() and not tor_fallback:
                        tor_fallback = True
                        proxy = None
                        consecutive_tunnel_fails = 0
                        _log(f"[{wid}] [Proxy] All proxy sessions appear dead — falling back to TOR (socks5://127.0.0.1:9050)", level="info")
                        _proxy_stats_line(wid)
                        await asyncio.sleep(1)
                        continue
                    _log(f"[{wid}] {consecutive_tunnel_fails} consecutive tunnel failures — aborting (all sessions appear dead)")
                    break
                _proxy_stats_line(wid)
                await asyncio.sleep(backoff)
                continue

        # ── Launch or reuse browser (fresh domain each attempt) ──
        domain = _pick_domain(cfg)
        if bot is None:
            bot = DiscordAutomation(
                headless=cfg.get("headless", True),
                domain=domain,
            )
            state["bot"] = bot
            try:
                await bot.initialize()
            except Exception as e:
                state["status"] = "error"
                _log(f"[{wid}] Browser launch failed: {e}", level="error")
                if proxy:
                    proxy_pool.release(proxy, ok=False)
                    proxy = None
                await asyncio.sleep(3)
                continue
        else:
            # Reuse browser: rotate to a fresh session / TOR circuit
            # Custom email (if set) always wins; otherwise blank = fresh inbox.
            bot._email = cfg.get("custom_email") or ""
            bot._domain = domain
            bot.phone_verify_detected = False
            bot._nav_ok = False
            if (proxy is not None and bot.proxy is not None
                    and proxy.get("key") == bot.proxy.get("key")):
                # Same sticky session — the browser is ALREADY on Discord.
                # Keep the page and just re-navigate; no context rebuild, no
                # bounce through about:blank.
                _log(f"[{wid}] Same proxy session reused - keeping browser on Discord")
            elif not await bot.switch_proxy(proxy):
                _log(f"[{wid}] Context rebuild failed", level="warn")
                if proxy:
                    proxy_pool.release(proxy, ok=False)
                    proxy = None
                consecutive_tunnel_fails += 1
                backoff = min(backoff * 2, 8)
                if consecutive_tunnel_fails >= 4:
                    if TOR_FALLBACK and _tor_check() and not tor_fallback:
                        tor_fallback = True
                        proxy = None
                        consecutive_tunnel_fails = 0
                        _log(f"[{wid}] [Proxy] All proxy sessions appear dead — falling back to TOR (socks5://127.0.0.1:9050)", level="info")
                        _proxy_stats_line(wid)
                        await asyncio.sleep(1)
                        continue
                    _log(f"[{wid}] {consecutive_tunnel_fails} consecutive tunnel failures — aborting (all sessions appear dead)")
                    break
                await asyncio.sleep(backoff)
                continue

        # ── Run signup ──
        try:
            state["status"] = "running"
            stagger = int(wid[1:]) - 1
            cam_task = asyncio.create_task(
                _worker_capture_loop(wid, cfg, stagger * int(cfg.get("camera_interval", 3))),
                name=f"capture-{wid}",
            )
            state["capture_task"] = cam_task
            try:
                ok = await bot.start_discord_signup()
            finally:
                # A cancelled worker does not implicitly cancel child tasks.
                # Stop must therefore end this screenshot loop explicitly.
                cam_task.cancel()
                await asyncio.gather(cam_task, return_exceptions=True)
                if state.get("capture_task") is cam_task:
                    state["capture_task"] = None

            # ── Capture final screenshot for the LIVE BROWSER view ──
            try:
                if bot is not None and bot._page is not None:
                    shot = await asyncio.wait_for(bot.capture_screenshot(), timeout=25)
                    if shot:
                        state["last_shot_b64"] = shot
                        state["screenshots"] += 1
            except Exception:
                pass

            # ── Phone verification hit → burn this domain + rotate everything ──
            if not ok and getattr(bot, "phone_verify_detected", False):
                _burn_domain(domain)
                _log(f"[{wid}] [Phone] Domain {domain} burned - proxy+fingerprint+domain will rotate", level="warn")

            # ── Clean up temp-mail session between attempts to prevent
            # aiohttp connector leaks (each failed attempt creates a new
            # duckmail inbox that must be closed).
            if bot._mail is not None:
                try:
                    await bot._mail.close()
                except Exception:
                    pass
                bot._mail = None

            acc = bot.get_account()
            state["email"] = acc["email"]
            state["username"] = acc["username"]
            state["password"] = acc["password"]
            state["token"] = acc["token"]
            if ok and acc["token"]:
                state["status"] = "done"

                _log(f"[{wid}] Done - token {len(acc['token'])} chars ({label})")
                if proxy:
                    proxy_pool.release(proxy, ok=True)
                    # Record the session's REAL egress IP (resolved by the
                    # browser) in the persistent store so future redeploys
                    # can see + skip this sticky IP.
                    try:
                        if (used_store is not None and bot is not None
                                and getattr(bot, "_exit_ip", "")):
                            await used_store.record(
                                proxy.get("key"), "valid", bot._exit_ip)
                    except Exception:
                        pass
                _proxy_stats_line(wid)
                # Park the browser on Discord (account visible in LIVE BROWSER)
                # so the next Start reuses it. The next run's switch_proxy
                # rotates to a fresh context/IP anyway.
                return
            elif ok:
                state["status"] = "done"
                _log(f"[{wid}] Signup ok (no token yet)")
                # Account was created but the token isn't there yet (usually a
                # custom email the user must verify manually). Persist it as
                # pending so it is never lost — the user clicks the verify
                # link in their own inbox and the account is theirs.
                # Park the browser on Discord for reuse on the next Start.
                return

            # ── Track consecutive tunnel failures ──
            # With residential proxies (proxy dict), 4 dead sessions in a row
            # means the pool is dry — abort early instead of burning attempts.
            # With TOR (no proxy), every attempt gets a fresh exit node via
            # _tor_newnym() — each circuit is independent, so short backoffs
            # and no early abort; let max_tries (12) run its course.
            using_tor = proxy is None and getattr(bot, "_tor_enabled", False)
            nav_ok = bool(getattr(bot, "_nav_ok", False))
            if not ok and not nav_ok:
                consecutive_tunnel_fails += 1
                if using_tor:
                    backoff = min(backoff * 2, 2.0)    # TOR: fast rotate, tiny backoff
                    abort_at = max_tries               # never early-abort on TOR
                else:
                    backoff = min(backoff * 2, 8)      # residential: longer cooldown
                    abort_at = 4                       # dry pool → stop fast
                if consecutive_tunnel_fails >= abort_at:
                    if (not using_tor and TOR_FALLBACK and _tor_check() and not tor_fallback):
                        tor_fallback = True
                        consecutive_tunnel_fails = 0
                        _log(f"[{wid}] [Proxy] All proxy sessions appear dead — falling back to TOR (socks5://127.0.0.1:9050)", level="info")
                    else:
                        reason = "all TOR circuits blocked" if using_tor else "all sessions appear dead"
                        _log(f"[{wid}] {consecutive_tunnel_fails} consecutive tunnel failures — aborting ({reason})")
                        break
            else:
                consecutive_tunnel_fails = 0
                backoff = 0.3

            # ── Per-attempt failure summary — self-explanatory ──
            nav_error = getattr(bot, "_nav_error", "") or ""
            has_email = bool(getattr(bot, "_email", ""))
            mail_failed = bool(getattr(bot, "_mail_failed", False))
            if mail_failed:
                reason = "inbox creation failed — no email available"
            elif not nav_ok:
                reason = nav_error or "navigation failed (no reason recorded)"
            elif not ok:
                reason = "signup failed (form/captcha/phone)"
            else:
                reason = "unknown"
            _log(f"[{wid}] Attempt {attempt+1}/{max_tries}: {reason} [{label}]", level="warn")
        except Exception as e:
            state["status"] = "error"
            _log(f"[{wid}] error: {e}")
        # Session failed — release it so the next attempt rotates to a new one
        if proxy:
            proxy_pool.release(proxy, ok=False)
            proxy = None
        _proxy_stats_line(wid)
        await asyncio.sleep(backoff)

    state["status"] = "error"
    state["step"] = "retries exhausted - all proxy/TOR attempts failed"
    # Park the browser on Discord — the next Start reuses it (or relaunches
    # if it died while parked). No close() here.
    if bot:
        state["bot"] = bot

def _browser_busy() -> bool:
    return any((state or {}).get("status") in ("starting", "running")
               for state in _workers.values())


async def _deferred_proxy_sweep(n_sessions: int) -> None:
    """Validate a large proxy pool only while the renderer is idle.

    Workers retain their one-at-a-time live probe, so deferring this optional
    dashboard-wide sweep does not weaken the connection gate for an attempt.
    """
    if proxy_pool is None or not n_sessions:
        return
    if LOW_MEMORY_MODE:
        _log(
            f"[Proxy] Low-memory mode: deferred {n_sessions}-session sweep; "
            "workers still probe their selected session individually"
        )
        await asyncio.sleep(LOW_MEMORY_SWEEP_DELAY_S)
        while _running and _browser_busy():
            _log("[Proxy] Low-memory mode: browser active, delaying bulk validation")
            await asyncio.sleep(LOW_MEMORY_SWEEP_DELAY_S)
        if not _running:
            return
        concurrency = LOW_MEMORY_SWEEP_CONCURRENCY
        window = max(20.0, LOW_MEMORY_SWEEP_DELAY_S)
    else:
        concurrency = None
        window = 10.0

    try:
        kwargs = {"window": window, "log": _log}
        if concurrency is not None:
            kwargs["concurrency"] = concurrency
        _log(f"[Proxy] Background sweep of {n_sessions} sessions against discord.com "
             f"(window={int(window)}s, concurrency={concurrency or 'default'})...")
        sw = await proxy_pool.sweep(**kwargs)
        _log(f"[Proxy] Sweep done: {sw['reachable']} Discord-reachable, "
             f"{sw['unproven']} unproven (available, re-checked on use), "
             f"{sw['untested']} untested of {n_sessions}")
        if sw.get("tested") and not sw.get("reachable"):
            _log("[Proxy] Bulk validation found no reachable sessions; workers will continue individual probe-gating", level="warn")
    except Exception as e:
        _log(f"[Proxy] Sweep error: {e}", level="warn")


async def _proxy_validate_loop() -> None:
    """Background: re-confirm which proxies can reach Discord, using the
    worker's single-shot HTTPS probe. Dead sessions get blacklisted so
    workers never waste a browser launch on them. Runs every 3 minutes."""
    while True:
        try:
            if _proxies_available and proxy_pool is not None and proxy_pool.count:
                # Count truly Discord-reachable sessions (proven by sweep
                # or worker success), not the "all loaded" default.
                reachable = sum(
                    1 for p in proxy_pool._proxies
                    if p.get("_valid") and p.get("key") not in proxy_pool._failed
                )
                bl = len(proxy_pool._failed)
                _log(f"[Proxy] Live validation: {reachable} Discord-reachable, "
                     f"{bl} blacklisted of {proxy_pool.count} loaded")
        except Exception as e:
            _log(f"[Proxy] validation error: {e}", level="warn")
        await asyncio.sleep(180)

async def _proxy_file_watcher(interval: float = 15.0) -> None:
    """Reload the proxy pool the moment proxies.txt / vaultproxies.txt changes.

    vaultproxies sessions carry a ~10 min TTL, so a list loaded at startup is
    stale within minutes. This watcher lets the user drop a FRESH session list
    into proxies.txt and have the bot pick it up without restarting the web
    server. Only triggers when the file CONTENT actually changes (re-saving
    the same expired list is a no-op)."""
    try:
        sig = proxy_files_signature()
    except Exception:
        sig = ""
    while True:
        await asyncio.sleep(interval)
        try:
            new_sig = proxy_files_signature()
            if new_sig and new_sig != sig:
                sig = new_sig
                if proxy_pool is not None:
                    await proxy_pool.refresh()
                    n = proxy_pool.count
                    src = ", ".join(p.name for p in proxy_files()) or "env"
                    _log(f"[Proxy] proxies file changed — reloaded {n} sessions from {src}", level="warn")
                    if n:
                        if LOW_MEMORY_MODE:
                            _log("[Proxy] Low-memory mode: re-sweep deferred until browser is idle")
                        else:
                            try:
                                sw = await proxy_pool.sweep(window=10.0, log=_log)
                                _log(f"[Proxy] Re-sweep: {sw['reachable']} Discord-reachable of {n} reloaded")
                            except Exception:
                                pass
        except Exception as e:
            _log(f"[Proxy] file watcher error: {e}", level="warn")

async def _start_all_async(cfg: dict) -> None:
    global _running, _start_time
    if _running:
        return
    _running = True
    _start_time = time.time()

    for wid in WORKER_IDS:
        # Preserve a browser parked on Discord by a previous run so Start
        # reuses it instantly instead of cold-launching Brave + CDP.
        parked = (_workers.get(wid) or {}).get("bot")
        _workers[wid] = _init_worker(wid)
        if parked is not None:
            _workers[wid]["bot"] = parked
            _log(f"[{wid}] Browser parked from previous run - will reuse it on Discord")

    # ── Proxy pool: load free + residential sessions (retry a few times) ──
    n_sessions = 0
    try:
        if _proxies_available and proxy_pool is not None:
            for _r in range(3):
                await proxy_pool.refresh()
                n_sessions = proxy_pool.count
                if n_sessions:
                    break
                await asyncio.sleep(2)
    except Exception as e:
        _log(f"[Proxy] pool refresh error: {e}", level="warn")
    if n_sessions:
        _src = ", ".join(p.name for p in proxy_files()) or "VAULTPROXY_* env"
        _log((f"[Proxy] {n_sessions} proxy sessions loaded from {_src} — "
              f"one IP per account (forced mode)") if PROXY_FORCE
             else f"[Proxy] {n_sessions} proxy sessions loaded from {_src} — one IP per account")
    elif PROXY_FORCE:
        _log("[Proxy] [ERROR] PROXY FORCE MODE but 0 sessions loaded — workers will keep retrying, TOR is DISABLED", level="error")
    else:
        _log("[Proxy] No proxy sessions — TOR-only fallback (fresh circuit per attempt)")

    # Watch for the user dropping a FRESH session list into proxies.txt —
    # TTL sessions expire ~10 min after issuance, so hot-reload beats restart.
    asyncio.create_task(_proxy_file_watcher())

    if n_sessions and _proxies_available and proxy_pool is not None:
        # ── Start workers IMMEDIATELY — they self-probe proxies ──
        # The sweep below runs concurrently; workers don't wait for it.
        # Each worker does a fast single-shot probe before launching a
        # browser, so dead sessions are caught in ~3s not 10s.
        for i, wid in enumerate(WORKER_IDS):
            _log(f"[{wid}] Starting worker...")
            task = asyncio.create_task(_run_worker(wid, cfg, None), name=f"worker-{wid}")
            _worker_tasks[wid] = task

        # A 530-session TLS sweep at the default 250-way concurrency can use
        # most of the 1 GB container while real Chrome is also hydrating
        # React. Run optional pool-wide validation only after browser work
        # is idle; the active worker still probes its chosen session before
        # every use.
        asyncio.create_task(_deferred_proxy_sweep(n_sessions), name="deferred-proxy-sweep")
        asyncio.create_task(_proxy_validate_loop(), name="proxy-validation-summary")

    if not n_sessions:
        # No proxy sessions — start workers directly (TOR fallback)
        for i, wid in enumerate(WORKER_IDS):
            _log(f"[{wid}] Starting worker...")
            task = asyncio.create_task(_run_worker(wid, cfg, None), name=f"worker-{wid}")
            _worker_tasks[wid] = task

async def _stop_all_async() -> None:
    global _running
    _running = False
    _APP_LOGS.clear()
    to_cancel = []

    for wid, state in list(_workers.items()):
        # Signal cooperative cancellation first so browser waits that honour
        # _stopped can finish cleanly.
        bot = state.get("bot")
        if bot is not None:
            try:
                bot._stopped.set()
            except Exception:
                pass

        # A page navigation or network wait may not observe _stopped promptly.
        # Cancelling the retained worker task makes Stop immediate and reliable.
        worker_task = _worker_tasks.get(wid)
        if worker_task is not None and not worker_task.done():
            worker_task.cancel()
            to_cancel.append(worker_task)

        capture_task = state.get("capture_task")
        if capture_task is not None and not capture_task.done():
            capture_task.cancel()
            to_cancel.append(capture_task)
        state["capture_task"] = None

        if state.get("status") in ("starting", "running"):
            state["status"] = "stopped"

    if to_cancel:
        await asyncio.gather(*to_cancel, return_exceptions=True)

    for wid, task in list(_worker_tasks.items()):
        if task.done():
            _worker_tasks.pop(wid, None)
    _log("[App] All workers stopped")

def _run_in_loop(coro) -> Optional[object]:
    if not _loop:
        _log("[Loop] Event loop not running!", level="error")
        return None
    try:
        fut = asyncio.run_coroutine_threadsafe(coro, _loop)
        return fut.result(timeout=120)
    except Exception as e:
        _log(f"[Loop] Error running coroutine: {e}", level="error")
        import traceback
        traceback.print_exc()
        return None

async def _live_navigate_robust(wid: str, bot, url: str) -> dict:
    """Navigate the live tab and self-heal a dead proxy tunnel.

    A parked/launched browser can sit on an expired residential session —
    discord.com then shows 'site can't be reached' (ERR_TUNNEL_CONNECTION_FAILED).
    Probe-gate the next session exactly like the worker loop, then fall back
    to TOR, then to a direct connection, so the LIVE tab never stays stuck on
    a browser error page.
    """
    st = await live_control.live_navigate(bot, url)
    if not st.get("error"):
        return st
    first_err = st.get("error", "")
    _log(f"[{wid}] [Live] Navigate failed ({first_err}) — rotating session and retrying", level="warn")
    # The session the browser is currently on just produced a browser error
    # page: blacklist it so it is never handed out again this run.
    if bot.proxy and proxy_pool is not None:
        try:
            proxy_pool.release(bot.proxy, ok=False)
        except Exception:
            pass
    for _attempt in range(3):
        proxy = await _probe_gated_proxy(wid, bot)
        swapped = False
        via = ""
        if proxy is not None:
            via = "proxy"
            try:
                swapped = await bot.switch_proxy(proxy)
            except Exception:
                swapped = False
            if not swapped and proxy_pool is not None:
                try:
                    proxy_pool.release(proxy, ok=False)
                except Exception:
                    pass
        elif TOR_FALLBACK and _tor_check():
            via = "tor"
            _log(f"[{wid}] [Live] No live proxy sessions — falling back to TOR", level="warn")
            try:
                swapped = await bot.switch_proxy(None)  # fresh TOR circuit
            except Exception:
                swapped = False
        else:
            via = "direct"
            _log(f"[{wid}] [Live] No live proxy and TOR unavailable — using direct connection", level="warn")
            try:
                swapped = await bot.switch_direct()
            except Exception:
                swapped = False
        if not swapped:
            continue
        st = await live_control.live_navigate(bot, url)
        if not st.get("error"):
            _log(f"[{wid}] [Live] Navigation recovered via {via}")
            return st
        if via == "proxy" and proxy is not None and proxy_pool is not None:
            try:
                proxy_pool.release(proxy, ok=False)
            except Exception:
                pass
    st["error"] = f"site unreachable after retries ({first_err})"
    return st

async def _start_live_browser(wid: str, url: str = "",
                              force: bool = False) -> dict:
    """Attach (or cold-launch) the worker's real browser for the LIVE tab.
    The bot shares this same page, so the operator can watch it work or take
    over. Proxy-first, TOR fallback — exactly like the worker. Navigates only
    on a cold launch or when ``force`` is set, so opening the tab never yanks
    a running signup off the page it is filling."""
    state = _workers.get(wid) or _init_worker(wid)
    _workers[wid] = state
    wanted = (url or "").strip()
    # Opening LIVE never yanks a parked demo/signup tab. A cold launch with
    # no URL still lands on Discord register so the camera can go there.
    # The gen is already driving this SAME browser — never relaunch a second
    # one on top of it and never yank it off the page it is filling.
    # Just report what the worker is doing so the LIVE tab shows it live.
    if state.get("status") in ("starting", "running"):
        bot = state.get("bot")
        if bot is not None:
            st = await live_control.get_live_state(bot)
            st["launching"] = state.get("status") == "starting"
            st["status"] = state.get("status", "")
            return st
        # Worker is mid-launch (bot not created yet) — wait for it instead of
        # racing it and leaking a second browser.
        return {"connected": False, "worker_id": wid, "url": url,
                "title": "", "viewport_width": 1920,
                "viewport_height": 1080, "browser": ENGINE,
                "screenshot": "", "error": "", "launching": True,
                "status": state.get("status", "")}
    if state.get("launching"):
        # A launch is already in flight — report it instead of starting a
        # second browser on top of the first (which would leak the first).
        return {"connected": False, "worker_id": wid, "url": url,
                "title": "", "viewport_width": 1920,
                "viewport_height": 1080, "browser": ENGINE,
                "screenshot": "", "error": "", "launching": True}
    state["launching"] = True
    via = "attach"
    launched = False
    try:
        bot = state.get("bot")
        cfg = load_config()
        if bot is None:
            bot = DiscordAutomation(
                headless=bool(cfg.get("headless", True)),
                domain=_pick_domain(cfg),
            )
            state["bot"] = bot
        try:
            alive = await bot.is_alive()
        except Exception:
            alive = False
        if not alive:
            # Probe-gate the first session: launching straight onto an expired
            # residential tunnel is what left the LIVE tab on a browser error
            # page.
            proxy = await _probe_gated_proxy(wid, bot)
            if proxy is not None:
                bot.proxy = proxy
                bot._direct = False
                via = "proxy"
            elif TOR_FALLBACK and _tor_check():
                bot.proxy = None
                bot._direct = False
                via = "tor"
            else:
                bot._direct = True
                bot.proxy = None
                via = "direct"
            _log(f"[{wid}] [Live] Launching browser ({via}, engine={ENGINE})…")
            await asyncio.wait_for(bot.initialize(), timeout=90)
            _log(f"[{wid}] [Live] Browser launched ({via})")
            launched = True
        if launched and not wanted:
            wanted = live_control.DISCORD_REGISTER_URL
        if wanted:
            # Navigate when the operator asked for a URL (REGISTER) or this
            # is a cold launch. A parked demo tab stays put unless force=1.
            cur = ""
            try:
                cur = str(bot._page.url or "") if bot._page else ""
            except Exception:
                cur = ""
            busy = state.get("status") in ("starting", "running", "demo")
            should_go = force or launched or (
                cur.rstrip("/") != wanted.rstrip("/") and not busy)
            if should_go:
                st = await _live_navigate_robust(wid, bot, wanted)
                if state.get("status") not in ("starting", "running"):
                    state["status"] = "live"
                    state["step"] = "live camera · discord register"
                if st.get("screenshot"):
                    state["last_shot_b64"] = st["screenshot"]
                st["status"] = state.get("status", "")
                return st
        if state.get("status") in ("idle", "stopped", "error", ""):
            state["status"] = "live"
            state["step"] = "live camera"
        st = await live_control.get_live_state(bot)
        st["status"] = state.get("status", "")
        return st
    except Exception as e:
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _log(f"[{wid}] [Live] Live browser failed ({via}): {e}\n{tb}", level="error")
        return {"connected": False, "worker_id": wid, "url": "",
                "title": "", "viewport_width": 1920,
                "viewport_height": 1080, "browser": ENGINE,
                "screenshot": "", "error": f"browser launch failed: {e}"}
    finally:
        state["launching"] = False

async def _close_live_browser(wid: str) -> bool:
    state = _workers.get(wid)
    bot = state.get("bot") if state else None
    if bot is None:
        return False
    try:
        bot._stopped.set()
    except Exception:
        pass
    try:
        await bot.close()
    except Exception:
        pass
    state["bot"] = None
    state["last_shot_b64"] = ""
    return True


async def _start_real_demo_browser(wid: str, cfg: dict) -> dict:
    """Prepare B1's shared LIVE browser for the official hCaptcha demo.

    The demo runner intentionally uses a direct connection.  It does not
    rotate proxies, use TOR, or borrow the Discord signup worker's transport;
    this is a small, transparent QA flow against hCaptcha's public demo.
    """
    state = _workers.get(wid) or _init_worker(wid)
    _workers[wid] = state
    if state.get("status") in ("starting", "running"):
        return {"error": "worker is busy with another browser task"}
    if state.get("launching"):
        return {"error": "browser is still launching"}

    state["launching"] = True
    try:
        bot = state.get("bot")
        if bot is None:
            bot = DiscordAutomation(
                headless=bool(cfg.get("headless", True)),
                worker_id=wid,
                domain=DEFAULT_MAIL_DOMAIN,
            )
            state["bot"] = bot
        try:
            alive = await bot.is_alive()
        except Exception:
            alive = False
        if not alive:
            bot.proxy = None
            bot._direct = True
            _log(f"[{wid}] [Demo] Launching direct real Chrome for {trainer.TARGET_DEMO_URL}")
            await asyncio.wait_for(bot.initialize(), timeout=90)
        elif not getattr(bot, "_direct", False) or bot.proxy:
            # A parked signup browser may have a proxy attached.  Rebuild it
            # directly before using it for the public demo.
            if not await bot.switch_direct():
                return {"error": "could not switch the shared browser to direct mode"}
        st = await live_control.live_navigate(bot, trainer.TARGET_DEMO_URL)
        if st.get("error"):
            return st
        state["status"] = "demo"
        state["step"] = "official hCaptcha demo"
        state["proxy"] = "direct"
        if st.get("screenshot"):
            state["last_shot_b64"] = st["screenshot"]
        return st
    except Exception as exc:
        _log(f"[{wid}] [Demo] Browser setup failed: {type(exc).__name__}: {exc}", level="error")
        return {"error": f"real demo browser failed: {exc}"}
    finally:
        state["launching"] = False


async def _start_real_demo_runner(speed: float, one_shot: bool = False) -> dict:
    """Attach the trainer to B1's browser on the app asyncio loop."""
    if trainer.trainer_engine.is_busy():
        return {"ok": False, "message": "Real demo runner already running or stopping"}
    cfg = load_config()
    browser = await _start_real_demo_browser("B1", cfg)
    if browser.get("error"):
        return {"ok": False, "message": browser["error"], "browser": browser}
    state = _workers.get("B1") or _init_worker("B1")
    bot = state.get("bot")
    result = trainer.trainer_engine.start_external(
        getattr(bot, "_page", None), speed=speed, one_shot=one_shot,
    )
    if result.get("ok"):
        state["status"] = "demo"
        state["step"] = "official hCaptcha demo"
    return {**result, "browser": browser}

# ── Flask app ─────────────────────────────────────────────

app = Flask(__name__)

@app.route('/')
def handle_root():
    html = DASHBOARD_HTML
    idx = html.rfind("</body>")
    if idx != -1:
        html = html[:idx] + live_ui.LIVE_INJECTION + html[idx:]
    return Response(html, content_type='text/html')

@app.route('/start', methods=['POST'])
def handle_start():
    global _workers
    if _running:
        return jsonify({"ok": False, "msg": "Already running"})
    try:
        cfg = load_config()
        threading.Thread(
            target=lambda: _run_in_loop(_start_all_async(cfg)),
            daemon=True,
        ).start()
        return jsonify({"ok": True, "msg": "Started — 1 browser launching"})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Start error: {e}"})

@app.route('/stop', methods=['POST'])
def handle_stop():
    _run_in_loop(_stop_all_async())
    return jsonify({"ok": True, "message": "Stopped"})

@app.route('/proxies/refresh', methods=['POST'])
def handle_proxy_refresh():
    if _proxies_available and proxy_pool is not None:
        _run_in_loop(proxy_pool.refresh())
        return jsonify(proxy_pool.stats())
    return jsonify({"error": "proxies module not loaded"})

def _mask_proxy_key(key: str) -> str:
    """Display-safe form of a proxy key (user:pass@host:port) — never leak
    the credentials: show the short session id + host only."""
    host = key.rsplit("@", 1)[-1] if "@" in key else key
    sid = ""
    m = re.search(r"-s-([A-Za-z0-9]+)", key)
    if m:
        sid = "s-" + m.group(1)[:10]
    return f"{sid} @ {host}" if sid else host

@app.route('/proxies')
def handle_proxies():
    """Proxy dashboard data: valid / used / invalid / unproven sessions.

    Merges the live pool state with the persistent used_proxies store so the
    tab still shows history (and the pool skips already-used sticky IPs)
    after a redeploy."""
    rows = []
    db_map = {r.get("key"): r for r in rows}

    def _mk(key, rec, live_flag=False):
        return {
            "label": _mask_proxy_key(key),
            "ip": (live_flag and rec.get("ip")) or rec.get("exit_ip") or "",
            "used": True,
            "invalid": rec.get("status") == "invalid",
            "valid": rec.get("status") == "valid",
        }

    valid, invalid, used = [], [], []
    live = set()
    if proxy_pool is not None:
        for p in proxy_pool._proxies:
            key = p.get("key", "")
            live.add(key)
            rec = db_map.get(key) or {}
            status = "invalid" if key in proxy_pool._failed else (
                "valid" if p.get("_valid") else rec.get("status") or "unproven")
            entry = {
                "label": _mask_proxy_key(key),
                "ip": p.get("_resolved_ip") or rec.get("exit_ip") or "",
                "used": (key in proxy_pool._used_at
                          or key in proxy_pool._used_before),
                "invalid": status == "invalid",
                "valid": status == "valid",
            }
            if entry["invalid"]:
                invalid.append(entry)
            elif entry["valid"]:
                valid.append(entry)
            if entry["used"]:
                used.append(entry)

    # Persistent history that survives redeploys: DB rows not in the live pool.
    for key, rec in db_map.items():
        if key in live:
            continue
        entry = {
            "label": _mask_proxy_key(key),
            "ip": rec.get("exit_ip") or "",
            "used": True,
            "invalid": rec.get("status") == "invalid",
            "valid": rec.get("status") == "valid",
        }
        if entry["invalid"]:
            invalid.append(entry)
        else:
            used.append(entry)

    return jsonify({
        "total": len(proxy_pool._proxies) if proxy_pool is not None else len(db_map),
        "valid": valid,
        "invalid": invalid,
        "used": used,
        "db": bool(db_map),
        "stats": proxy_pool.stats() if proxy_pool is not None else {},
    })

@app.route('/status')
def handle_status():
    workers = []
    for wid in WORKER_IDS:
        s = _workers.get(wid) or _init_worker(wid)
        workers.append({
            "id": wid,
            "status": s["status"],
            "step": s["step"],
            "email": s["email"],
            "username": s["username"],
            "token": s["token"],
            "proxy": s["proxy"],
            "screenshots": s["screenshots"],
            "started_at": s["started_at"],
        })
    try:
        cfg_now = load_config()
    except Exception:
        cfg_now = dict(DEFAULT_CONFIG)
    _mail_domains = cfg_now.get("mail_domains", []) or []
    return jsonify({
        "running": _running,
        "uptime": int(time.time() - _start_time) if _start_time else 0,
        "workers": workers,
        "headless": bool(cfg_now.get("headless", True)),
        "workerCount": int(cfg_now.get("worker_count", WORKER_COUNT)),
        "mail_domains": _mail_domains,
        "custom_email": cfg_now.get("custom_email", ""),
        "proxies": proxy_pool.stats() if (_proxies_available and proxy_pool is not None) else {},
    })

@app.route('/latest')
def handle_latest_screenshot():
    wid = request.args.get("worker", "B1")
    s = _workers.get(wid)
    if s and s.get("last_shot_b64"):
        try:
            return Response(base64.b64decode(s["last_shot_b64"]),
                            content_type='image/png')
        except Exception:
            pass
    # Fallback: try the bot's own screenshot store
    if s:
        bot = s.get("bot")
        if bot is not None:
            try:
                shot = bot.get_latest_screenshot()
                if shot:
                    return Response(base64.b64decode(shot),
                                    content_type='image/png')
            except Exception:
                pass
    return Response(status=404)

# ── LIVE CONTROL routes ──────────────────────────────────

@app.route('/browser/state')
def handle_browser_state():
    wid = request.args.get("worker", "B1")
    s = _workers.get(wid)
    bot = s.get("bot") if s else None
    if bot is None:
        return jsonify({"connected": False, "worker_id": wid, "url": "",
                        "title": "", "viewport_width": 1920,
                        "viewport_height": 1080, "browser": ENGINE,
                        "screenshot": "", "error": "browser not started"})
    st = _run_in_loop(live_control.get_live_state(bot))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid, "url": "",
                        "title": "", "viewport_width": 1920,
                        "viewport_height": 1080, "browser": ENGINE,
                        "screenshot": "", "error": "event loop unavailable"}), 503
    if st.get("screenshot"):
        s["last_shot_b64"] = st["screenshot"]
    elif s.get("last_shot_b64"):
        st["screenshot"] = s["last_shot_b64"]
    # Surface the gen's status so the LIVE tab shows "launching browser…"
    # during START instead of a misleading "browser not started".
    st["launching"] = bool(s.get("launching") or s.get("status") == "starting")
    st["status"] = s.get("status", "")
    return jsonify(st)

@app.route('/browser/navigate', methods=['POST'])
def handle_browser_navigate():
    wid = request.args.get("worker", "B1")
    data = request.get_json(silent=True) or {}
    url = str(data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "empty url"}), 400
    s = _workers.get(wid)
    bot = s.get("bot") if s else None
    if bot is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "browser not started"}), 409
    st = _run_in_loop(_live_navigate_robust(wid, bot, url))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "event loop unavailable"}), 503
    if st.get("screenshot"):
        s["last_shot_b64"] = st["screenshot"]
    return jsonify(st)

@app.route('/browser/action', methods=['POST'])
def handle_browser_action():
    wid = request.args.get("worker", "B1")
    data = request.get_json(silent=True) or {}
    s = _workers.get(wid)
    bot = s.get("bot") if s else None
    if bot is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "browser not started"}), 409
    st = _run_in_loop(live_control.live_action(bot, data))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "event loop unavailable"}), 503
    if st.get("screenshot"):
        s["last_shot_b64"] = st["screenshot"]
    return jsonify(st)

@app.route('/browser/start', methods=['POST'])
def handle_browser_start():
    wid = request.args.get("worker", "B1")
    data = request.get_json(silent=True) or {}
    url = str(data.get("url") or "").strip()
    force = bool(data.get("force"))
    st = _run_in_loop(_start_live_browser(wid, url, force=force))
    if st is None:
        return jsonify({"connected": False, "worker_id": wid,
                        "error": "event loop unavailable"}), 503
    s = _workers.get(wid)
    if s is not None and st.get("screenshot"):
        s["last_shot_b64"] = st["screenshot"]
    return jsonify(st)

@app.route('/browser/close', methods=['POST'])
def handle_browser_close():
    wid = request.args.get("worker", "B1")
    closed = _run_in_loop(_close_live_browser(wid))
    return jsonify({"closed": bool(closed)})

@app.route('/worker/<wid>/logs')
def handle_worker_logs(wid):
    s = _workers.get(wid)
    if not s:
        return jsonify({"logs": list(_APP_LOGS[-200:]), "status": _running and "starting" or "idle"})
    bot = s.get("bot")
    bot_logs = bot.get_activity_log() if bot else []
    # Merge app-level lines ([Proxy] stats, [B1] Done/Failed, errors) with the
    # bot's internal activity log so the terminal shows everything.
    merged = list(bot_logs)
    seen = {(e.get("time"), e.get("message")) for e in bot_logs}
    for e in _APP_LOGS:
        k = (e.get("time"), e.get("message"))
        if k not in seen:
            seen.add(k)
            merged.append(dict(e))
    merged.sort(key=lambda e: e.get("timestamp", 0))
    return jsonify({
        "id": wid,
        "status": s["status"],
        "email": s.get("email", ""),
        "username": s.get("username", ""),
        "proxy": s.get("proxy", ""),
        "screenshots": s.get("screenshots", 0),
        "started_at": s.get("started_at", 0),
        "logs": merged,  # store caps (500 bot / 400 app) bound the size
    })

@app.route('/tokens')
def handle_tokens():
    if not False or db is None:
        return jsonify({"count": 0, "valid": 0, "expired": 0, "pending": 0,
                        "accounts": [], "stats": {"total": 0, "valid": 0,
                        "expired": 0, "pending": 0}, "error": "DB not available"})
    expired = sum(1 for a in accounts if a.get("status") == "invalid")
    valid = sum(1 for a in accounts if a.get("status") == "valid")
    pending = len(accounts) - expired - valid
    return jsonify({
        "count": len(accounts),
        "valid": valid,
        "expired": expired,
        "pending": pending,
        "accounts": accounts,
        "stats": {"total": len(accounts), "valid": valid,
                   "expired": expired, "pending": pending},
    })

@app.route('/validate', methods=['POST'])
def handle_validate():
    if not False or db is None:
        return jsonify({"error": "DB not available"})
    # Cap at 200 so the synchronous validate stays inside the 120s loop budget
    expired = sum(1 for a in accounts if a.get("status") == "invalid")
    return jsonify({"count": len(accounts), "valid": valid, "expired": expired,
                    "accounts": accounts})

@app.route('/export', methods=['POST'])
def handle_export():
    """Preview the next N accounts for export (does NOT delete)."""
    if not False or db is None:
        return jsonify({"error": "DB not available"})
    data = request.get_json(silent=True) or {}
    try:
        count = max(1, min(int(data.get('count', 5)), 100))
    except Exception:
        count = 5
    mode = 'full' if data.get('mode') == 'full' else 'tokens'
    chosen = [a for a in accounts if a.get('token')][:count]
    out = []
    for a in chosen:
        if mode == 'full':
            text = "\n".join([
                a.get('token') or '',
                "Email: " + str(a.get('email') or ''),
                "Password: " + str(a.get('password') or ''),
                "Username: " + str(a.get('username') or ''),
            ])
        else:
            text = a.get('token') or ''
        out.append({
            "id": a.get("id"),
            "text": text,
            "token": a.get('token'),
            "email": a.get('email'),
            "username": a.get('username'),
        })
    return jsonify({"count": len(out), "accounts": out})

@app.route('/export/delete', methods=['POST'])
def handle_export_delete():
    """Delete exported accounts after the user confirms the copy."""
    if not False or db is None:
        return jsonify({"error": "DB not available"})
    data = request.get_json(silent=True) or {}
    ids = []
    for i in (data.get('ids') or []):
        try:
            ids.append(int(i))
        except Exception:
            pass
    if not ids:
        return jsonify({"ok": False, "msg": "no ids"})
    return jsonify({"ok": True, "deleted": deleted})

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        cfg = load_config()
        if 'headless' in data:
            cfg['headless'] = bool(data['headless'])
        if 'mail_domains' in data:
            domains = [str(d).strip().lower() for d in data['mail_domains'] if str(d).strip()]
            cfg['mail_domains'] = domains or ["glasswhitehub.com"]
        if 'custom_email' in data:
            cfg['custom_email'] = str(data.get('custom_email') or '').strip().lower()
        save_config(cfg)
        return jsonify({"ok": True, "config": cfg})
    cfg = load_config()
    avail = [d for d in cfg.get("mail_domains", [DEFAULT_MAIL_DOMAIN])
             if d not in _BURNED_DOMAINS]
    return jsonify({"headless": cfg.get("headless", True),
                    "worker_count": cfg.get("worker_count", WORKER_COUNT),
                    "mail_domains": cfg.get("mail_domains", ["glasswhitehub.com"]),
                    "custom_email": cfg.get("custom_email", ""),
                    "burned_domains": sorted(_BURNED_DOMAINS),
                    "available_domains": avail})

# ── Background event loop ─────────────────────────────────


# ── Trainer Endpoints ─────────────────────────────────────

@app.route('/trainer/status')
def handle_trainer_status():
    return jsonify(trainer.trainer_engine.get_state())

@app.route('/trainer/start', methods=['POST'])
def handle_trainer_start():
    data = request.get_json(silent=True) or {}
    try:
        speed = float(data.get('speed', 2.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "speed must be a number"}), 400
    result = _run_in_loop(_start_real_demo_runner(speed, one_shot=False))
    if result is None:
        return jsonify({"ok": False, "message": "event loop unavailable"}), 503
    return jsonify(result)

@app.route('/trainer/stop', methods=['POST'])
def handle_trainer_stop():
    return jsonify(trainer.trainer_engine.stop())

@app.route('/trainer/step', methods=['POST'])
def handle_trainer_step():
    result = _run_in_loop(_start_real_demo_runner(
        trainer.trainer_engine.speed, one_shot=True,
    ))
    if result is None:
        return jsonify({"ok": False, "message": "event loop unavailable"}), 503
    return jsonify(result)

@app.route('/trainer/clear', methods=['POST'])
def handle_trainer_clear():
    return jsonify(trainer.trainer_engine.clear())

@app.route('/trainer/questions')
def handle_trainer_questions():
    st = trainer.trainer_engine.get_state()
    return jsonify(st.get('questions', []))

@app.route('/trainer/speed', methods=['POST'])
def handle_trainer_speed():
    data = request.get_json(silent=True) or {}
    try:
        speed = float(data.get('speed', 2.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "speed must be a number"}), 400
    return jsonify(trainer.trainer_engine.set_speed(speed))


@app.route('/challenges/<name>')
def handle_challenge_file(name):
    """Serve a saved challenge / companion browser PNG. Never deleted."""
    path = live_control.challenge_file_path(name)
    if not path:
        return Response(status=404)
    return send_file(path, mimetype="image/png")


def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def _vision_keepalive(interval: float = 240.0) -> None:
    """Keep a HOSTED vision endpoint warm so the platform doesn't stop it.

    The bot only calls vision when a captcha appears. Between captchas the
    service idles — and on Railway's Hobby plan a deployment with no traffic
    is STOPPED after ~15 minutes. The next captcha then hits a dead edge
    (TLS reset in <0.1s, not a cold start) and every round fails until
    someone manually restarts the deploy. A tiny GET / every 4 minutes
    counts as traffic and keeps the deployment running. Local Ollama
    (localhost) is skipped — it never idles out.
    """
    import vision_solver as _vs
    base = (_vs.OLLAMA_BASE or "").rstrip("/")
    if not base:
        return
    host = base.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]",
                "host.docker.internal"):
        return
    headers = {}
    if _vs.VISION_API_KEY:
        headers["Authorization"] = f"Bearer {_vs.VISION_API_KEY}"
    was_up: Optional[bool] = None
    while True:
        try:
            timeout = aiohttp.ClientTimeout(total=25)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(base + "/", headers=headers) as r:
                    up = (r.status == 200)
        except Exception:
            up = False
        if was_up is not None and up != was_up:
            if up:
                _log(f"[Vision] Endpoint back UP at {base}", level="warn")
            else:
                _log(f"[Vision] Endpoint DOWN at {base} — captcha rounds "
                     "will fail until it answers (check the hosted deploy)",
                     level="warn")
        was_up = up
        await asyncio.sleep(interval)

def main() -> None:
    global _loop
    _load_burned()
    config = load_config()
    web_port = config.get("web_port", 8080)

    if not os.path.exists(_config_path):
        save_config(config)

    _loop = asyncio.new_event_loop()
    t = threading.Thread(target=_run_event_loop, args=(_loop,), daemon=True)
    t.start()

    # Keep the hosted vision endpoint from being stopped for inactivity
    # (no-op when VISION_API_BASE is unset or points at localhost).
    try:
        asyncio.run_coroutine_threadsafe(_vision_keepalive(), _loop)
    except Exception as e:
        print(f"[app] vision keepalive not started: {e}", flush=True)

    # Auto-migrate DB (DATABASE_URL from env)

    print("=" * 56, flush=True)
    print("  EYES GEN - multi-browser Discord token generator", flush=True)
    print(f"  Browsers per Start: {WORKER_COUNT}", flush=True)
    print(f"  Dashboard: http://0.0.0.0:{web_port}", flush=True)
    print("=" * 56, flush=True)

    app.run(host='0.0.0.0', port=web_port, debug=False,
            use_reloader=False, threaded=True)

# ═══════════════════════════════════════════════════════════
# EYES GEN DASHBOARD — mobile-first
# ═══════════════════════════════════════════════════════════

DASHBOARD_HTML = r'''
<!doctype html><html lang="en"><head>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0a0a0b">
<meta charset="utf-8">
<title>EY3 - Token Forge & Trainer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html{background:#0a0a0b}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:radial-gradient(900px 400px at 85% -10%,rgba(255,255,255,.045),transparent 60%),
    radial-gradient(700px 380px at -10% 0%,rgba(255,255,255,.03),transparent 55%),#0a0a0b;
  color:#e7e7ea;min-height:100vh;max-width:980px;margin:0 auto;padding:16px 16px 90px}
h1{font-family:'JetBrains Mono','Courier New',monospace;font-size:28px;font-weight:700;
  letter-spacing:2px;display:flex;align-items:center;gap:10px}
.sub{color:#8a8a92;font-size:12px;margin:6px 0 16px;display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:#5c5c64;display:inline-block}
.dot.on{background:#34d399;box-shadow:0 0 12px #34d399}
nav{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap;border-bottom:1px solid #26262b;padding-bottom:12px}
nav button{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;
  padding:9px 18px;border-radius:10px;border:1px solid #26262b;background:#131316;
  color:#e7e7ea;cursor:pointer;transition:.15s ease}
nav button:hover{background:#1a1a1e;border-color:#34343a}
nav button.act{background:#34d399;color:#0a0a0b;border-color:#34d399}
.card{background:#131316;border:1px solid #26262b;border-radius:14px;padding:18px 20px;margin-bottom:12px}
.card h3{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:500;
  letter-spacing:1px;margin-bottom:10px;color:#8a8a92;text-transform:uppercase}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.col{flex:1;min-width:110px}
.stat{background:#1a1a1e;border:1px solid #26262b;border-radius:10px;padding:14px;text-align:center}
.stat .n{font-size:28px;font-weight:700;font-family:'JetBrains Mono','Courier New',monospace}
.stat .l{font-size:10px;color:#8a8a92;letter-spacing:1px;margin-top:4px;text-transform:uppercase}
button{padding:9px 22px;border-radius:10px;border:1px solid #26262b;background:#131316;
  color:#e7e7ea;cursor:pointer;font-size:13px;font-weight:600}
button:hover{background:#1a1a1e;border-color:#34343a}
button.primary{background:#34d399;color:#0a0a0b;border-color:#34d399}
button.danger{background:#f87171;color:#0a0a0b;border-color:#f87171}
input,select{padding:9px 12px;border-radius:10px;border:1px solid #26262b;background:#1a1a1e;
  color:#e7e7ea;width:100%;font-size:13px}
label{font-size:11px;color:#8a8a92;display:block;margin-bottom:4px;letter-spacing:1px;
  text-transform:uppercase}
.log-box,.all-logs-box{background:#060608;border:1px solid #26262b;border-radius:10px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;font-weight:400;
  line-height:1.55;padding:12px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;color:#fff}
.log-box{max-height:340px}
#logOverlay{display:none;position:fixed;inset:0;z-index:300;padding:18px;background:rgba(0,0,0,.72);backdrop-filter:blur(4px);align-items:center;justify-content:center}
#logOverlay.on{display:flex}
.logs-modal{width:min(980px,100%);height:min(720px,calc(100vh - 36px));display:flex;flex-direction:column;gap:12px;padding:16px;background:#131316;border:1px solid #34343a;border-radius:16px;box-shadow:0 26px 80px rgba(0,0,0,.62)}
.logs-head{display:flex;align-items:center;gap:12px}
.logs-head h2{font-size:15px;font-weight:700;color:#fff;letter-spacing:.2px}
.logs-close{margin-left:auto;padding:7px 12px;border-radius:9px;border:1px solid #5a2323;background:#2a1212;color:#fff;cursor:pointer;font-size:13px;font-weight:700;line-height:1}
.all-logs-box{flex:1;min-height:0;max-height:none}
@media(max-width:640px){#logOverlay{padding:10px}.logs-modal{height:calc(100vh - 20px);padding:12px;border-radius:13px}.log-box,.all-logs-box{font-size:13px}}
.badge{display:inline-block;font-size:10px;padding:3px 10px;border-radius:99px;
  font-family:'JetBrains Mono','Courier New',monospace;letter-spacing:1px}
.badge-ok{background:rgba(52,211,153,.15);color:#34d399}
.badge-warn{background:rgba(251,191,36,.15);color:#fbbf24}
.badge-err{background:rgba(248,113,113,.15);color:#f87171}
.flex{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.hide{display:none!important}
.mt{margin-top:12px}.mb{margin-bottom:12px}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1a1a1e;
  border:1px solid #34343a;border-radius:10px;padding:12px 22px;font-size:13px;z-index:999;box-shadow:0 12px 30px rgba(0,0,0,.6)}

/* Live official-demo dashboard layout */
.trainer-grid{display:grid;grid-template-columns:1.05fr 1fr;gap:14px;align-items:start}
@media(max-width:820px){.trainer-grid{grid-template-columns:1fr}}

.trainer-ss-box{background:#0b0c10;border:1px solid #272a3a;border-radius:12px;padding:12px;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:340px;position:relative;overflow:hidden}
#trainerModalImg{max-width:100%;height:auto;max-height:480px;border-radius:8px;border:1px solid #33374b;box-shadow:0 12px 36px rgba(0,0,0,.6);object-fit:contain}
.trainer-ph{color:#8a8a92;font-family:'JetBrains Mono',monospace;font-size:12px;text-align:center;padding:34px 16px;line-height:1.6}

.q-list-wrap{max-height:480px;overflow-y:auto;background:#08080a;border:1px solid #26262b;border-radius:10px;padding:6px}
.q-row{display:flex;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid #1a1a1f;transition:.15s ease}
.q-row:last-child{border-bottom:none}
.q-row:hover{background:#131318}
.q-num{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:13px;color:#34d399;min-width:24px}
.q-body{flex:1;min-width:0}
.q-title{font-size:13px;font-weight:600;color:#fff;margin-bottom:3px;word-break:break-word}
.q-meta{font-size:10px;color:#8a8a92;font-family:'JetBrains Mono',monospace;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.btn-copy-q{padding:5px 11px;font-size:11px;font-weight:700;background:#1a1a22;border:1px solid #323547;color:#34d399;border-radius:7px;cursor:pointer;white-space:nowrap;transition:.15s}
.btn-copy-q:hover{background:#232738;border-color:#34d399}
.btn-copy-q.copied{background:#059669;color:#fff;border-color:#34d399}

.badge-tag{font-size:9px;padding:2px 6px;border-radius:4px;background:#232738;color:#93c5fd;font-weight:600;letter-spacing:.5px}

.stage-banner{background:#16171d;border:1px solid #2a2d3d;border-radius:10px;padding:11px 16px;font-size:12px;font-family:'JetBrains Mono',monospace;color:#a1a1aa;margin-bottom:14px;display:flex;align-items:center;gap:10px}
.stage-dot{width:9px;height:9px;border-radius:50%;background:#52525b;display:inline-block;flex-shrink:0}
.stage-dot.active{background:#34d399;box-shadow:0 0 10px #34d399;animation:pulseDot 1.2s infinite}
@keyframes pulseDot{0%{opacity:.4}50%{opacity:1}100%{opacity:.4}}

/* Real official-demo status card */
.demo-form-stage{background:#101116;border:1px solid #262835;border-radius:10px;padding:14px}
.demo-input-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
@media(max-width:500px){.demo-input-row{grid-template-columns:1fr}}
.demo-input-box{background:#181920;border:1px solid #2a2c3a;border-radius:8px;padding:7px 10px;font-size:12px;color:#d4d4d8;font-family:'JetBrains Mono',monospace}
.demo-link{color:#34d399;text-decoration:none;word-break:break-all}
.demo-note{color:#94a3b8;font-size:11px;line-height:1.55;margin-top:10px}
</style></head><body>

<h1>EY3 <span style="font-size:11px;color:#8a8a92;border:1px solid #26262b;border-radius:99px;padding:4px 10px;font-weight:500;letter-spacing:2px">TOKEN FORGE</span></h1>

<div class="sub"><span class="dot" id="statusDot"></span> <span id="statusText">loading...</span></div>

<nav id="tabNav">
<button class="act" data-tab="main" onclick="showTab('main')">Dashboard</button>
<button data-tab="tokens" onclick="showTab('tokens')">Tokens</button>
<button data-tab="trainer" onclick="showTab('trainer')">Live Demo</button>
</nav>

<div id="tabmain">
<div class="row mb">
<div class="col stat"><div class="n" id="statGenerated">0</div><div class="l">Generated</div></div>
<div class="col stat"><div class="n" id="statValid">0</div><div class="l">Valid</div></div>
<div class="col stat"><div class="n" id="statExpired">0</div><div class="l">Expired</div></div>
</div>

<div class="card">
<div class="flex mb">
<h3 style="margin:0">Proxies</h3>
<span id="proxyStats" class="badge badge-ok">--</span>
</div>
<div class="row">
<div class="col stat"><div class="n" id="proxyTotal">0</div><div class="l">Total</div></div>
<div class="col stat"><div class="n" id="proxyValid">0</div><div class="l">Valid</div></div>
<div class="col stat"><div class="n" id="proxyUsed">0</div><div class="l">Used</div></div>
<div class="col stat"><div class="n" id="proxyInvalid">0</div><div class="l">Invalid</div></div>
</div>
</div>

<div class="card">
<h3>Control</h3>
<div class="flex">
<button class="primary" id="btnStart" onclick="startBot()">START</button>
<button class="danger" id="btnStop" onclick="stopBot()">STOP</button>
<button id="btnLive" onclick="openLive()">LIVE</button>
<button onclick="refreshProxies()">Refresh Proxies</button>
</div>
</div>

<div class="card">
<h3>Logs</h3>
<div class="log-box" id="logBox">Waiting for logs...</div>
<div class="flex mt">
<button onclick="viewAllLogs()">ALL LOGS</button>
<button onclick="copyLogs()">COPY LOGS</button>
</div>
</div>
</div>

<div id="tabtokens" class="hide">
<div class="card">
<h3>Tokens</h3>
<div class="flex mb"><button onclick="refreshTokens()">Refresh</button></div>
<div id="tokenList"><p style="color:#8a8a92">No tokens yet.</p></div>
</div>
</div>

<!-- ═══════════════════════════════════════════════════════════
     TRAINER TAB: REAL OFFICIAL DEMO (HUMAN-IN-THE-LOOP)
     ═══════════════════════════════════════════════════════════ -->
<div id="tabtrainer" class="hide">
  <div class="row mb">
    <div class="col stat"><div class="n" id="statCaptured">0</div><div class="l">Captured Challenges</div></div>
    <div class="col stat"><div class="n" id="statBypassed">0</div><div class="l">Completed Checks</div></div>
    <div class="col stat"><div class="n" id="statCycles">0</div><div class="l">Demo Cycles</div></div>
    <div class="col stat"><div class="n" id="statSpeed">2.0s</div><div class="l">Poll Delay</div></div>
  </div>

  <div class="card">
    <div class="flex" style="justify-content:space-between;margin-bottom:12px">
      <h3 style="margin:0">Real hCaptcha Demo Controls</h3>
      <span id="trainerStatusBadge" class="badge badge-warn">IDLE</span>
    </div>
    <div class="flex">
      <button class="primary" id="btnTrainerStart" onclick="startTrainer()">▶ START REAL DEMO</button>
      <button class="danger" id="btnTrainerStop" onclick="stopTrainer()" disabled>⏸ STOP</button>
      <button id="btnTrainerStep" onclick="stepTrainer()">⏭ RUN ONCE</button>
      <button onclick="openLive()">👁 OPEN LIVE BROWSER</button>
      <select id="trainerSpeedSelect" style="width:auto;min-width:130px" onchange="updateTrainerSpeed(this.value)">
        <option value="1.0">Fast (1.0s)</option>
        <option value="2.0" selected>Normal (2.0s)</option>
        <option value="3.5">Relaxed (3.5s)</option>
      </select>
      <button onclick="clearTrainerQuestions()">🗑 Clear</button>
    </div>
    <div class="demo-note">
      The runner uses a real Chrome tab and the official demo below. It fills the optional field,
      clicks the real checkbox, captures a real challenge, and waits for you to complete it.
      Open LIVE BROWSER to interact with that same tab. It never fabricates a challenge,
      creates a token, or selects challenge answers.
    </div>
  </div>

  <div class="stage-banner">
    <span class="stage-dot" id="trainerStageDot"></span>
    <span id="trainerStageText">Ready to open the official hCaptcha demo.</span>
  </div>

  <div class="trainer-grid">
    <div>
      <div class="card" style="margin-bottom:12px">
        <div class="flex" style="justify-content:space-between;margin-bottom:10px">
          <h3 style="margin:0">Latest Real Challenge</h3>
          <button class="btn-copy-q" id="btnCopyLatest" onclick="copyLatestQuestion()" style="display:none">📋 Copy Question</button>
        </div>
        <div class="trainer-ss-box">
          <div id="trainerPlaceholder" class="trainer-ph">
            Start the real demo runner.<br>A live Chrome screenshot refreshes here every 3 seconds.
          </div>
          <img id="trainerModalImg" style="display:none" alt="Screenshot of the real hCaptcha challenge">
        </div>
        <div id="trainerPointerLog" class="demo-note" style="margin-top:10px;min-height:18px;font-family:monospace"></div>
      </div>

      <div class="card">
        <h3>Official hCaptcha Demo</h3>
        <div class="demo-form-stage">
          <div style="font-size:11px;color:#8a8a92;margin-bottom:8px;font-family:monospace">TARGET</div>
          <a class="demo-link" href="https://accounts.hcaptcha.com/demo" target="_blank" rel="noopener">https://accounts.hcaptcha.com/demo</a>
          <div class="demo-input-row" style="margin-top:12px">
            <div>
              <label>Generated sample</label>
              <input class="demo-input-box" id="demoFormComment" readonly placeholder="Filled in the real demo">
            </div>
            <div>
              <label>Current runner value</label>
              <input class="demo-input-box" id="demoFormName" readonly placeholder="Waiting for runner">
            </div>
          </div>
          <label>Browser form field</label>
          <input class="demo-input-box" id="demoFormEmail" readonly placeholder="The official page has one optional field">
          <div class="demo-note">Checkbox interaction and challenge rendering happen in Chrome, not in this dashboard.</div>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="flex" style="justify-content:space-between;margin-bottom:10px">
          <h3 style="margin:0">Captured Questions <span id="qCountBadge" class="badge badge-ok" style="margin-left:6px">0</span></h3>
          <div class="flex" style="gap:6px">
            <button class="btn-copy-q" onclick="copyAllQuestions()">📋 Copy All</button>
            <button class="btn-copy-q" onclick="exportQuestionsJson()">⬇️ JSON</button>
          </div>
        </div>
        <div class="q-list-wrap" id="trainerQuestionsList">
          <div style="color:#8a8a92;font-size:12px;padding:28px 16px;text-align:center;font-family:monospace">
            No real challenge captured yet.<br>Start the runner to open the official demo.
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="logOverlay" role="dialog" aria-modal="true" aria-labelledby="allLogsTitle">
  <div class="logs-modal">
    <div class="logs-head">
      <h2 id="allLogsTitle">ALL LOGS</h2>
      <button class="logs-close" type="button" onclick="closeAllLogs()" aria-label="Close all logs">X</button>
    </div>
    <div id="allLogsBox" class="all-logs-box">Loading logs...</div>
  </div>
</div>

<script>
function $(id){return document.getElementById(id)}
function api(path,opts){
  opts=opts||{};
  var f=Object.assign({method:'POST',headers:{'Content-Type':'application/json'}},opts);
  if(opts.body)f.body=JSON.stringify(opts.body);
  return fetch(path,f);
}
function toast(m){
  var t=document.createElement('div');
  t.className='toast';
  t.textContent=m;
  document.body.appendChild(t);
  setTimeout(function(){if(t.parentNode)t.parentNode.removeChild(t)},3000);
}
function showTab(name){
  var ids=['main','tokens','trainer'];
  ids.forEach(function(t){
    var el=$('tab'+t);
    if(el)el.classList.toggle('hide',t!==name);
  });
  document.querySelectorAll('#tabNav button').forEach(function(b){
    b.classList.toggle('act',b.getAttribute('data-tab')===name);
  });
  if(name==='trainer'){
    refreshTrainer();
  }
}
window.showTab=showTab;

function startBot(){
  api('/start').then(function(r){return r.json()}).then(function(d){
    toast(d.message||'Started');
  }).catch(function(e){toast('Error: '+e.message)});
}
function stopBot(){
  var btn=$('btnStop');
  if(btn)btn.disabled=true;
  api('/stop').then(function(r){
    if(!r.ok)throw new Error('Stop request failed');
    return r.json();
  }).then(function(d){
    toast(d.message||'Stopped');
    refreshStatus();
  }).catch(function(e){
    toast('Error: '+e.message);
  }).finally(function(){
    if(btn)btn.disabled=false;
  });
}
window.startBot=startBot;
window.stopBot=stopBot;

function refreshStatus(){
  fetch('/status').then(function(r){return r.json()}).then(function(s){
    try{
      $('statusDot').className='dot'+(s&&s.running?' on':'');
      $('statusText').textContent=s&&s.running?'running ('+((s.workers&&s.workers.length)||0)+' workers)':'idle';
      $('statGenerated').textContent=(s&&s.generated)||0;
      $('statValid').textContent=(s&&s.valid)||0;
      $('statExpired').textContent=(s&&s.expired)||0;
      if(s&&s.proxies){
        $('proxyTotal').textContent=s.proxies.total||0;
        $('proxyValid').textContent=s.proxies.valid||0;
        $('proxyUsed').textContent=s.proxies.used||0;
        $('proxyInvalid').textContent=s.proxies.invalid||0;
      }
    }catch(e){}
  }).catch(function(){});
}
window.refreshStatus=refreshStatus;

function logText(logs){
  return logs.map(function(l){return (l.time||'')+' '+(l.message||l.m||'');}).join('\n');
}
function compactLogLine(entry){
  var message=String(entry.message||entry.m||'');
  var level=String(entry.level||'').toLowerCase();
  var time=entry.time||'';
  var browserFailure=/page crashed|targetclosed|target page.*closed|browser restart.*failed|page closed before/i.test(message);
  if(browserFailure)return time+' Browser page failed — see ALL LOGS for diagnostics.';
  var important=/\[FAIL\]|\[OK\]|\[ERROR\]|browser launch failed|retries exhausted|all workers stopped|stopped by user|\[diag\]/i.test(message)||level==='error';
  return important?(time+' '+message):'';
}
function renderLogs(box,logs,compact){
  if(!box)return;
  var items=logs||[];
  var rendered;
  if(compact){
    var cutoff=Date.now()/1000-300;
    rendered=items.filter(function(l){return(l.timestamp||l.time||0)>=cutoff;})
      .map(compactLogLine).filter(Boolean);
    rendered=rendered.filter(function(line,index,all){return index===0||line!==all[index-1];}).slice(-30);
  }else{
    rendered=items.map(function(l){return (l.time||'')+' '+(l.message||l.m||'');});
  }
  box.textContent=rendered.length?rendered.join('\n'):(compact?'No recent activity.':'No logs yet.');
  box.scrollTop=box.scrollHeight;
}
function refreshLogs(){
  fetch('/worker/B1/logs').then(function(r){return r.json()}).then(function(d){
    var logs=(d&&d.all_logs||d&&d.logs||[]);
    renderLogs($('logBox'),logs,true);
    var overlay=$('logOverlay');
    if(overlay&&overlay.classList.contains('on'))renderLogs($('allLogsBox'),logs,false);
  }).catch(function(){});
}
window.refreshLogs=refreshLogs;

function refreshTokens(){
  fetch('/tokens').then(function(r){return r.json()}).then(function(d){
    try{
      var list=$('tokenList');
      if(!d||!d.length){list.innerHTML='<p style="color:#8a8a92">No tokens yet.</p>';return;}
      var html='<table style="width:100%;border-collapse:collapse;font-size:12px">';
      d.forEach(function(a){
        var st=a.valid===true?'<span class="badge badge-ok">Valid</span>':'<span class="badge badge-err">Invalid</span>';
        html+='<tr style="border-bottom:1px solid #26262b">';
        html+='<td style="padding:6px">'+(a.email||'?')+'</td>';
        html+='<td style="padding:6px">'+st+'</td>';
        html+='<td style="padding:6px;font-family:monospace;font-size:10px;color:#5c5c64;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+(a.token||'').slice(0,40)+'...</td>';
        html+='</tr>';
      });
      html+='</table>';
      list.innerHTML=html;
    }catch(e){}
  }).catch(function(){});
}
window.refreshTokens=refreshTokens;

function refreshProxies(){
  api('/proxies/refresh').then(function(r){return r.json()}).then(function(d){
    toast(d.message||d.error||'Refreshed');
  }).catch(function(e){toast('Error: '+e.message)});
}
window.refreshProxies=refreshProxies;

function viewAllLogs(){
  var overlay=$('logOverlay');
  if(!overlay)return;
  overlay.classList.add('on');
  $('allLogsBox').textContent='Loading logs...';
  fetch('/worker/B1/logs').then(function(r){return r.json()}).then(function(d){
    renderLogs($('allLogsBox'),(d&&d.all_logs||d&&d.logs||[]),false);
  }).catch(function(){
    $('allLogsBox').textContent='Unable to load logs.';
  });
}
function closeAllLogs(){
  var overlay=$('logOverlay');
  if(overlay)overlay.classList.remove('on');
}
window.viewAllLogs=viewAllLogs;
window.closeAllLogs=closeAllLogs;

function copyLogs(){
  fetch('/worker/B1/logs').then(function(r){return r.json()}).then(function(d){
    var logs=(d&&d.all_logs||d&&d.logs||[]);
    var text=logs.map(function(l){return(l.time||'')+' '+(l.message||l.m||'')}).join('\n');
    copyToClipboard(text, 'Logs copied to clipboard');
  }).catch(function(e){alert('Error: '+e.message);});
}
window.copyLogs=copyLogs;

// ── Generic Clipboard Helper ──
function copyToClipboard(text, successMsg, btn){
  function done(){
    if(successMsg)toast(successMsg);
    if(btn){
      btn.classList.add('copied');
      var old = btn.textContent;
      btn.textContent = '✓ Copied!';
      setTimeout(function(){btn.classList.remove('copied');btn.textContent=old;}, 1500);
    }
  }
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(done).catch(function(){fallbackCopy(text,done);});
  }else{
    fallbackCopy(text,done);
  }
}
function fallbackCopy(text,cb){
  var ta=document.createElement('textarea');
  ta.value=text; ta.style.position='fixed'; ta.style.opacity='0';
  document.body.appendChild(ta); ta.select();
  try{document.execCommand('copy');cb();}
  catch(e){alert('Copy failed');}
  document.body.removeChild(ta);
}

// ═══════════════════════════════════════════════════════════
// REAL OFFICIAL-DEMO RUNNER UI
// ═══════════════════════════════════════════════════════════
var trainerState = {
  running: false,
  questions: [],
  latestQuestion: '',
  speed: 2.0
};

function startTrainer(){
  var speed = parseFloat($('trainerSpeedSelect').value) || 2.0;
  api('/trainer/start', {body: {speed: speed}}).then(function(r){return r.json();})
    .then(function(d){
      toast(d.message||'Real demo runner started');
      refreshTrainer();
    }).catch(function(e){toast('Error: '+e.message);});
}
function stopTrainer(){
  api('/trainer/stop').then(function(r){return r.json();})
    .then(function(d){
      toast(d.message||'Real demo runner stopped');
      refreshTrainer();
    }).catch(function(e){toast('Error: '+e.message);});
}
function stepTrainer(){
  toast('Opening the real hCaptcha demo once...');
  api('/trainer/step').then(function(r){return r.json();})
    .then(function(d){
      toast(d.message||'Real demo cycle queued');
      refreshTrainer();
    }).catch(function(e){toast('Error: '+e.message);});
}
function clearTrainerQuestions(){
  api('/trainer/clear').then(function(r){return r.json();})
    .then(function(d){
      toast('Captured questions cleared');
      refreshTrainer();
    }).catch(function(e){toast('Error: '+e.message);});
}
function updateTrainerSpeed(val){
  var speed = parseFloat(val) || 2.0;
  $('statSpeed').textContent = speed.toFixed(1) + 's';
  api('/trainer/speed', {body: {speed: speed}}).catch(function(){});
}
window.startTrainer = startTrainer;
window.stopTrainer = stopTrainer;
window.stepTrainer = stepTrainer;
window.clearTrainerQuestions = clearTrainerQuestions;
window.updateTrainerSpeed = updateTrainerSpeed;

function copySingleQuestion(qText, btn){
  copyToClipboard(qText, 'Copied: "' + qText + '"', btn);
}
window.copySingleQuestion = copySingleQuestion;

function copyLatestQuestion(){
  if(trainerState.latestQuestion){
    copyToClipboard(trainerState.latestQuestion, 'Copied question: ' + trainerState.latestQuestion);
  }
}
window.copyLatestQuestion = copyLatestQuestion;

function copyAllQuestions(){
  if(!trainerState.questions||!trainerState.questions.length){
    toast('No questions to copy');
    return;
  }
  var fullText = trainerState.questions.map(function(q, i){
    return (i+1) + '. ' + (q.question || q.full_prompt || '');
  }).join('\n');
  copyToClipboard(fullText, 'Copied ' + trainerState.questions.length + ' questions to clipboard');
}
window.copyAllQuestions = copyAllQuestions;

function exportQuestionsJson(){
  if(!trainerState.questions||!trainerState.questions.length){
    toast('No questions to export');
    return;
  }
  var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(trainerState.questions, null, 2));
  var dlAnchor = document.createElement('a');
  dlAnchor.setAttribute("href", dataStr);
  dlAnchor.setAttribute("download", "hcaptcha_demo_questions.json");
  document.body.appendChild(dlAnchor);
  dlAnchor.click();
  dlAnchor.remove();
  toast('Exported captured questions JSON');
}
window.exportQuestionsJson = exportQuestionsJson;

function renderQuestionsList(questions){
  var wrap = $('trainerQuestionsList');
  if(!wrap) return;
  wrap.textContent = '';
  if(!questions || !questions.length){
    var empty = document.createElement('div');
    empty.style.cssText = 'color:#8a8a92;font-size:12px;padding:28px 16px;text-align:center;font-family:monospace';
    empty.textContent = 'No real challenge captured yet.\nStart the runner to open the official demo.';
    wrap.appendChild(empty);
    return;
  }
  questions.forEach(function(q, idx){
    var row = document.createElement('div'); row.className = 'q-row';
    var num = document.createElement('div'); num.className = 'q-num'; num.textContent = (idx + 1) + '.';
    var body = document.createElement('div'); body.className = 'q-body';
    var title = document.createElement('div'); title.className = 'q-title';
    title.textContent = q.question || q.full_prompt || 'Challenge prompt unavailable';
    var meta = document.createElement('div'); meta.className = 'q-meta';
    var tag = document.createElement('span'); tag.className = 'badge-tag'; tag.textContent = q.type || 'REAL HCAPTCHA';
    var time = document.createElement('span'); time.textContent = q.time || '';
    meta.appendChild(tag); meta.appendChild(time);
    if(q.url){ var url = document.createElement('span'); url.style.color='#64748b'; url.textContent = '· ' + q.url; meta.appendChild(url); }
    body.appendChild(title); body.appendChild(meta);
    var btn = document.createElement('button'); btn.className = 'btn-copy-q'; btn.textContent = '📋 Copy';
    btn.addEventListener('click', function(){copySingleQuestion(title.textContent, btn);});
    row.appendChild(num); row.appendChild(body); row.appendChild(btn); wrap.appendChild(row);
  });
  wrap.scrollTop = wrap.scrollHeight;
}

function refreshTrainer(){
  fetch('/trainer/status?t=' + Date.now()).then(function(r){return r.json();})
    .then(function(s){
      trainerState.running = !!s.running;
      trainerState.questions = s.questions || [];
      trainerState.latestQuestion = s.latest_question || '';
      trainerState.speed = parseFloat(s.speed) || trainerState.speed;
      $('statCaptured').textContent = s.captured_count || 0;
      $('statBypassed').textContent = s.pass_count || 0;
      $('statCycles').textContent = s.total_cycles || 0;
      $('statSpeed').textContent = trainerState.speed.toFixed(1) + 's';
      $('qCountBadge').textContent = (s.questions && s.questions.length) || 0;
      $('btnTrainerStart').disabled = !!s.running;
      $('btnTrainerStop').disabled = !s.running;
      var badge = $('trainerStatusBadge');
      badge.className = s.running ? 'badge badge-ok' : 'badge badge-warn';
      badge.textContent = s.running ? 'REAL BROWSER ACTIVE' : 'IDLE';
      var dot = $('trainerStageDot');
      dot.className = s.running ? 'stage-dot active' : 'stage-dot';
      $('trainerStageText').textContent = s.status_text || 'Ready to open the official hCaptcha demo.';
      if(s.form){
        $('demoFormName').value = s.form.name || '';
        $('demoFormEmail').value = s.form.field || '';
        $('demoFormComment').value = s.form.comment || s.form.field || '';
      }
      var img = $('trainerModalImg');
      var ph = $('trainerPlaceholder');
      var btnCopyLatest = $('btnCopyLatest');
      var shot = s.latest_challenge_image || s.latest_screenshot;
      if(shot){
        if(img.getAttribute('data-src') !== shot){
          img.src = shot;
          img.setAttribute('data-src', shot);
        }
        img.style.display = 'block';
        if(ph) ph.style.display = 'none';
        if(btnCopyLatest) btnCopyLatest.style.display = 'inline-block';
      } else {
        img.removeAttribute('src');
        img.removeAttribute('data-src');
        img.style.display = 'none';
        if(ph) ph.style.display = 'block';
        if(btnCopyLatest) btnCopyLatest.style.display = 'none';
      }
      renderQuestionsList(s.questions);
      var ptr = $('trainerPointerLog');
      if(ptr){
        var items = s.pointer_log || [];
        if(!items.length){
          ptr.textContent = '';
        }else{
          ptr.textContent = items.slice(-6).map(function(p){
            var t = p.t ? p.t + ' ' : '';
            if(p.kind === 'click') return t + 'click ' + Math.round(p.x||0) + ', ' + Math.round(p.y||0) + (p.selector ? ' ' + p.selector : '');
            if(p.kind === 'drag') return t + 'drag ' + Math.round(p.x1||0) + ',' + Math.round(p.y1||0) + ' -> ' + Math.round(p.x2||0) + ',' + Math.round(p.y2||0);
            return t + (p.kind || '');
          }).join('  |  ');
        }
      }
    }).catch(function(){});
}
window.refreshTrainer = refreshTrainer;

// Init
refreshStatus();
refreshLogs();
refreshTokens();
refreshTrainer();
setInterval(refreshStatus,5000);
setInterval(refreshLogs,2200);
setInterval(refreshTokens,12000);
setInterval(function(){
  var tabTrainer = $('tabtrainer');
  if(tabTrainer && !tabTrainer.classList.contains('hide')){
    refreshTrainer();
  }
}, 3000);
</script>
</body></html>
'''

if __name__ == "__main__":
    main()
