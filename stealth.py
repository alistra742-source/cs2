"""Functional browser settings for the nodriver / Google Chrome engine.

This module deliberately keeps identity modification out of page JavaScript
and out of spoofed launch flags. Chrome OWNS the fingerprint: it is a real
user's browser — genuine JS/TLS/HTTP2 fingerprints, real fonts, real
capabilities. nodriver only strips the automation tells (navigator.webdriver,
CDP hooks, headless UA artifacts); nothing here needs to fake an identity.
What's left for the application is stable container/runtime behavior and
standard context options.
"""

import random

from browser_engine import ENGINE


def launch_args(headless: bool = True) -> list:
    """Stable container-runtime Chrome flags.

    Headless is handled by the engine (nodriver adds --headless=new itself);
    these flags only make Chrome well-behaved inside an unprivileged 1 GB
    container (no sandbox setup race, no /dev/shm exhaustion, no GPU, no
    first-run / component-update / background network noise).
    """
    return [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--mute-audio",
    ]


def build_init_script(fingerprint: dict, ua: str) -> str:
    """No page-level identity overrides are injected for real Chrome."""
    return f"// {ENGINE}: no page-level identity overrides"


def build_context_options(fingerprint: dict, ua: str, proxy=None,
                          viewport=None) -> dict:
    """Return only standard, engine-neutral Playwright context options.

    These are exactly the options nodriver_engine applies per tab (viewport
    via Emulation.setDeviceMetricsOverride, color scheme via
    Emulation.setEmulatedMedia, etc.); identity is engine-owned.
    """
    vp = viewport or {"width": 1280, "height": 720}
    return {
        "viewport": vp,
        "ignore_https_errors": True,
        "color_scheme": random.choice(["light", "light", "dark"]),
        "java_script_enabled": True,
    }


async def apply_cdp_stealth(context, page) -> None:
    """Contract no-op.

    nodriver performs its automation-tell stripping (webdriver property,
    Runtime.enable detection, headless UA fix, ...) at the CDP level inside
    the driver itself, before any page script runs — there is nothing for
    the application layer to strip.
    """
    return
