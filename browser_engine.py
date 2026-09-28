"""Playwright-compatible async driver backed by nodriver + Google Chrome.

The worker imports ``async_playwright`` and ``ENGINE`` from this module, so
keeping that small contract lets the application drive a REAL Google Chrome
(google-chrome-stable) through nodriver — an undetected, CDP-native driver
with no Selenium/WebDriver layer — without changing any navigation, live-view,
or worker code. The actual Playwright-compatible facade lives in
nodriver_engine.py.
"""

from nodriver_engine import async_playwright, ENGINE

__all__ = ["async_playwright", "ENGINE"]
