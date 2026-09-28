"""End-to-end: exercise the REAL _submit_landed / _real_click_create_button /
_log_form_state methods on a simulated register form.

Case A: a real submit (form unmounts + hCaptcha CHALLENGE frame spawns).
Case B: an inert button (the "isn't clicking" case) - must NOT report landed.
Case C: a PRE-EXISTING hCaptcha WIDGET iframe (Discord mounts it with the
        form before any click) - must NOT report landed on its own, or the
        bot skips to the captcha phase while the form was never submitted.
"""
import asyncio

from browser_engine import async_playwright
from server import DiscordAutomation

# Static pre-rendered challenge iframe (OUTSIDE the form, like Discord);
# the click only removes the form. Inline iframe creation in the click
# handler crashes the patched engine (EPIPE), so keep it static.
OK_HTML = """<html><body>
<iframe title='hCaptcha challenge' srcdoc='<html><body><div id=&quot;hcaptcha-body&quot; style=&quot;height:120px&quot;>Halt! Bist du ein Mensch?</div></body></html>'></iframe>
<form id='f'>
  <input name='email' type='email' value='ok@example.com'>
  <button type='button' onclick="var f=document.getElementById('f');if(f)f.remove()">Create Account</button>
</form></body></html>"""

BLOCKED_HTML = """<html><body>
<form id='f'>
  <input name='email' type='email' value='bad@example.combad@example.com'>
  <div class='error_abc'>Format der E-Mail-Adresse ist ungültig</div>
  <button type='button'>Create Account</button>
</form></body></html>"""

# Widget iframe present WITH the form, button does nothing (the bug case).
WIDGET_HTML = """<html><body>
<form id='f'>
  <input name='email' type='email' value='ok@example.com'>
  <iframe src='https://newassets.hcaptcha.com/captcha/v1/frame.html' title='Widget containing checkbox for hCaptcha security challenge'></iframe>
  <button type='button'>Create Account</button>
</form></body></html>"""

# CHALLENGE iframe present WITH the form still on screen (Discord preloads
# the captcha iframes inside the modal) - the 2026-08-16 field bug: bot
# declared the submit landed on this while the form never went away, then
# spun on an empty pre-init widget frame.
CHALLENGE_WITH_FORM_HTML = """<html><body>
<form id='f'>
  <input name='email' type='email' value='ok@example.com'>
  <iframe src='https://newassets.hcaptcha.com/captcha/v1/abc/hcaptcha-challenge.html' title='hCaptcha challenge'></iframe>
  <button type='button'>Create Account</button>
</form></body></html>"""

# Challenge iframe + form GONE (real landed submit: click removes the form).
# Static pre-rendered challenge iframe OUTSIDE the form (Discord mounts the
# challenge as a sibling, not inside the <form>); the click removes the form.
CHALLENGE_NO_FORM_HTML = """<html><body>
<iframe title='hCaptcha challenge' srcdoc='<html><body><div id=&quot;hcaptcha-body&quot; style=&quot;height:120px&quot;>Halt! Bist du ein Mensch?</div></body></html>'></iframe>
<form id='f'>
  <input name='email' type='email' value='ok@example.com'>
  <button type='button' onclick="var f=document.getElementById('f');if(f)f.remove()">Create Account</button>
</form>
</body></html>"""

# The REAL 2026-08-16 field case: the form STAYS in the DOM (Discord layers
# the challenge over it) and the challenge iframe is genuinely RENDERED
# (painted hCaptcha content: header + 3-dots menu button). This must be
# detected as a landed submit — the strict form-gone rule kept the bot
# clicking Create Account 3x while the challenge was already up.
RENDERED_CHALLENGE_WITH_FORM_HTML = """<html><body>
<form id='f'>
  <input name='email' type='email' value='ok@example.com'>
  <iframe id='chall' title='hCaptcha challenge' src='https://newassets.hcaptcha.com/captcha/v1/abc/hcaptcha-challenge.html'></iframe>
  <button type='button'>Create Account</button>
</form>
<script>
  var doc = document.getElementById('chall').contentDocument;
  doc.open();
  doc.write('<html><body style="font-family:sans-serif"><div id="hcaptcha-body" style="height:120px"><div class="header">Halt! Bist du ein Mensch?</div><button id="menu-info" aria-label="About hCaptcha &amp; Accessibility Options">...</button><p>W&auml;hle ein Bild aus</p></div></body></html>');
  doc.close();
</script>
</body></html>"""


def data_url(html: str) -> str:
    return ("data:text/html," + html.replace(" ", "%20").replace("<", "%3C")
            .replace(">", "%3E").replace('"', "%22").replace("'", "%27"))


async def check_landed(bot, label):
    before = await bot._submit_landed(timeout=2.0)
    print(f"{label} before click, landed:", repr(before))
    clicked = await bot._real_click_create_button()
    print(f"{label} real click sent:", clicked)
    await asyncio.sleep(1.0)
    after = await bot._submit_landed(timeout=2.0)
    print(f"{label} after click, landed:", repr(after))
    return before, after


async def main():
    print("starting playwright...")
    pw = await async_playwright().start()
    b = await pw.chromium.launch(headless=True)
    ctx = await b.new_context()
    page = await ctx.new_page()
    bot = DiscordAutomation(headless=True)
    bot._page = page

    # ── Case A: real submit lands (form gone + challenge frame) ──
    print("case A: real submit lands")
    await page.goto(data_url(OK_HTML))
    await asyncio.sleep(0.5)
    before, after = await check_landed(bot, "A")
    assert after, "real submit should be detected as landed (form gone)"

    # ── Case B: inert form (the 'isn't clicking' case) ──
    print("case B: inert form")
    await page.goto(data_url(BLOCKED_HTML))
    await asyncio.sleep(0.5)
    before, after = await check_landed(bot, "B")
    assert after == "", "inert form must NOT report a landed submit"
    print("B4 form-state dump:")
    await bot._log_form_state("test (blocked)")

    # ── Case C: pre-existing widget iframe + unsent form ──
    print("case C: pre-existing widget iframe")
    await page.goto(data_url(WIDGET_HTML))
    await asyncio.sleep(0.5)
    before, after = await check_landed(bot, "C")
    assert before == "", "pre-existing widget must not be 'landed' before the click"
    assert after == "", "pre-existing widget alone must not report a landed submit"

    # ── Case D: EMPTY challenge iframe + form still on screen ──
    # A preloaded shell (no painted content) must NOT be treated as landed.
    print("case D: empty challenge shell with form still up")
    await page.goto(data_url(CHALLENGE_WITH_FORM_HTML))
    await asyncio.sleep(0.5)
    before, after = await check_landed(bot, "D")
    assert before == "", "empty challenge shell must not be 'landed' before the click"
    assert after == "", "empty challenge shell must not report a landed submit"

    # ── Case E: challenge present, form REMOVED = real landed submit ──
    print("case E: challenge iframe, form gone")
    await page.goto(data_url(CHALLENGE_NO_FORM_HTML))
    await asyncio.sleep(0.5)
    before, after = await check_landed(bot, "E")
    assert "captcha" in after, "challenge + form gone should be a landed submit"

    # ── Case F: RENDERED challenge + form STILL in DOM ──
    # The real field case: Discord keeps the form and overlays the painted
    # challenge. Must be detected as landed.
    print("case F: rendered challenge with form still up")
    await page.goto(data_url(RENDERED_CHALLENGE_WITH_FORM_HTML))
    await asyncio.sleep(0.5)
    before, after = await check_landed(bot, "F")
    assert "captcha" in after, "rendered challenge must report a landed submit even with the form up"

    print("SUBMIT TEST PASSED")
    await b.close()


asyncio.run(main())
