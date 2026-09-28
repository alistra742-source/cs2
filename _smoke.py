import asyncio, json, os
from browser_engine import async_playwright

HTML = ("<html><body>"
        "<button id='b'>Submit</button>"
        "<input id='email' placeholder='Email'>"
        "<input id='pass' type='password'>"
        "<div id='drag' style='width:100px;height:50px;background:red'></div>"
        "<iframe id='f' srcdoc=\"<h2>inner text</h2><button id='ib'>inner btn</button>\" style='position:absolute;left:40px;top:70px;width:300px;height:200px'></iframe>"
        "</body></html>")
url = ("data:text/html," + HTML.replace(" ", "%20").replace("<", "%3C")
       .replace(">", "%3E").replace('"', "%22").replace("'", "%27")
       .replace("#", "%23"))


async def main():
    pw = await async_playwright().start()
    b = await pw.chromium.launch(headless=True)
    ctx = await b.new_context()
    page = await ctx.new_page()
    await page.goto(url)
    await asyncio.sleep(0.8)

    await page.locator("#email").fill("test@example.com")
    v = await page.evaluate("document.getElementById('email').value")
    print("fill:", v)

    await page.locator("#pass").type("secret123")
    v2 = await page.evaluate("document.getElementById('pass').value")
    print("type:", v2)

    await page.locator("#b").click()
    print("click ok")

    el = await page.wait_for_selector("#b", timeout=3000)
    print("wait_for_selector:", el is not None)

    await page.keyboard.press("Tab")
    print("keyboard ok")

    box = await page.locator("#drag").bounding_box()
    await page.mouse.move(box["x"] + 10, box["y"] + 25, steps=5)
    await page.mouse.down()
    await page.mouse.move(box["x"] + 60, box["y"] + 25, steps=5)
    await page.mouse.up()
    print("mouse drag ok")

    fl = page.frame_locator("#f")
    h2 = await fl.locator("h2").inner_text()
    print("frame_locator h2:", h2)
    cf = await (await page.locator("#f").first.element_handle()).content_frame()
    print("content_frame:", cf is not None, (cf.url or "")[:20] if cf else "")

    btn = await page.get_by_role("button", name="Submit").first.inner_text()
    print("get_by_role:", btn)

    calls = []

    async def handler(route):
        calls.append(route.url)
        await route.abort()

    await page.route("**/blocked.js", handler)
    print("route registered ok")

    st = await page.evaluate(
        """() => { return JSON.stringify({url: location.href, title: document.title}); }""")
    print("nav-state:", json.loads(st))

    await b.close()
    print("FULL SMOKE OK")


asyncio.run(main())
