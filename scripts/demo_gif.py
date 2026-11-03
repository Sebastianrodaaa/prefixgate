"""Hero screenshot, live-mode screenshot and a demo GIF from the running dashboard (real scores, real gate).

Needs `python3 app/server.py 8765` running. Uses Playwright with an installed Chrome/Chromium
(set CHROME=/path/to/chrome, or CDP=http://host:port to drive an already-open browser).
"""
import io, json, os, sys, urllib.parse, urllib.request
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
URL = os.environ.get("URL", "http://127.0.0.1:8765")


def pick():
    """The demo call: an attacked held-out call that the stream denies, with the largest lead that agrees with batch."""
    calls = json.load(urllib.request.urlopen(URL + "/api/calls?alpha=0.05,0.1"))
    ok = [c for c in calls if c["attack"] and c["stream"] and c["batch"] and c["stream"][0] == "deny" == c["batch"][0]
          and 120 <= c["T"] <= 400]
    return max(ok, key=lambda c: c["T"] - c["stream"][1]) if ok else calls[0]


def frame(page):
    return Image.open(io.BytesIO(page.screenshot())).convert("L")


def main():
    c = pick()
    with sync_playwright() as p:
        if os.environ.get("CDP"):
            b = p.chromium.connect_over_cdp(os.environ["CDP"])
            page = b.new_page()
        else:
            b = p.chromium.launch(executable_path=os.environ.get("CHROME"))
            page = b.new_page()
        page.set_viewport_size({"width": 1600, "height": 960})
        page.goto(f"{URL}/?id={urllib.parse.quote(c['id'], safe='')}")
        page.wait_for_function("typeof cur !== 'undefined' && cur && cur.id")
        page.wait_for_timeout(400)
        frame(page).save(OUT / "hero.png")
        frames, T = [], c["T"]
        stops = sorted({max(1, round(T * i / 48)) for i in range(1, 49)} | {c["stream"][1]})
        for t in stops:
            page.evaluate(f"pos={t};document.getElementById('scrub').value={t};draw()")
            im = frame(page).resize((1200, 720), Image.LANCZOS)
            frames.append(im)
            if t == c["stream"][1]:
                frames += [im] * 10  # hold on the commit
        frames += [frames[-1]] * 14
        frames[0].save(OUT / "demo.gif", save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
        page.goto(f"{URL}/?live")
        page.wait_for_function("typeof S !== 'undefined' && S.runs")
        page.click("#go")
        page.wait_for_function("cur && cur.live && pos >= cur.T", timeout=120000)
        page.wait_for_timeout(300)
        frame(page).save(OUT / "live.png")
        b.close()
    print("demo call", c["id"], "T", c["T"], "stream", c["stream"], "batch", c["batch"], file=sys.stderr)


if __name__ == "__main__":
    main()
