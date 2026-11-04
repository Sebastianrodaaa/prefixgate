"""1280x400 black-and-white banner: pixel blackletter name, mono tagline, two dithered public-domain engravings."""
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "art"
SRC = {  # Wikimedia Commons, public domain (credited in NOTICE.md)
    "dore.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Gustave_Dor%C3%A9_-_Dante_Alighieri_-_Inferno_-_Plate_8_%28Canto_III_-_Abandon_all_hope_ye_who_enter_here%29.jpg/960px-Gustave_Dor%C3%A9_-_Dante_Alighieri_-_Inferno_-_Plate_8_%28Canto_III_-_Abandon_all_hope_ye_who_enter_here%29.jpg",
    "durer.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/D%C3%BCrer_Melancholia_I.jpg/960px-D%C3%BCrer_Melancholia_I.jpg",
    "pirata.ttf": "https://github.com/google/fonts/raw/main/ofl/pirataone/PirataOne-Regular.ttf",
}
W, H, INK, ART = 1280, 400, 232, 120


def get(name):
    p = CACHE / name
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(SRC[name], headers={"User-Agent": "prefixgate-banner/0.1"})
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return p


def engraving(name, box, width, fade_left, gamma):
    """Crop, scale to banner height, fade toward the centre, then 2-tone Floyd-Steinberg dither."""
    im = Image.open(get(name)).convert("L").crop(box)
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS).crop((0, 0, width, H))
    a = (np.asarray(im, float) / 255) ** gamma
    ramp = np.linspace(1, 0, width) if fade_left else np.linspace(0, 1, width)
    a = a * np.clip(ramp * 1.6, 0, 1)[None, :] ** 1.3
    bw = Image.fromarray((a * 255).astype(np.uint8)).convert("1")  # PIL's default "1" conversion dithers (FS)
    return Image.fromarray(np.where(np.asarray(bw), ART, 0).astype(np.uint8))


def pixel_text(text, font, size, scale):
    f = ImageFont.truetype(str(font), size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 2, b - t + 2), 0)
    ImageDraw.Draw(im).text((1 - l, 1 - t), text, font=f, fill=255)
    im = im.point(lambda v: INK if v > 110 else 0)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main():
    out = Image.new("L", (W, H), 0)
    out.paste(engraving("dore.jpg", (200, 320, 620, 762), 380, True, 0.8), (0, 0))
    out.paste(engraving("durer.jpg", (560, 330, 960, 790), 380, False, 1.6), (W - 380, 0))
    name = pixel_text("prefixgate", get("pirata.ttf"), 46, 3)
    out.paste(name, ((W - name.width) // 2, 92))
    mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    for i, line in enumerate(("a calibrated gate that rules", "before the tool call is finished")):
        t = pixel_text(line, mono, 11, 2)
        out.paste(t.point(lambda v: 170 if v else 0), ((W - t.width) // 2, 250 + i * 34))
    dst = ROOT / "assets" / "banner.png"
    out.convert("RGB").save(dst)
    print(dst, out.size)


if __name__ == "__main__":
    main()
