"""Render synthetic Khmer text-line images across many fonts and styles.

Targets gaps in the existing training data: font variety (Google Fonts Khmer families), italic
(Kantumruy Pro Italic plus a synthetic shear, since almost no Khmer font ships an italic),
handwriting-like fonts (Fasthand, Freehand), scanned / low-res documents, and light/dark UI
panels.

Khmer needs complex shaping (COENG subscripts, reordered vowel signs). Pillow only shapes it with
the libraqm layout engine, so fonts are always opened with ``Layout.RAQM`` and ``require_raqm``
fails closed when raqm is missing — unshaped renders would silently produce wrong labels.
"""
from __future__ import annotations

import io
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

# google/fonts repo path -> kind. Kinds steer font sampling in render_sample.
GOOGLE_KHMER_FONTS = {
    "ofl/angkor/Angkor-Regular.ttf": "display",
    "ofl/battambang/Battambang-Regular.ttf": "text",
    "ofl/battambang/Battambang-Bold.ttf": "text",
    "ofl/battambang/Battambang-Light.ttf": "text",
    "ofl/bayon/Bayon-Regular.ttf": "display",
    "ofl/bokor/Bokor-Regular.ttf": "display",
    "ofl/chenla/Chenla.ttf": "display",
    "ofl/content/Content-Regular.ttf": "text",
    "ofl/content/Content-Bold.ttf": "text",
    "ofl/dangrek/Dangrek-Regular.ttf": "display",
    "ofl/fasthand/Fasthand-Regular.ttf": "handwriting",
    "ofl/freehand/Freehand-Regular.ttf": "handwriting",
    "ofl/hanuman/Hanuman[wght].ttf": "text",
    "ofl/kantumruypro/KantumruyPro[wght].ttf": "text",
    "ofl/kantumruypro/KantumruyPro-Italic[wght].ttf": "italic",
    "ofl/kdamthmorpro/KdamThmorPro-Regular.ttf": "text",
    "ofl/khmer/Khmer.ttf": "text",
    "ofl/kohsantepheap/KohSantepheap-Regular.ttf": "text",
    "ofl/kohsantepheap/KohSantepheap-Bold.ttf": "text",
    "ofl/konkhmersleokchher/KonkhmerSleokchher-Regular.ttf": "display",
    "ofl/koulen/Koulen-Regular.ttf": "display",
    "ofl/metal/Metal-Regular.ttf": "display",
    "ofl/moul/Moul-Regular.ttf": "display",
    "ofl/moulpali/Moulpali-Regular.ttf": "display",
    "ofl/nokora/Nokora[wght].ttf": "text",
    "ofl/notosanskhmer/NotoSansKhmer[wdth,wght].ttf": "text",
    "ofl/notoserifkhmer/NotoSerifKhmer[wdth,wght].ttf": "text",
    "ofl/odormeanchey/OdorMeanChey-Regular.ttf": "text",
    "ofl/preahvihear/Preahvihear-Regular.ttf": "text",
    "ofl/siemreap/Siemreap.ttf": "text",
    "ofl/suwannaphum/Suwannaphum-Regular.ttf": "text",
    "ofl/suwannaphum/Suwannaphum-Bold.ttf": "text",
    "ofl/taprom/Taprom-Regular.ttf": "display",
}

# Latin-only OFL fonts for clean English UI/document lines. The Khmer fine-tune misread
# plain English ("Warrantly", "Batter", lowercased "condition") because English in its data is
# mostly scanned Gutenberg text. Khmer text never picks these (no Khmer glyphs). Eval renders
# English in macOS system fonts (Georgia, Arial, Verdana, New York, SF, Tahoma), all held out.
GOOGLE_LATIN_FONTS = {
    "ofl/notosans/NotoSans[wdth,wght].ttf": "latin",
    "ofl/notosans/NotoSans-Italic[wdth,wght].ttf": "latin",
    "ofl/notoserif/NotoSerif[wdth,wght].ttf": "latin",
    "ofl/notoserif/NotoSerif-Italic[wdth,wght].ttf": "latin",
    "ofl/notoserifdisplay/NotoSerifDisplay[wdth,wght].ttf": "latin",
    "ofl/opensans/OpenSans[wdth,wght].ttf": "latin",
    "ofl/lato/Lato-Regular.ttf": "latin",
    "ofl/lato/Lato-Bold.ttf": "latin",
    "ofl/sourcesans3/SourceSans3[wght].ttf": "latin",
    "ofl/sourceserif4/SourceSerif4[opsz,wght].ttf": "latin",
    "ofl/ptserif/PT_Serif-Web-Regular.ttf": "latin",
    "ofl/ptserif/PT_Serif-Web-Bold.ttf": "latin",
    "ofl/inter/Inter[opsz,wght].ttf": "latin",
    "ofl/montserrat/Montserrat[wght].ttf": "latin",
    "ofl/roboto/Roboto[wdth,wght].ttf": "latin",
    "ofl/robotomono/RobotoMono[wght].ttf": "latin",
    "ofl/nunito/Nunito[wght].ttf": "latin",
    "ofl/poppins/Poppins-Regular.ttf": "latin",
    "ofl/poppins/Poppins-Bold.ttf": "latin",
    "ofl/firasans/FiraSans-Regular.ttf": "latin",
    "ofl/firasans/FiraSans-Bold.ttf": "latin",
    "ofl/crimsontext/CrimsonText-Regular.ttf": "latin",
    "ofl/ebgaramond/EBGaramond[wght].ttf": "latin",
    "ofl/merriweather/Merriweather[opsz,wdth,wght].ttf": "latin",
    "ofl/worksans/WorkSans[wght].ttf": "latin",
    "ofl/raleway/Raleway[wght].ttf": "latin",
}
# Khmer OS family (Danh Hong / Open Forum of Cambodia, LGPL-2.1+), the faces most Cambodian documents and
# sites use. Fetched from Debian's fonts-khmeros (scripts/fetch_fonts.py). Before v7 these were eval-only;
# the screen / slice eval now keeps Apple's Khmer MN and Khmer Sangam MN as the held-out fonts.
KHMER_OS_FONTS = {
    "khmeros/KhmerOS.ttf": "text",
    "khmeros/KhmerOSbattambang.ttf": "text",
    "khmeros/KhmerOScontent.ttf": "text",
    "khmeros/KhmerOSsiemreap.ttf": "text",
    "khmeros/KhmerOSsys.ttf": "text",
    "khmeros/KhmerOSbokor.ttf": "display",
    "khmeros/KhmerOSmetalchrieng.ttf": "display",
    "khmeros/KhmerOSmuol.ttf": "display",
    "khmeros/KhmerOSmuollight.ttf": "display",
    "khmeros/KhmerOSmuolpali.ttf": "display",
    "khmeros/KhmerOSfasthand.ttf": "handwriting",
    "khmeros/KhmerOSfreehand.ttf": "handwriting",
}
ALL_FONTS = {**GOOGLE_KHMER_FONTS, **KHMER_OS_FONTS, **GOOGLE_LATIN_FONTS}

KIND_WEIGHTS = {"text": 0.55, "handwriting": 0.2, "display": 0.15, "italic": 0.1, "latin": 1.0}
WIDE_STRIP_P = 0.5  # UI renders: app line strips span the whole clip width
SYNTHETIC_ITALIC_P = 0.25  # extra shear on non-italic fonts
COENG = "្"
_IGNORABLE = {0x20, 0x200B, 0x200C, 0x200D}  # spaces / zero-width: need no glyph


@dataclass(frozen=True)
class FontInfo:
    path: Path
    kind: str
    codepoints: frozenset[int]


def font_info(path: Path, kind: str) -> FontInfo:
    from fontTools.ttLib import TTFont

    return FontInfo(Path(path), kind, frozenset(TTFont(str(path), lazy=True).getBestCmap()))


def load_fonts(font_dir: Path) -> list[FontInfo]:
    fonts = []
    for repo_path, kind in ALL_FONTS.items():
        path = Path(font_dir) / Path(repo_path).name
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run scripts/fetch_fonts.py")
        fonts.append(font_info(path, kind))
    return fonts


def covers(font: FontInfo, text: str) -> bool:
    """True when the font has a glyph for every visible character (no tofu boxes)."""
    return all(ord(c) in font.codepoints or ord(c) in _IGNORABLE for c in text)


def require_raqm() -> None:
    if not features.check("raqm"):
        raise RuntimeError("Pillow lacks libraqm; Khmer would render unshaped. Install libraqm and reinstall Pillow.")


def render_text(text, font_path, size, fg, bg, pad=(8, 6)) -> Image.Image:
    """Shaped single-line render on a tight canvas plus padding."""
    require_raqm()
    font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    px, py = pad
    image = Image.new("RGB", (right - left + 2 * px, bottom - top + 2 * py), bg)
    ImageDraw.Draw(image).text((px - left, py - top), text, font=font, fill=fg)
    return image


def shear(image: Image.Image, k: float, bg) -> Image.Image:
    """Synthetic italic: lean right by ``k`` (dx per dy). The canvas widens by ``ceil(k*h)`` so
    nothing clips; the bottom row stays put and the top row shifts right."""
    w, h = image.size
    return image.transform(
        (w + math.ceil(k * h), h),
        Image.Transform.AFFINE,
        (1, k, -k * h, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=bg,
    )


def degrade_document(image: Image.Image, rng: random.Random) -> Image.Image:
    """Scanned / photocopied look: low resolution, optional blur, sensor noise, JPEG artifacts.
    Noise is drawn from ``rng`` so renders are reproducible."""
    w, h = image.size
    scale = rng.uniform(0.35, 0.8)
    out = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    out = out.resize((w, h), Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(4, 14), (h, w, 1))
    out = Image.fromarray(np.clip(np.asarray(out, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=rng.randint(30, 75))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _colors(style: str, rng: random.Random):
    if style == "ui_dark":
        bg = rng.randint(15, 45)
        fg = rng.randint(205, 250)
        return (fg, fg, fg), (bg, bg, bg + rng.randint(0, 8))
    if style == "ui_light":
        bg = rng.randint(235, 255)
        fg = rng.randint(10, 60)
        return (fg, fg, fg), (bg, bg, bg)
    paper = rng.randint(225, 255)
    ink = rng.randint(0, 70)
    return (ink, ink, ink + rng.randint(0, 40)), (paper, paper, rng.randint(paper - 15, paper))


def render_sample(text: str, fonts: list[FontInfo], rng: random.Random):
    """Render one line in a random font/style. Returns ``(image, meta)``, or ``None`` when no
    font has glyphs for every character of ``text``."""
    usable = [f for f in fonts if covers(f, text)]
    if not usable:
        return None
    kinds = sorted({f.kind for f in usable})
    kind = rng.choices(kinds, weights=[KIND_WEIGHTS[k] for k in kinds])[0]
    font = rng.choice([f for f in usable if f.kind == kind])
    style = rng.choices(["plain", "document", "ui_dark", "ui_light"], weights=[0.35, 0.35, 0.15, 0.15])[0]
    fg, bg = _colors(style, rng)
    image = render_text(text, font.path, rng.randint(22, 64), fg, bg, pad=(rng.randint(4, 24), rng.randint(3, 16)))
    italic = kind == "italic"
    if not italic and rng.random() < SYNTHETIC_ITALIC_P:
        image = shear(image, rng.uniform(0.12, 0.3), bg)
        italic = True
    if style.startswith("ui") and rng.random() < WIDE_STRIP_P:
        wide = Image.new("RGB", (image.width + int(image.width * rng.uniform(0.1, 1.5)), image.height), bg)
        wide.paste(image, (0, 0))
        image = wide
    if style == "document":
        image = degrade_document(image, rng)
    return image, {"font": font.path.name, "kind": kind, "style": style, "italic": italic}


def _ink_rows(image: Image.Image, bg) -> np.ndarray:
    px = np.asarray(image, dtype=np.int16)
    return (np.abs(px - np.array(bg, dtype=np.int16)) > 40).any(axis=2).sum(axis=1)


def render_screen_sample(text: str, neighbours: tuple[str | None, str | None], fonts: list[FontInfo],
                         rng: random.Random, sizes: tuple[int, int] = (11, 24), dark: float = 0.3,
                         bold: float = 0.0, busy: float = 0.0):
    """One line cut from a rendered screen paragraph, the way the app's line splitter cuts it.

    Screens set Khmer at 11-24 px with tight line heights, so the previous / next line's
    subscripts and above-marks reach into the strip, and the strip spans the whole clip width.
    Renders up to three lines (``neighbours`` = text above / below, or None) at 1x or 2x and
    cuts the middle one at the sparsest ink row between baselines. Returns ``(image, meta)``, or
    ``None`` when no text font covers every line.

    sizes / dark / bold steer the style mix (v4 targets small, dark and bold text, where v3 errs
    most); bold is a 1px stroke at >= 20 px. busy is the share set on a photo / gradient /
    texture panel instead of a flat one (v5, see busy_background). The defaults draw exactly what
    v3's synth_screen drew."""
    lines = [t for t in (neighbours[0], text, neighbours[1]) if t]
    usable = [f for f in fonts if f.kind in ("text", "italic") and all(covers(f, t) for t in lines)]
    if not usable:
        return None
    font = rng.choice(usable)
    style = rng.choices(["screen_light", "screen_dark"], weights=[1 - dark, dark])[0]
    fg, bg = _colors("ui_dark" if style == "screen_dark" else "ui_light", rng)
    size = rng.randint(*sizes) * rng.choice([1, 2])
    # A 1px stroke only reads as bold from ~20 px (2x); below that it smears the glyphs shut.
    stroke = 1 if bold and size >= 20 and rng.random() < bold else 0
    pitch = int(size * rng.uniform(1.25, 1.8))
    face = ImageFont.truetype(str(font.path), size, layout_engine=ImageFont.Layout.RAQM)
    ascent, descent = face.getmetrics()
    draw0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    widths = [draw0.textlength(t, font=face) for t in lines]
    pad_x = rng.randint(2, 20)
    width = int(max(widths) * rng.uniform(1.0, 1.4)) + 2 * pad_x
    top = ascent + descent  # room for marks above the first line
    height = top + pitch * (len(lines) - 1) + 2 * (ascent + descent)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    baselines = [top + ascent + k * pitch for k in range(len(lines))]
    for t, y in zip(lines, baselines):
        draw.text((pad_x, y), t, font=face, fill=fg, anchor="ls", stroke_width=stroke, stroke_fill=fg)
    rows = _ink_rows(image, bg)
    mid = lines.index(text) if neighbours[0] is None else 1
    ys = np.nonzero(rows)[0]
    if ys.size == 0:
        return None

    def valley(a: int, b: int) -> int:
        a, b = max(0, a), min(len(rows), b)
        return a + int(np.argmin(rows[a:b])) if a < b else a

    y_mid = baselines[mid]
    cut_top = valley(baselines[mid - 1], y_mid - ascent // 2) if mid > 0 else max(0, int(ys[0]) - rng.randint(1, 8))
    has_next = mid < len(lines) - 1
    cut_bottom = (valley(y_mid, baselines[mid + 1] - ascent // 2) if has_next
                  else min(height, int(ys[-1]) + 1 + rng.randint(1, 8)))
    image = image.crop((0, cut_top, width, cut_bottom))
    if busy and rng.random() < busy:
        image, style = _on_busy_background(image, fg, bg, rng)
    italic = font.kind == "italic"
    return image, {"font": font.path.name, "kind": font.kind, "style": style, "italic": italic}


BUSY_KINDS = ("gradient", "blobs", "fractal", "scene")


def busy_background(width: int, height: int, rng: random.Random, kind: str | None = None) -> np.ndarray:
    """A float RGB panel (h, w, 3) like the photos, gradients and textures text sits on in apps.

    Generated, never real photos. Deliberately unlike training/eval's noisy fixtures (two-colour
    CSS gradients and one blurred-noise texture): 3-4 colour stops, blurred shapes, multi-octave
    noise and a two-region "scene", each with optional film grain."""
    kind = kind or rng.choice(BUSY_KINDS)
    nprng = np.random.default_rng(rng.randrange(2**32))
    palette = nprng.uniform(0, 255, (4, 3))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    if kind == "gradient":
        a = rng.uniform(0, math.pi)
        t = xs * math.cos(a) + ys * math.sin(a)
        t = (t - t.min()) / max(1e-6, np.ptp(t))
        stops = np.sort(nprng.uniform(0, 1, rng.randint(3, 4)))
        stops[0], stops[-1] = 0.0, 1.0
        panel = np.stack([np.interp(t, stops, palette[: len(stops), c]) for c in range(3)], axis=-1)
    elif kind == "fractal":
        t = np.zeros((height, width), np.float32)
        for octave in range(rng.randint(3, 5)):
            cells = 2 + 2 ** octave
            grid = Image.fromarray(nprng.random((cells, cells * max(1, width // max(1, height)))).astype(np.float32))
            t += np.asarray(grid.resize((width, height), Image.BILINEAR)) / 2 ** octave
        t = (t - t.min()) / max(1e-6, np.ptp(t))
        panel = np.stack([np.interp(t, [0, 0.5, 1], palette[:3, c]) for c in range(3)], axis=-1)
    else:
        img = Image.new("RGB", (width, height), tuple(int(v) for v in palette[0]))
        draw = ImageDraw.Draw(img)
        if kind == "scene":  # sky / ground, or a wall and a floor, split at a tilted line
            y0, y1 = rng.uniform(0, height), rng.uniform(0, height)
            draw.polygon([(0, y0), (width, y1), (width, height), (0, height)], fill=tuple(int(v) for v in palette[1]))
        for _ in range(rng.randint(4, 16) if kind == "blobs" else rng.randint(1, 5)):
            r = rng.uniform(0.2, 1.2) * height
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            draw.ellipse((cx - r, cy - r * rng.uniform(0.5, 1.5), cx + r, cy + r), fill=tuple(nprng.integers(0, 256, 3).tolist()))
        panel = np.asarray(img.filter(ImageFilter.GaussianBlur(rng.uniform(1, max(1.5, height / 3)))), np.float32)
    if rng.random() < 0.5:
        panel = panel + nprng.normal(0, rng.uniform(3, 14), panel.shape)
    return np.clip(panel, 0, 255)


def _on_busy_background(image: Image.Image, fg, bg, rng: random.Random):
    """Move the rendered glyphs (anti-aliasing kept as alpha) onto a busy_background panel, in
    black or white, whichever stands out more from the panel under the text."""
    px = np.asarray(image, np.float32)
    span = max(1.0, float(np.abs(np.array(fg, np.float32) - np.array(bg, np.float32)).max()))
    alpha = np.clip(np.abs(px - np.array(bg, np.float32)).max(axis=2, keepdims=True) / span, 0, 1)
    panel = busy_background(image.width, image.height, rng)
    lum = float((panel.mean(axis=2, keepdims=True) * alpha).sum() / max(1.0, float(alpha.sum())))
    ink = np.full(3, rng.randint(0, 50) if lum > 128 else rng.randint(215, 255), np.float32)
    # Unreadable strips would teach the model to guess: fade a panel too close to the ink away from it.
    under = np.abs(panel.mean(axis=2, keepdims=True) - ink[0])
    if float((under * alpha).sum() / max(1.0, float(alpha.sum()))) < 110:
        panel = panel * 0.55 + (255.0 if ink[0] < 128 else 0.0) * 0.45
    out = panel * (1 - alpha) + ink * alpha
    return Image.fromarray(out.astype(np.uint8)), "screen_busy"


def _is_khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿"


def _starts_cluster(ch: str, prev: str) -> bool:
    # A consonant / independent vowel starts a cluster unless a COENG just subscripted it;
    # any non-Khmer character starts its own unit.
    return not _is_khmer(ch) or ("ក" <= ch <= "ឳ" and prev != COENG)


def split_clusters(run: str) -> list[str]:
    """Split an unbroken run into Khmer orthographic clusters (safe cut points)."""
    clusters: list[str] = []
    for i, ch in enumerate(run):
        if clusters and not _starts_cluster(ch, run[i - 1]):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def chunk_line(text: str, max_chars: int) -> list[str]:
    """Pack a label into renderable lines of at most ``max_chars``. Words split on spaces and
    ZWSP; words longer than the limit (common — Khmer rarely uses spaces) are cut at cluster
    boundaries. ZWSP is dropped: it is invisible, so it can't be learned from pixels."""
    chunks: list[str] = []
    cur = ""

    def push(piece: str, sep: str) -> None:
        nonlocal cur
        cand = f"{cur}{sep}{piece}" if cur else piece
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            cur = piece

    for word in text.replace("​", " ").split():
        if len(word) <= max_chars:
            push(word, " ")
        else:
            for i, cluster in enumerate(split_clusters(word)):
                push(cluster, " " if i == 0 else "")
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) <= max_chars]


def build_corpus(texts, exclude, max_chars: int = 32, min_chars: int = 2) -> list[str]:
    """Chunk labels into unique renderable lines, dropping any that equal, or contain, eval text
    (fixtures + UI strings) or one of its chunks."""
    banned = {e.strip() for e in exclude if e.strip()}
    for text in list(banned):
        banned.update(chunk_line(text, max_chars))
    contained = [b for b in banned if len(b) >= 4]
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for chunk in chunk_line(text, max_chars):
            if len(chunk) < min_chars or chunk in seen or chunk in banned:
                continue
            if any(b in chunk for b in contained):
                continue
            seen.add(chunk)
            out.append(chunk)
    return out


def corpus_part(text: str) -> str:
    """Stable ~2% val / 98% train assignment by content hash, so train and val renders never
    share a line across runs or seeds."""
    return "val" if zlib.crc32(text.encode("utf-8")) % 100 < 2 else "train"


def is_latin_only(text: str) -> bool:
    """Printable ASCII (plus a few common typographic marks) with at least one letter."""
    return any(c.isalpha() for c in text) and all(" " <= c <= "~" or c in "’‘“”–—…°€£•" for c in text)


def augment_case(text: str, rng: random.Random) -> str:
    """Chunked prose mostly starts lowercase mid-sentence; UI labels start with a capital.
    Mix in capitalised / title / upper case so the decoder doesn't learn lowercase starts."""
    r = rng.random()
    if r < 0.5:
        return text[:1].upper() + text[1:]
    if r < 0.65:
        return " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    if r < 0.7:
        return text.upper()
    return text


def _warp(image: Image.Image, rng: random.Random, amp: float, scale: float, bg) -> Image.Image:
    """Smooth random displacement (elastic distortion): a coarse noise grid upsampled into a
    per-pixel x/y shift of up to ``amp`` px, so strokes wobble the way a pen does."""
    w, h = image.size
    nprng = np.random.default_rng(rng.getrandbits(32))
    gw, gh = max(2, int(w / scale)), max(2, int(h / scale))
    dx = np.asarray(Image.fromarray(nprng.uniform(-1, 1, (gh, gw)).astype(np.float32)).resize((w, h), Image.BICUBIC)) * amp
    dy = np.asarray(Image.fromarray(nprng.uniform(-1, 1, (gh, gw)).astype(np.float32)).resize((w, h), Image.BICUBIC)) * amp
    ys, xs = np.mgrid[0:h, 0:w]
    sx = np.clip(np.rint(xs + dx).astype(int), 0, w - 1)
    sy = np.clip(np.rint(ys + dy).astype(int), 0, h - 1)
    return Image.fromarray(np.asarray(image)[sy, sx])


def render_handwriting_sample(text: str, fonts: list[FontInfo], rng: random.Random):
    """A handwritten-looking line (v6): a handwriting font, then pen-like distortion -- elastic
    wobble, a wavy baseline, uneven stroke weight -- in blue or black ink on white / cream paper,
    sometimes ruled, scanned (blur, JPEG). The real handwriting set is short words in dark pen,
    loose strokes, uneven weight. Returns ``(image, meta)`` or ``None`` when no handwriting font
    covers ``text``."""
    usable = [f for f in fonts if f.kind == "handwriting" and covers(f, text)]
    if not usable:
        return None
    font = rng.choice(usable)
    paper = rng.choice([(255, 255, 255), (250, 247, 238), (244, 240, 228), (236, 236, 236)])
    ink = rng.choice([(20, 20, 25), (35, 35, 60), (25, 40, 120), (40, 60, 150)])
    size = rng.randint(28, 72)
    image = render_text(text, font.path, size, ink, paper, pad=(rng.randint(8, 30), rng.randint(8, 24)))
    w, h = image.size
    # Wavy baseline: shift each column by a slow sine.
    amp, period, phase = rng.uniform(0, size * 0.08), rng.uniform(1.5, 5) * size, rng.uniform(0, 6.3)
    px = np.asarray(image)
    shifted = np.empty_like(px)
    for x in range(w):
        shifted[:, x] = np.roll(px[:, x], int(round(amp * math.sin(phase + 2 * math.pi * x / period))), axis=0)
    image = Image.fromarray(shifted)
    image = _warp(image, rng, amp=rng.uniform(0.5, size * 0.035), scale=rng.uniform(14, 30), bg=paper)
    # Uneven stroke weight: a heavier or lighter pen.
    r = rng.random()
    if r < 0.3:
        image = image.filter(ImageFilter.MinFilter(3))  # thicker dark strokes
    elif r < 0.4 and size >= 56:
        image = image.filter(ImageFilter.MaxFilter(3))  # thinner strokes
    if rng.random() < 0.3:  # faint ruled lines
        draw = ImageDraw.Draw(image)
        rule = tuple(int(c * 0.92) for c in paper)
        step = rng.randint(int(size * 0.9), int(size * 1.6))
        for y in range(rng.randint(0, step), h, step):
            draw.line((0, y, w, y), fill=rule, width=1)
    if rng.random() < 0.6:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=rng.randint(40, 90))
    buf.seek(0)
    return Image.open(buf).convert("RGB"), {"font": font.path.name, "kind": "handwriting", "style": "handwriting",
                                            "italic": False}
