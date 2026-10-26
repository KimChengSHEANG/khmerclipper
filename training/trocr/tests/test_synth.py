import numpy as np
import random
from pathlib import Path

import pytest
from PIL import Image, ImageFont, features

from trocr_khmer import synth
from trocr_khmer.synth import (
    COENG,
    build_corpus,
    chunk_line,
    corpus_part,
    covers,
    degrade_document,
    font_info,
    render_sample,
    render_text,
    shear,
)

NOTO = Path(__file__).resolve().parents[2] / "eval/fonts/NotoSansKhmer-Regular.ttf"
CLUSTER = "ស្ត្រ"  # s + coeng t + coeng r: subscripts stack only when shaped
# Rendering needs libraqm (render_text fails closed without it). Skip, don't fail,
# on dev machines whose Pillow can't load it; the render host must have it.
needs_raqm = pytest.mark.skipif(not features.check("raqm"), reason="Pillow lacks raqm: wheels need libfribidi (brew install fribidi)")


@needs_raqm
def test_raqm_shapes_coeng_cluster():
    raqm = ImageFont.truetype(str(NOTO), 48, layout_engine=ImageFont.Layout.RAQM).getbbox(CLUSTER)
    basic = ImageFont.truetype(str(NOTO), 48, layout_engine=ImageFont.Layout.BASIC).getbbox(CLUSTER)
    assert raqm != basic  # shaping repositioned the subscripts


@needs_raqm
def test_render_text_is_rgb_with_padding():
    image = render_text(CLUSTER, NOTO, 40, fg=(0, 0, 0), bg=(255, 255, 255), pad=(10, 5))
    assert image.mode == "RGB" and image.width > 20 and image.height > 10
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_shear_widens_canvas_and_keeps_corners_background():
    image = Image.new("RGB", (100, 40), (255, 255, 255))
    out = shear(image, 0.25, (255, 255, 255))
    assert out.size == (110, 40)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((109, 39)) == (255, 255, 255)


@needs_raqm
def test_degrade_document_keeps_size_and_mode():
    image = render_text(CLUSTER, NOTO, 40, fg=(0, 0, 0), bg=(250, 250, 245))
    out = degrade_document(image, random.Random(0))
    assert out.size == image.size and out.mode == "RGB"


@needs_raqm
def test_render_sample_is_deterministic():
    fonts = [font_info(NOTO, "text")]
    a, meta_a = render_sample("ភាសាខ្មែរ ២០២៤", fonts, random.Random(7))
    b, meta_b = render_sample("ភាសាខ្មែរ ២០២៤", fonts, random.Random(7))
    assert meta_a == meta_b and a.tobytes() == b.tobytes()
    assert set(meta_a) == {"font", "kind", "style", "italic"}


def test_render_sample_skips_fonts_missing_glyphs():
    fonts = [font_info(NOTO, "text")]
    assert not covers(fonts[0], "ក\U0001F600")
    assert render_sample("ក\U0001F600", fonts, random.Random(0)) is None


def test_chunk_line_respects_limit_and_keeps_coeng_attached():
    text = "ព្រះរាជាណាចក្រកម្ពុជាជាតិសាសនាព្រះមហាក្សត្រ​សួស្តី world"
    chunks = chunk_line(text, max_chars=12)
    assert chunks and all(len(c) <= 12 for c in chunks)
    assert all(not c.endswith(COENG) and not c.startswith(COENG) for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace("​", "").replace(" ", "")


def test_build_corpus_excludes_eval_text():
    fixture = "រក្សាទុកជា"
    corpus = build_corpus(["សួស្តី", fixture, f"ចុច {fixture} ឥឡូវ", "សួស្តី"], exclude=[fixture], max_chars=32)
    assert corpus == ["សួស្តី"]


def test_corpus_part_is_stable_and_mostly_train():
    parts = [corpus_part(f"line {i}") for i in range(2000)]
    assert parts == [corpus_part(f"line {i}") for i in range(2000)]
    assert 0 < parts.count("val") < 100


def test_is_latin_only():
    assert synth.is_latin_only("Battery health 100%")
    assert not synth.is_latin_only("ដក language")
    assert not synth.is_latin_only("100%")


def test_augment_case_only_changes_case():
    rng = random.Random(0)
    outs = {synth.augment_case("battery health 100%", rng) for _ in range(200)}
    assert "Battery health 100%" in outs and "battery health 100%" in outs
    assert all(o.lower() == "battery health 100%" for o in outs)


def test_latin_fonts_listed_for_fetch_and_khmer_text_never_uses_them():
    assert set(synth.GOOGLE_LATIN_FONTS) <= set(synth.ALL_FONTS)
    assert all(kind in synth.KIND_WEIGHTS for kind in synth.ALL_FONTS.values())


@needs_raqm
def test_render_screen_sample_is_a_deterministic_strip():
    fonts = [font_info(NOTO, "text")]
    args = ("ភាសាខ្មែរ ២០២៤", ("ស្ត្រីម្នាក់", "ក្រុមហ៊ុន"), fonts)
    a, meta_a = synth.render_screen_sample(*args, random.Random(3))
    b, meta_b = synth.render_screen_sample(*args, random.Random(3))
    assert a.tobytes() == b.tobytes() and meta_a == meta_b
    assert set(meta_a) == {"font", "kind", "style", "italic"} and meta_a["style"].startswith("screen")
    alone, _ = synth.render_screen_sample("ភាសាខ្មែរ ២០២៤", (None, None), fonts, random.Random(3))
    # A strip holds one line: never as tall as two line pitches.
    assert a.height < 2 * alone.height + 8


@needs_raqm
def test_render_screen_sample_style_knobs():
    fonts = [font_info(NOTO, "text")]
    args = ("ភាសាខ្មែរ ២០២៤", (None, None), fonts)
    # Default knobs draw exactly what v3 drew (no extra rng draws).
    plain, _ = synth.render_screen_sample(*args, random.Random(5))
    same, _ = synth.render_screen_sample(*args, random.Random(5), sizes=(11, 24), dark=0.3, bold=0.0)
    assert plain.tobytes() == same.tobytes()
    # Forced dark, small and bold: dark style, a short strip, heavier ink than the same strip thin.
    _, meta = synth.render_screen_sample(*args, random.Random(5), sizes=(11, 11), dark=1.0)
    assert meta["style"] == "screen_dark"
    thin, _ = synth.render_screen_sample(*args, random.Random(7), sizes=(20, 20), dark=0.0)
    bold, _ = synth.render_screen_sample(*args, random.Random(7), sizes=(20, 20), dark=0.0, bold=1.0)
    ink = lambda im: int((np.asarray(im.convert("L")) < 128).sum())
    assert ink(bold) > ink(thin)


def test_render_screen_sample_skips_uncovered_text():
    assert synth.render_screen_sample("ក\U0001F600", (None, None), [font_info(NOTO, "text")], random.Random(0)) is None
