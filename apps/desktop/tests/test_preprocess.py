import numpy as np
from PIL import Image, ImageDraw, ImageFont

from khmerclipper.ocr.preprocess import content_crop, ink_mask, line_strips, pixel_values, two_line_cut
from khmerclipper.paths import resource_dir


def bars(size, rows, bg="white", fg="black"):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    for top, bottom in rows:
        draw.rectangle((20, top, size[0] - 20, bottom), fill=fg)  # inclusive rows top..bottom
    return img


def test_ink_mask_none_for_degenerate_image():
    assert ink_mask(Image.new("RGB", (1, 50))) is None


def test_ink_mask_downscales_long_edge_to_scan_max():
    mask = ink_mask(Image.new("RGB", (2000, 100), "white"))
    assert mask.scale == 4
    assert mask.ink.shape == (25, 500)


def test_line_strips_splits_two_bars():
    strips = line_strips(bars((400, 200), [(30, 60), (120, 150)]))
    assert [s.size for s in strips] == [(400, 90), (400, 110)]  # boundary at (60 + 120 + 1) // 2


def test_line_strips_merges_small_mark_into_nearest_line():
    # 4-row dot (< 35% of the 31-row line) sits 5 rows below line 1, 42 above line 2. Mark-sized
    # (40 px wide): a full-width 4 px bar is a divider, which text_mask removes.
    img = bars((400, 200), [(20, 50), (100, 130)])
    ImageDraw.Draw(img).rectangle((100, 55, 139, 58), fill="black")
    strips = line_strips(img)
    assert [s.size for s in strips] == [(400, 79), (400, 121)]


def test_line_strips_single_line_returns_original():
    img = bars((400, 100), [(30, 60)])
    strips = line_strips(img)
    assert len(strips) == 1 and strips[0] is img


def test_line_strips_three_khmer_lines():
    font = ImageFont.truetype(str(resource_dir() / "fonts" / "NotoSansKhmer-Regular.ttf"), 32)
    img = Image.new("RGB", (600, 290), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(["ច្បាប់ភូមិបាល", "ផ្តល់សិទ្ធិអំណាចឲ្យរដ្ឋ", "Hello world"]):
        draw.text((20, 10 + i * 90), line, font=font, fill="black")
    assert len(line_strips(img)) == 3


def test_content_crop_pads_four_percent_around_ink():
    img = bars((400, 200), [])
    ImageDraw.Draw(img).rectangle((100, 80, 299, 119), fill="black")
    assert content_crop(img).size == (216, 42)  # x 92..308, y 79..121


def test_content_crop_works_in_dark_mode():
    img = Image.new("RGB", (400, 200), "black")
    ImageDraw.Draw(img).rectangle((100, 80, 299, 119), fill="white")
    assert content_crop(img).size == (216, 42)


def test_content_crop_keeps_blank_and_sliver_images():
    blank = Image.new("RGB", (400, 200), "white")
    assert content_crop(blank) is blank
    sliver = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(sliver).rectangle((10, 10, 11, 11), fill="black")
    assert content_crop(sliver) is sliver


def test_pixel_values_shape_and_normalization():
    arr = pixel_values(Image.new("RGB", (50, 20), "white"))
    assert arr.shape == (1, 3, 384, 384) and arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]
    assert np.allclose(arr, 1.0)
    rgba = Image.new("RGBA", (50, 20), (0, 0, 0, 255))
    assert np.allclose(pixel_values(rgba), -1.0)


def test_line_strips_splits_touching_lines_at_the_pitch():
    # Mirrors apps/mac TrOCRLineSplitTests.testLineStripsSplitsTouchingLinesAtThePitch: four lines
    # whose subscripts touch the next line's marks (no blank row between them).
    width, pitch, n = 260, 22, 4
    img = Image.new("RGB", (width, n * pitch + 4), "white")
    draw = ImageDraw.Draw(img)
    for i in range(n):
        top = 2 + i * pitch
        draw.rectangle((10, top + 7, width - 10, top + 15), fill="black")
        draw.rectangle((30, top, 60, top + 6), fill="black")
        draw.rectangle((90, top + 15, 120, top + pitch + 1), fill="black")
        draw.rectangle((150, top + 15, 151, top + pitch + 8), fill="black")
    strips = line_strips(img)
    assert len(strips) == 4
    assert sum(s.height for s in strips) == img.height


# Mirrors apps/mac TrOCRWideSplitTests and training/eval/tests/test_pipeline.py.
def wide(size, boxes, fg="black", bg="white"):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, fill=fg)
    return img


def test_split_wide_gap_and_join_rule():
    from khmerclipper.ocr.preprocess import split_wide

    assert [s for s, _ in split_wide(wide((400, 24), [(0, 2, 189, 21), (210, 2, 399, 21)]), 10)] == ["", " "]
    assert [s for s, _ in split_wide(wide((400, 24), [(0, 2, 198, 21), (201, 2, 399, 21)]), 10)] == ["", ""]
    assert len(split_wide(wide((400, 24), [(0, 2, 399, 21)]), 10)) == 1
    dark = wide((400, 24), [(0, 2, 189, 21), (210, 2, 399, 21)], fg=(230, 230, 230), bg=(25, 25, 25))
    assert len(split_wide(dark, 10)) == 2


def test_split_wide_ignores_blank_tail_and_neighbour_marks():
    from khmerclipper.ocr.preprocess import split_wide

    assert len(split_wide(wide((400, 24), [(0, 6, 69, 18), (80, 6, 150, 18)]), 10)) == 1
    pieces = split_wide(wide((600, 24), [(0, 6, 140, 18), (160, 6, 300, 18), (420, 0, 440, 2), (520, 21, 540, 23)]), 8)
    assert len(pieces) == 2 and all(p.width <= 320 for _, p in pieces)


def test_two_line_cut_splits_a_long_and_a_short_line_but_not_one_line():
    # A full line, a 2-row bridge where marks touch, then a short wrapped line.
    two = np.array([100.0] * 10 + [5.0] * 2 + [40.0] * 10)
    assert two_line_cut(two) == 10
    # One line: sparse above-marks, a dense core across the middle, sparse subscripts.
    one = np.array([10.0] * 6 + [100.0] * 10 + [10.0] * 6)
    assert two_line_cut(one) is None


def gradient_bars(size, rows):
    """Black bars on a left-to-right blue-to-orange gradient: the corner colour marks most of it as ink."""
    w, h = size
    t = np.linspace(0, 1, w)[None, :, None]
    px = (np.array([40, 40, 200]) * (1 - t) + np.array([240, 200, 40]) * t).repeat(h, axis=0)
    img = Image.fromarray(px.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for top, bottom in rows:
        draw.rectangle((20, top, w - 20, bottom), fill="black")
    return img


def test_line_strips_finds_lines_on_a_gradient():
    img = gradient_bars((400, 200), [(30, 60), (120, 150)])
    assert ink_mask(img).busy
    assert [s.size for s in line_strips(img)] == [(400, 90), (400, 110)]


def test_line_strips_drops_a_texture_only_run_on_a_busy_clip():
    img = gradient_bars((400, 200), [(120, 150)])
    rng = np.random.default_rng(0)
    for x, y in zip(rng.integers(20, 380, 200), rng.integers(30, 76, 200)):  # 2 px grain over a line-tall band
        ImageDraw.Draw(img).rectangle((int(x), int(y), int(x) + 1, int(y) + 1), fill="black")
    strips = line_strips(img)
    assert len(strips) == 1 and strips[0].height < 200  # the grain's own strip is gone


def test_band_varies_on_a_gradient_not_on_a_flat_clip_with_edge_glyphs():
    from khmerclipper.ocr.preprocess import band_varies

    flat = np.full((64, 100, 3), 245)
    flat[0:30, 0:12] = 20  # a glyph touching the top-left corner (printed-031)
    assert not band_varies(flat)
    t = np.linspace(0, 1, 100)[None, :, None]
    assert band_varies((np.array([235, 235, 245]) * (1 - t) + np.array([150, 160, 215]) * t).repeat(64, axis=0).astype(int))
