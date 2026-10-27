"""Noise-robust preprocessing (spec 2026-10-23), mirrored from training/eval/tests/test_noise.py."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from khmerclipper.ocr import preprocess as P
from khmerclipper.paths import resource_dir


def draw_line(text, fg=(0, 0, 0), bg=(255, 255, 255), size=28):
    font = ImageFont.truetype(str(resource_dir() / "fonts" / "NotoSansKhmer-Regular.ttf"), size)
    img = Image.new("RGB", (int(font.getlength(text)) + 24, size * 2), bg)
    ImageDraw.Draw(img).text((12, size // 3), text, font=font, fill=fg)
    return img


def test_background_ignores_an_icon_in_a_corner():
    px = np.full((40, 200, 3), 250)
    px[:12, :12] = (200, 30, 30)
    assert tuple(P.background_color(px)) == (250, 250, 250)


def test_background_when_light_text_touches_much_of_the_border():
    px = np.full((30, 100, 3), 20)
    px[0, :40] = px[-1, 10:50] = 230
    assert tuple(P.background_color(px)) == (20, 20, 20)


def test_ink_mask_keeps_the_page_clean_when_an_icon_covers_a_corner():
    img = draw_line("ការអប់រំ")
    ImageDraw.Draw(img).rectangle((0, 0, 14, 14), fill=(200, 30, 30))
    mask = P.ink_mask(img, scan_max=max(img.size)).ink
    assert not mask[-1, -1] and mask.mean() < 0.3


def test_ink_mask_keeps_the_corner_estimate_when_corners_agree_with_the_edge():
    rng = np.random.default_rng(0)
    px = np.clip(rng.normal(232, 3, (64, 100, 3)), 0, 255).astype(np.uint8)
    px[0, 0] = px[0, -1] = px[-1, 0] = px[-1, -1] = 219
    px[20:40, 30:70] = 30
    px[19, 30:70] = px[40, 30:70] = 185  # ink against 232, not against the 219 corner mean
    mask = P.ink_mask(Image.fromarray(px), scan_max=100).ink
    assert not mask[19, 30:70].any()


def test_text_mask_drops_lines_and_blocks_but_keeps_glyphs():
    m = np.zeros((60, 300), bool)
    for x in range(10, 200, 9):
        m[20:30, x:x + 5] = True
    m[16:18, 20:22] = True
    m[45, 0:300] = True
    m[5:55, 250:290] = True
    kept, removed = P.text_mask(m)
    assert kept[20:30, 10:15].all() and kept[16:18, 20:22].all()
    assert removed[45, :].all() and removed[5:55, 250:290].all()


def test_text_mask_keeps_tiny_1x_glyphs():
    m = np.zeros((12, 80), bool)
    for x in range(2, 78, 5):
        m[3:8, x:x + 3] = True
    assert P.text_mask(m)[0].sum() == m.sum()


def test_text_mask_keeps_short_thin_strokes_like_khan_and_equals():
    m = np.zeros((50, 200), bool)
    for x in range(10, 120, 12):
        m[15:35, x:x + 7] = True
    m[14:36, 130:132] = True
    m[22:24, 145:160] = m[27:29, 145:160] = True
    kept, _ = P.text_mask(m)
    assert kept[14:36, 130:132].all() and kept[22:24, 145:160].all()


def test_text_mask_keeps_a_small_mark_set_apart_from_its_word():
    m = np.zeros((40, 120), bool)
    for x in (10, 22, 34):
        m[15:30, x:x + 8] = True
    m[8:12, 100:103] = True
    assert P.text_mask(m)[0][8:12, 100:103].all()


def test_text_mask_keeps_a_dense_blob_touching_the_clip_edge():
    m = np.zeros((60, 300), bool)
    for x in range(40, 280, 12):
        m[20:32, x:x + 7] = True
    m[0:50, 0:46] = True
    assert P.text_mask(m)[0][20:32, 40:47].all()


def test_text_mask_leaves_a_mask_that_is_mostly_ink_alone():
    m = np.ones((40, 200), bool)
    m[:, ::7] = False
    kept, removed = P.text_mask(m)
    assert (kept == m).all() and not removed.any()


def test_line_strips_ignores_a_2x_box_border():
    a, b = draw_line("ការអប់រំ", size=56), draw_line("បច្ចេកវិទ្យា", size=56)
    img = Image.new("RGB", (max(a.width, b.width) + 20, a.height + b.height + 20), "white")
    img.paste(a, (10, 10))
    img.paste(b, (10, 10 + a.height))
    ImageDraw.Draw(img).rounded_rectangle((4, 4, img.width - 5, img.height - 5), radius=12, outline="gray", width=4)
    kept, _ = P.text_mask(P.ink_mask(img, scan_max=max(img.size)).ink)
    assert not kept[4:8, 30:-30].any() and not kept[-8:-4, 30:-30].any()
    assert len(P.line_strips(img)) == 2


def test_edge_fragments():
    assert P.drop_edge_fragments([[0, 4], [10, 30], [40, 60], [70, 79]], 80) == [[10, 30], [40, 60]]
    assert P.drop_edge_fragments([[0, 22]], 23) == [[0, 22]]
    assert P.drop_edge_fragments([[0, 20], [30, 50]], 51) == [[0, 20], [30, 50]]
    # Every run on an edge: judged against the tallest (noisy-edgecut-029), the 5-row sliver goes.
    assert P.drop_edge_fragments([[0, 4], [70, 79]], 80) == [[70, 79]]


def test_line_strips_drops_a_cut_line_at_the_top():
    line = draw_line("ការអប់រំ និងបច្ចេកវិទ្យា")
    w, h = line.size
    img = Image.new("RGB", (w, h * 2 - h // 2), "white")
    img.paste(line.crop((0, h // 2, w, h)), (0, 0))
    img.paste(line, (0, h // 2))
    strips = P.line_strips(img)
    assert len(strips) == 1 and strips[0].height < img.height


def test_clean_line_is_unchanged():
    img = draw_line("ការអប់រំ និងបច្ចេកវិទ្យា")
    assert len(P.line_strips(img)) == 1 and P.line_strips(img)[0].size == img.size


def test_text_mask_keeps_a_lone_digit_away_from_other_text():
    m = np.zeros((110, 300), bool)
    for y0 in (5, 30):
        for x in range(10, 280, 12):
            m[y0:y0 + 14, x:x + 7] = True
    m[90:104, 140:148] = True
    assert P.text_mask(m)[0][90:104, 140:148].all()


def test_edge_fragment_needs_a_sliced_edge_row():
    keep = np.zeros(80)
    keep[0:20] = keep[30:50] = 100.0
    keep[60:71], keep[71:80], keep[79] = 90.0, 40.0, 8.0
    assert P.drop_edge_fragments([[0, 19], [30, 49], [60, 79]], 80, keep) == [[0, 19], [30, 49], [60, 79]]
    cut = np.zeros(80)
    cut[0:5] = 60.0
    cut[10:31] = cut[40:61] = 100.0
    assert P.drop_edge_fragments([[0, 4], [10, 30], [40, 60]], 80, cut) == [[10, 30], [40, 60]]


def test_band_background_needs_to_explain_more_of_the_clip():
    # Review I1: on a noisy dark photo the background spreads over many colour bins, so flat light
    # text touching the edges wins the band's mode. That colour makes the photo "ink": keep the corners.
    px = np.clip(np.random.default_rng(0).normal(40, 12, (60, 200, 3)), 0, 255).astype(int)
    for x in range(10, 190):
        if (x // 10) % 5 in (0, 1):
            px[:, x] = 250  # tall light strokes crossing the top and bottom edge band
    corner = np.stack([px[0, 0], px[0, -1], px[-1, 0], px[-1, -1]]).sum(axis=0) // 4
    assert np.abs(P.background_color(px) - corner).max() > 40  # the band mode is the text colour
    ink = lambda bg: int((np.abs(px - bg) > 40).any(axis=2).sum())
    assert ink(P.clip_background(px)) <= ink(corner)
