from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from khmerclipper_eval import pipeline as P

FONT = Path(__file__).parents[1] / "fonts/NotoSansKhmer-Regular.ttf"


def draw_line(text, fg=(0, 0, 0), bg=(255, 255, 255), size=28):
    font = ImageFont.truetype(str(FONT), size)
    w = int(font.getlength(text)) + 24
    img = Image.new("RGB", (w, size * 2), bg)
    ImageDraw.Draw(img).text((12, size // 3), text, font=font, fill=fg)
    return img


def test_background_ignores_an_icon_in_a_corner():
    px = np.full((40, 200, 3), 250)
    px[:12, :12] = (200, 30, 30)  # red icon top-left
    assert tuple(P.background_color(px)) == (250, 250, 250)


def test_background_when_light_text_touches_much_of_the_border():
    px = np.full((30, 100, 3), 20)  # dark-mode background
    px[0, :40] = px[-1, 10:50] = 230  # tight crop: glyph tops/bottoms on 40% of top and bottom rows
    assert tuple(P.background_color(px)) == (20, 20, 20)


def test_robust_bg_is_identical_on_a_clean_clip():
    img = draw_line("ការអប់រំ")
    assert np.array_equal(P.ink_mask(img, robust_bg=True)[0], P.ink_mask(img)[0])


def test_robust_bg_keeps_the_page_clean_when_an_icon_covers_a_corner():
    img = draw_line("ការអប់រំ")
    ImageDraw.Draw(img).rectangle((0, 0, 14, 14), fill=(200, 30, 30))  # icon on the corner pixel
    mask, _ = P.ink_mask(img, robust_bg=True)
    assert not mask[-1, -1] and mask.mean() < 0.3  # the white page is not ink


def test_text_mask_drops_lines_and_blocks_but_keeps_glyphs():
    m = np.zeros((60, 300), bool)
    for x in range(10, 200, 9):
        m[20:30, x:x + 5] = True  # a line of glyph-sized blobs
    m[16:18, 20:22] = True  # a detached above-mark
    m[45, 0:300] = True  # 1 px divider across the clip
    m[5:55, 250:290] = True  # solid image block
    kept, removed = P.text_mask(m)
    assert kept[20:30, 10:15].all() and kept[16:18, 20:22].all()
    assert removed[45, :].all() and removed[5:55, 250:290].all()
    assert not (kept & removed).any() and ((kept | removed) == m).all()


def test_text_mask_keeps_tiny_1x_glyphs():
    m = np.zeros((12, 80), bool)
    for x in range(2, 78, 5):
        m[3:8, x:x + 3] = True  # 3x5 px glyphs of 10 px text at 1x
    kept, _ = P.text_mask(m)
    assert kept.sum() == m.sum()


def test_text_mask_on_all_noise_keeps_line_strips_working():
    img = Image.new("RGB", (200, 60), "white")
    ImageDraw.Draw(img).rectangle((10, 10, 190, 50), fill="black")
    assert len(P.line_strips(img, split_touching=True, robust_bg=True, drop_noise=True)) == 1


def test_drop_noise_is_identical_on_a_clean_clip():
    img = draw_line("ការអប់រំ និងបច្ចេកវិទ្យា")
    a = P.line_strips(img, split_touching=True, robust_bg=True, drop_noise=True)
    b = P.line_strips(img, split_touching=True, robust_bg=True)
    assert [s.size for s in a] == [s.size for s in b]
    assert P.content_crop(img, robust_bg=True, drop_noise=True).size == P.content_crop(img, robust_bg=True).size


def test_drop_noise_splits_two_lines_joined_by_a_border():
    a, b = draw_line("ការអប់រំ"), draw_line("បច្ចេកវិទ្យា")
    img = Image.new("RGB", (max(a.width, b.width), a.height + b.height), "white")
    img.paste(a, (0, 0)); img.paste(b, (0, a.height))
    ImageDraw.Draw(img).rectangle((0, 0, img.width - 1, img.height - 1), outline="gray", width=2)  # input box
    assert len(P.line_strips(img, split_touching=True, robust_bg=True, drop_noise=True)) == 2


def test_edge_fragment_dropped():
    assert P.drop_edge_fragments([[0, 4], [10, 30], [40, 60], [70, 79]], 80) == [[10, 30], [40, 60]]


def test_single_line_with_marks_touching_top_is_kept():
    assert P.drop_edge_fragments([[0, 22]], 23) == [[0, 22]]


def test_full_height_first_line_touching_edge_is_kept():
    assert P.drop_edge_fragments([[0, 20], [30, 50]], 51) == [[0, 20], [30, 50]]


def test_two_fragments_and_nothing_else_keeps_both():
    assert P.drop_edge_fragments([[0, 4], [70, 79]], 80) == [[0, 4], [70, 79]]  # no real line to compare with
    assert P.drop_edge_fragments([[0, 4], [70, 79]], 80, tallest_ref=True) == [[70, 79]]  # the apps


def test_drop_edge_removes_a_cut_line_from_line_strips():
    line = draw_line("ការអប់រំ និងបច្ចេកវិទ្យា")
    w, h = line.size
    img = Image.new("RGB", (w, h * 2 - h // 2), "white")
    img.paste(line.crop((0, h // 2, w, h)), (0, 0))  # bottom half of a line above, cut by the selection
    img.paste(line, (0, h // 2))
    assert len(P.line_strips(img, split_touching=True, drop_edge=True)) < len(P.line_strips(img, split_touching=True))


def test_text_mask_leaves_a_mask_that_is_mostly_ink_alone():
    m = np.ones((40, 200), bool)  # textured background misread as ink: nothing to judge
    m[:, ::7] = False
    kept, removed = P.text_mask(m)
    assert (kept == m).all() and not removed.any()


def test_text_mask_keeps_a_small_mark_set_apart_from_its_word():
    m = np.zeros((40, 120), bool)
    for x in (10, 22, 34):
        m[15:30, x:x + 8] = True  # three glyphs
    m[8:12, 100:103] = True  # a small sign well to the right (gap 58 > 2 x median 15), above the line
    kept, _ = P.text_mask(m)
    assert kept[8:12, 100:103].all()


def test_drop_noise_removes_a_2x_box_border_4px_thick():
    a, b = draw_line("ការអប់រំ", size=56), draw_line("បច្ចេកវិទ្យា", size=56)  # 2x capture
    img = Image.new("RGB", (max(a.width, b.width) + 20, a.height + b.height + 20), "white")
    img.paste(a, (10, 10)); img.paste(b, (10, 10 + a.height))
    ImageDraw.Draw(img).rounded_rectangle((4, 4, img.width - 5, img.height - 5), radius=12, outline="gray", width=4)
    mask, _ = P.ink_mask(img, scan_max=None, robust_bg=True)
    kept, _ = P.text_mask(mask)
    assert not kept[4:8, 30:-30].any() and not kept[-8:-4, 30:-30].any()  # top and bottom edges gone
    assert len(P.line_strips(img, split_touching=True, robust_bg=True, drop_noise=True)) == 2


def test_robust_bg_keeps_the_corner_estimate_when_corners_agree_with_the_edge():
    rng = np.random.default_rng(0)
    px = np.clip(rng.normal(232, 3, (64, 100, 3)), 0, 255).astype(np.uint8)  # scanned paper
    px[0, 0] = px[0, -1] = px[-1, 0] = px[-1, -1] = 219  # corners a little darker (shadow, foxing)
    px[20:40, 30:70] = 30  # text
    px[19, 30:70] = px[40, 30:70] = 185  # anti-aliased glyph edge: ink against 232, not against 219
    img = Image.fromarray(px)
    assert np.array_equal(P.ink_mask(img, None, robust_bg=True)[0], P.ink_mask(img, None)[0])


def test_text_mask_keeps_short_thin_strokes_like_khan_and_equals():
    m = np.zeros((50, 200), bool)
    for x in range(10, 120, 12):
        m[15:35, x:x + 7] = True  # glyphs, 20 px tall
    m[14:36, 130:132] = True  # Khmer full stop "។": a thin vertical bar about one glyph tall
    m[22:24, 145:160] = m[27:29, 145:160] = True  # "=": two short thin bars
    kept, _ = P.text_mask(m)
    assert kept[14:36, 130:132].all() and kept[22:24, 145:160].all() and kept[27:29, 145:160].all()


def test_text_mask_keeps_a_dense_blob_touching_the_clip_edge():
    m = np.zeros((60, 300), bool)
    for x in range(40, 280, 12):
        m[20:32, x:x + 7] = True  # glyphs
    m[0:50, 0:46] = True  # background patch leaking in from the left edge, merged with the first glyph
    kept, _ = P.text_mask(m)
    assert kept[20:32, 40:47].all()  # the glyph fused to the patch survives


def test_fill_removed_paints_only_removed_pixels():
    img = Image.new("RGB", (40, 20), (255, 255, 255))
    img.putpixel((5, 5), (0, 0, 0))
    img.putpixel((30, 10), (0, 0, 0))
    removed = np.zeros((20, 40), bool)
    removed[10, 30] = True
    out = P.fill_removed(img, removed, 1, np.array([255, 255, 255]))
    assert out.getpixel((30, 10)) == (255, 255, 255) and out.getpixel((5, 5)) == (0, 0, 0)


def test_fill_noise_is_identical_on_a_clean_clip():
    img = draw_line("ការអប់រំ និងបច្ចេកវិទ្យា")
    flags = dict(split_touching=True, robust_bg=True, drop_noise=True)
    assert [s.tobytes() for s in P.line_strips(img, fill_noise=True, **flags)] == \
           [s.tobytes() for s in P.line_strips(img, **flags)]


def test_fill_noise_leaves_an_underline_for_the_model_but_paints_an_avatar():
    line = draw_line("ការអប់រំ និងបច្ចេកវិទ្យា")
    w, h = line.size
    img = Image.new("RGB", (w + 70, h), "white")
    img.paste(line, (70, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 6, 8 + h - 12, h - 6), fill=(90, 120, 200))  # avatar
    d.line((70, h - 6, w + 69, h - 6), fill="black", width=2)  # underline
    strip = P.line_strips(img, split_touching=True, robust_bg=True, drop_noise=True, fill_noise=True)[0]
    px = np.asarray(strip.convert("L"))
    assert px[h // 2, 8 + (h - 12) // 2] > 200  # avatar centre painted with background
    assert px[h - 6, 80:w].min() < 100  # underline still there for the model


def test_robust_bg_does_not_pick_bold_light_text_on_a_noisy_dark_background():
    # Review I1: heavy light text tightly cropped on a dark gradient with per-pixel noise (a photo,
    # the lock screen). The noisy background spreads over many colour bins; the flat text wins the mode.
    font = ImageFont.truetype(str(FONT), 90)
    txt = "ការអប់រំ"
    w, h = int(font.getlength(txt)) + 40, 144
    g = np.zeros((h, w, 3), int)
    for x in range(w):
        g[:, x] = (20 + 100 * x // w, 30 + 60 * x // w, 60 + 80 * x // w)
    g = np.clip(g + np.random.default_rng(0).integers(-12, 12, g.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(g)
    ImageDraw.Draw(img).text((10, 0), txt, font=font, fill=(250, 250, 250), stroke_width=2, stroke_fill=(250, 250, 250))
    ys, xs = np.nonzero(np.asarray(img).min(axis=2) > 200)
    tight = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    assert P.ink_mask(tight, None, robust_bg=True)[0].mean() <= P.ink_mask(tight, None)[0].mean()


def test_text_mask_keeps_a_lone_digit_away_from_other_text():
    m = np.zeros((110, 300), bool)
    for y0 in (5, 30):
        for x in range(10, 280, 12):
            m[y0:y0 + 14, x:x + 7] = True  # two lines of glyphs
    m[90:104, 140:148] = True  # a page number well below (gap 46 > 2 x glyph height)
    assert P.text_mask(m)[0][90:104, 140:148].all()


def test_edge_fragment_keeps_a_short_real_last_line_whose_edge_row_is_sparse():
    # Three lines in a tight crop; the last is x-height only (shorter), its bottom row = sparse stroke ends.
    profile = np.zeros(80)
    profile[0:20] = profile[30:50] = 100.0  # two full lines
    profile[60:71] = 90.0
    profile[71:80] = 40.0
    profile[79] = 8.0  # bottom row: a few stroke ends, not a slice through glyph bodies
    runs = [[0, 19], [30, 49], [60, 79]]
    assert P.drop_edge_fragments(runs, 80, profile) == runs


def test_edge_fragment_dropped_when_its_edge_row_slices_glyphs():
    profile = np.zeros(80)
    profile[0:5] = 60.0  # bottom half of a line above, cut through the glyph bodies at row 0
    profile[10:31] = profile[40:61] = 100.0
    assert P.drop_edge_fragments([[0, 4], [10, 30], [40, 60]], 80, profile) == [[10, 30], [40, 60]]
