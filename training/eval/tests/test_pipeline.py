from PIL import Image, ImageDraw

from khmerclipper_eval.pipeline import blocked_token, line_strips, split_wide


def blocks(size, boxes, fg="black", bg="white"):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    for box in boxes:
        d.rectangle(box, fill=fg)
    return img


def test_blocked_token():
    assert blocked_token([1, 5, 5, 5, 5]) == 5
    assert blocked_token([1, 5, 5, 5]) is None
    assert blocked_token([5, 5, 5, 6, 5]) is None


def test_line_strips_merges_detached_mark():
    # 21px base line with a 4px mark 2px above it: one line, not two.
    img = blocks((200, 60), [(10, 20, 190, 40), (50, 14, 60, 17)])
    assert len(line_strips(img)) == 1


def test_line_strips_three_lines_cover_image():
    img = blocks((200, 120), [(10, 10, 190, 30), (10, 50, 190, 70), (10, 90, 190, 110)])
    strips = line_strips(img)
    assert len(strips) == 3
    assert sum(s.height for s in strips) == 120


def test_split_wide_gap():
    # 400x20 line, blank 20px gap in the middle -> two pieces joined by a space.
    img = blocks((400, 24), [(0, 2, 189, 21), (210, 2, 399, 21)])
    pieces = split_wide(img, max_aspect=10)
    assert [sep for sep, _ in pieces] == ["", " "]
    assert sum(p.width for _, p in pieces) <= 400


def test_split_wide_join_rule_narrow_gap():
    img = blocks((400, 24), [(0, 2, 198, 21), (201, 2, 399, 21)])
    assert [sep for sep, _ in split_wide(img, max_aspect=10)] == ["", ""]


def test_split_wide_no_gap_keeps_whole():
    img = blocks((400, 24), [(0, 2, 399, 21)])
    pieces = split_wide(img, max_aspect=10)
    assert len(pieces) == 1 and pieces[0][1].size == (400, 24)


def test_split_wide_dark():
    img = blocks((400, 24), [(0, 2, 189, 21), (210, 2, 399, 21)], fg=(230, 230, 230), bg=(25, 25, 25))
    assert len(split_wide(img, max_aspect=10)) == 2


def test_split_wide_narrow_strip_untouched():
    img = blocks((100, 24), [(0, 2, 40, 21), (60, 2, 99, 21)])
    assert len(split_wide(img, max_aspect=10)) == 1


def touching_lines(n, pitch=22, core=8, width=260):
    """n text lines whose descender/ascender zones touch through a 2px-wide bridge (no blank row)."""
    img = Image.new("RGB", (width, n * pitch + 4), "white")
    d = ImageDraw.Draw(img)
    for i in range(n):
        top = 2 + i * pitch
        d.rectangle((10, top + 7, width - 10, top + 7 + core), fill="black")  # dense core band
        d.rectangle((30, top, 60, top + 6), fill="black")  # above marks
        d.rectangle((90, top + 7 + core, 120, top + pitch + 1), fill="black")  # subscripts
        d.rectangle((150, top + 7 + core, 151, top + pitch + 8), fill="black")  # bridge to next line
    return img


def test_split_touching_four_lines():
    img = touching_lines(4)
    assert len(line_strips(img)) < 4  # the app's blank-row split merges them
    strips = line_strips(img, split_touching=True)
    assert len(strips) == 4
    assert sum(s.height for s in strips) == img.height


def test_split_touching_keeps_single_line_with_marks():
    img = blocks((200, 60), [(10, 20, 190, 40), (50, 14, 60, 17), (80, 41, 90, 50)])
    assert len(line_strips(img, split_touching=True)) == 1


def test_split_wide_measures_the_line_not_the_blank_tail():
    # A short line in a wide multi-line clip: ink spans 150 of 400 px -> aspect 6.25 < 10, keep whole.
    img = blocks((400, 24), [(0, 6, 69, 18), (80, 6, 150, 18)])
    assert len(split_wide(img, max_aspect=10)) == 1


def test_split_wide_never_emits_a_piece_without_core_ink():
    # Neighbour-line marks at the strip's top edge past the line's end must not become pieces.
    img = blocks((600, 24), [(0, 6, 140, 18), (160, 6, 300, 18), (420, 0, 440, 2), (520, 21, 540, 23)])
    pieces = split_wide(img, max_aspect=8)  # line 0-300 px -> 2 pieces, ideal cut on the gap
    assert len(pieces) == 2
    assert all(p.width <= 320 for _, p in pieces)


def test_cycle_blocked_token():
    from khmerclipper_eval.pipeline import cycle_blocked_token

    assert cycle_blocked_token([0, 7, 9, 7, 9, 7, 9, 7, 9, 7]) == 9  # a 5th "7 9" would follow
    assert cycle_blocked_token([0, 7, 9, 7, 9, 7, 9, 7]) is None  # ". . . " stays legal
    assert cycle_blocked_token([0, 5, 5, 5, 5]) is None  # single-token runs: blocked_token's job


def gradient_bars(rows, grain=False):
    """Black bars on a blue-to-orange gradient (mirrors apps/desktop tests gradient_bars)."""
    import numpy as np

    t = np.linspace(0, 1, 400)[None, :, None]
    px = (np.array([40, 40, 200]) * (1 - t) + np.array([240, 200, 40]) * t).repeat(200, axis=0)
    img = Image.fromarray(px.astype("uint8"))
    draw = ImageDraw.Draw(img)
    for top, bottom in rows:
        draw.rectangle((20, top, 379, bottom), fill="black")
    if grain:
        rng = np.random.default_rng(0)
        for x, y in zip(rng.integers(20, 380, 200), rng.integers(30, 76, 200)):
            draw.rectangle((int(x), int(y), int(x) + 1, int(y) + 1), fill="black")
    return img


def test_local_bg_finds_lines_on_a_gradient():
    kw = dict(split_touching=True, robust_bg=True, drop_noise=True, drop_edge=True)
    img = gradient_bars([(30, 60), (120, 150)])
    assert len(line_strips(img, **kw)) == 1  # one corner colour: the whole gradient is "ink"
    assert [s.size for s in line_strips(img, local_bg=True, **kw)] == [(400, 90), (400, 110)]
    one = line_strips(gradient_bars([(120, 150)], grain=True), local_bg=True, **kw)
    assert len(one) == 1 and one[0].height < 200  # the grain band's own strip is dropped
