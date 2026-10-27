"""TrOCR preprocessing, ported from apps/mac/Sources/OCR/TrOCREngine.swift.

Keep the constants and integer arithmetic identical to the Swift version so both apps
split lines and crop the same way.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

SCAN_MAX = 512
THRESHOLD = 40          # per-channel difference from background that counts as ink
MERGE_PERCENT = 35      # runs shorter than this % of the tallest merge into a neighbour
SIZE = 384              # TrOCR ViT input


@dataclass
class InkMask:
    ink: np.ndarray  # bool, shape (h, w)
    scale: int       # mask coords * scale = full-res coords
    busy: bool = False  # over BUSY_INK ink: measured locally; letter_runs compares with the tallest run


def background_color(px: np.ndarray) -> np.ndarray:
    """Background RGB of a clip: the most common colour (each channel binned to 16 levels,
    averaged within the bin) in the band around its edge, min(h, w) // 8 px deep. Unlike the
    four-corner mean it survives an icon or avatar in a corner, a box border drawn on the edge
    (the band is deeper than the border) and glyphs touching the edge of a tight crop; a solid
    image block inside the clip never enters the band."""
    h, w = px.shape[:2]
    d = max(1, min(h, w) // 8)
    band = np.concatenate([px[:d].reshape(-1, 3), px[-d:].reshape(-1, 3),
                           px[d:-d, :d].reshape(-1, 3), px[d:-d, -d:].reshape(-1, 3)])
    bins = band // 16
    keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    top = np.bincount(keys).argmax()
    return band[keys == top].mean(axis=0).round().astype(int)


def ink_mask(image: Image.Image, scan_max: int = SCAN_MAX) -> InkMask | None:
    w, h = image.size
    if w <= 1 or h <= 1:
        return None
    scale = max(1, -(-max(w, h) // scan_max))
    sw, sh = w // scale, h // scale
    if sw <= 1 or sh <= 1:
        return None
    px = np.asarray(image.convert("RGB").resize((sw, sh), Image.BILINEAR), dtype=np.int32)
    bg = clip_background(px)
    ink = (np.abs(px - bg) > THRESHOLD).any(axis=2)
    if ink.mean() > BUSY_INK or (ink.mean() > LOCAL_INK and band_varies(px)):
        return InkMask(local_ink_mask(px, ink), scale, busy=bool(ink.mean() > BUSY_INK))
    return InkMask(ink, scale)


def band_varies(px: np.ndarray, share: float = 0.75) -> bool:
    """True when under `share` of the clip's edge band (as in background_color) sits within the
    ink threshold of the band's most common colour: a gradient or photo panel, not a flat
    background with glyphs touching the edge."""
    h, w = px.shape[:2]
    d = max(1, min(h, w) // 8)
    band = np.concatenate([px[:d].reshape(-1, 3), px[-d:].reshape(-1, 3),
                           px[d:-d, :d].reshape(-1, 3), px[d:-d, -d:].reshape(-1, 3)])
    return float((np.abs(band - background_color(px)) <= THRESHOLD).all(axis=1).mean()) < share


BUSY_INK = 0.3      # share of "ink" past which one background colour can't describe the clip
LOCAL_INK = 0.15    # ... or past this share when the edge band also varies (band_varies)
LOCAL_RADIUS = 0.5  # local background window radius, x the clip's short side


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)^2 window (clipped at the edges), via summed-area tables."""
    h, w = a.shape[:2]
    c = np.pad(a, ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2)).cumsum(0).cumsum(1)
    y0, y1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    return c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]


def local_ink_mask(px: np.ndarray, mask: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Ink against a smooth per-pixel background for busy clips (gradients, photos, textures),
    where one colour marks most of the clip as ink and lines merge or vanish: the window mean of
    the pixels not currently marked ink, re-estimated a few rounds. It starts from the pixels the
    single-colour `mask` calls background, so dense clean text keeps exactly its mask; windows
    with none of them start from the mean of everything. ponytail: a card inside a margin of
    another colour stays all ink (real-003); starting from the plain mean finds its lines, but
    merges solid shapes on clean clips, so that needs a test that tells the two apart."""
    r = max(2, int(min(px.shape[:2]) * LOCAL_RADIUS))
    f = px.astype(np.float64)
    bg = box_mean(f, r) / box_mean(np.ones(px.shape[:2] + (1,)), r)
    for _ in range(rounds + 1):
        keep = (~mask)[..., None].astype(np.float64)
        n = box_mean(keep, r)
        bg = np.where(n > 0.5, box_mean(f * keep, r) / np.maximum(n, 1e-9), bg)
        mask = (np.abs(f - bg) > THRESHOLD).any(axis=2)
    return mask


SPECKLE = 0.35  # a run whose tallest component is below this x the reference run's is not text


def letter_runs(mask: np.ndarray, runs: list[list[int]], busy: bool) -> list[list[int]]:
    """Runs that hold a letter-sized component, measured against the tallest run on busy clips
    and the median run otherwise (a blob of leaked background can be the tallest there). Paper
    grain, wood texture, a button border or a divider can form a run of their own, which the
    model fills with invented words (eval: noisy_chrome -5.5%, results/busy_background.md)."""
    _, boxes = components(mask)
    tall = [max([b[3] - b[1] + 1 for b in boxes if b[1] >= lo - 2 and b[3] <= hi + 2], default=0) for lo, hi in runs]
    ref = max(tall) if busy else float(np.median(tall))
    return [r for r, t in zip(runs, tall) if t >= SPECKLE * ref]


def clip_background(px: np.ndarray, robust_bg: bool = True) -> np.ndarray:
    """The background `ink_mask` measures ink against: the mean corner colour, overruled with
    robust_bg by `background_color` when the two disagree by more than the ink threshold (a
    corner sits on an icon, avatar or border). Clean and scanned clips, where they differ by a
    few levels, keep the corner estimate."""
    bg = np.stack([px[0, 0], px[0, -1], px[-1, 0], px[-1, -1]]).sum(axis=0) // 4
    if robust_bg:
        band = background_color(px)
        # ...and only when it explains more of the clip as background: on a noisy photo or
        # gradient, flat bold text can win the band's colour mode instead (review I1).
        if np.abs(band - bg).max() > THRESHOLD and \
                (np.abs(px - band) > THRESHOLD).any(axis=2).sum() < (np.abs(px - bg) > THRESHOLD).any(axis=2).sum():
            bg = band
    return bg



PITCH_STRENGTH = 0.1   # autocorrelation peak a line pitch needs
TOUCHING_VALLEY = 0.2  # a cut between touching lines: this sparse vs the run's densest row


def line_pitch(profile: np.ndarray) -> int | None:
    """Line pitch in rows: first autocorrelation peak of the row-ink profile above
    PITCH_STRENGTH, searched between n/12 and n/1.5 rows; None for a single line."""
    n = len(profile)
    x = profile - profile.mean()
    ac = np.correlate(x, x, "full")[n - 1:]
    if n < 8 or ac[0] <= 0:
        return None
    ac = ac / ac[0]
    for k in range(max(4, n // 12), min(n - 1, int(n / 1.5) + 1)):
        if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > PITCH_STRENGTH:
            return k
    return None


def touching_cuts(profile: np.ndarray, pitch: int) -> list[int]:
    """Rows to cut a run holding several touching lines: the sparsest row near each expected
    boundary, kept only when <= TOUCHING_VALLEY x the run's densest row."""
    h = len(profile)
    n = int((h - 0.5 * pitch) // pitch) + 1
    if n < 2:
        return []
    line_h = h - (n - 1) * pitch
    cuts = []
    for i in range(1, n):
        centre = (i - 1) * pitch + (pitch + line_h) / 2
        a, b = int(max(1, centre - pitch / 3)), int(min(h - 1, centre + pitch / 3 + 1))
        if a < b:
            c = a + int(np.argmin(profile[a:b]))
            if profile[c] <= TOUCHING_VALLEY * profile.max():
                cuts.append(c)
    return cuts


TWO_LINE_VALLEY = 0.15  # no pitch found: a mid-run cut must be this sparse vs the lighter line
TWO_LINE_BAND = 0.3     # ... and the lighter line's densest row this dense vs the run's


def two_line_cut(profile: np.ndarray) -> int | None:
    """Row to cut a run holding two touching lines when line_pitch finds no pitch (two lines,
    the second often short, are too little repetition for the autocorrelation): the sparsest
    row in the middle 35-65%, kept only when both sides carry a real line of ink."""
    h = len(profile)
    a, b = max(1, int(h * 0.35)), min(h - 1, int(h * 0.65) + 1)
    if h < 9 or b - a < 2:
        return None
    c = a + int(np.argmin(profile[a:b]))
    weak = min(profile[:c].max(), profile[c + 1:].max())
    if weak < TWO_LINE_BAND * profile.max() or profile[c] > TWO_LINE_VALLEY * weak:
        return None
    return c


def merge_marks(runs: list[list[int]]) -> list[list[int]]:
    """A run < 35% of the tallest, or < 50% within a quarter of the tallest's height of a
    neighbour, is a detached Khmer mark: merge it into the nearer neighbour.
    ponytail: fixed cutoffs (same as mac); tune if headings next to small body text merge."""
    while len(runs) > 1:
        tallest = max(hi - lo + 1 for lo, hi in runs)

        def gap(k: int) -> float:
            return min(runs[k][0] - runs[k - 1][1] if k > 0 else math.inf,
                       runs[k + 1][0] - runs[k][1] if k < len(runs) - 1 else math.inf)

        i = next((k for k, (lo, hi) in enumerate(runs)
                  if (hi - lo + 1) * 100 < tallest * MERGE_PERCENT
                  or ((hi - lo + 1) * 100 < tallest * 50 and gap(k) * 4 < tallest)), None)
        if i is None:
            break
        above = runs[i][0] - runs[i - 1][1] if i > 0 else math.inf
        below = runs[i + 1][0] - runs[i][1] if i < len(runs) - 1 else math.inf
        j = i - 1 if above <= below else i + 1
        runs[min(i, j)] = [min(runs[i][0], runs[j][0]), max(runs[i][1], runs[j][1])]
        del runs[max(i, j)]
    return runs


# Non-text components (noise-robust spec step 3). ponytail: fixed cutoffs, tuned on the eval sets
# (results/noise_preprocessing.md); same values in apps/desktop preprocess.py and TrOCREngine.swift.
LINE_ASPECT = 8      # a component this many times longer than thick ...
LINE_THICK = 3       # ... and at most max(this, LINE_THICK_REL x median height) mask px thick is a border /
LINE_THICK_REL = 0.2  # underline / grid line (a 2 px CSS border is 4 px at 2x)
BLOCK_FILL = 0.6     # a component filling this share of its box ...
BLOCK_HEIGHT = 1.8   # ... and taller than this x the median component height is an image / avatar
RULE_LENGTH = 3.0    # a <= LINE_THICK-thick ink run this many x the median height long is a rule
MAX_INK = 0.3        # a mask more than this share ink means the background was misread or is a noisy
                     # scan: filter nothing (frozen 200: 23 -> 16 clips losing ink at 0.5 -> 0.3)
MIN_KEPT = 0.3       # filtering that would keep less than this share of the ink is a misread: undo
HEIGHT_PCT = 90      # "typical glyph height" = this percentile of component heights; the median is
                     # dragged down by marks and scan speckle (frozen 200: 50 clips lost > 2% ink at 50, 25 at 90)
def glyph_height(boxes) -> float:
    """Typical glyph height: the HEIGHT_PCT percentile over components at least 3 px tall with 6+ px
    of ink, so scan speckle and dots don't drag it down. 0 when there are none."""
    hs = [b[3] - b[1] + 1 for b in boxes if b[3] - b[1] >= 2 and b[4] >= 6]
    return float(np.percentile(hs, HEIGHT_PCT)) if hs else 0.0




def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Per ink pixel, the length of the horizontal ink run it belongs to (0 off ink)."""
    d = np.diff(np.pad(mask.view(np.int8), ((0, 0), (1, 1))), axis=1)
    lengths = np.nonzero(d.ravel() == -1)[0] - np.nonzero(d.ravel() == 1)[0]  # runs in row-major order
    out = np.zeros(mask.shape, np.int32)
    out[mask] = np.repeat(lengths, lengths)
    return out


def line_thick(med: float) -> int:
    return max(LINE_THICK, round(LINE_THICK_REL * med))


def rules(mask: np.ndarray, med: float) -> np.ndarray:
    """Pixels of long thin horizontal or vertical strokes (borders, frames, underlines, dividers),
    even where they touch glyphs or images."""
    thick = line_thick(med)
    long = max(RULE_LENGTH * med, LINE_ASPECT * thick)
    h_len, v_len = run_lengths(mask), run_lengths(mask.T).T
    return mask & (((h_len >= long) & (v_len <= thick)) | ((v_len >= long) & (h_len <= thick)))


def components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """8-connected components of a bool mask: label image (0 = none, k + 1 = boxes[k]) and boxes
    (x0, y0, x1, y1, area), inclusive. Row runs joined with union-find, no scipy."""
    h, w = mask.shape
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs: list[tuple[int, int, int, int]] = []  # (y, x0, x1, run id)
    prev: list[tuple[int, int, int]] = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        cur = []
        j = 0
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            while j < len(prev) and prev[j][1] < x0 - 1:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] <= x1 + 1:
                a, b = find(prev[k][2]), find(rid)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                k += 1
            cur.append((x0, x1, rid))
            runs.append((y, x0, x1, rid))
        prev = cur
    labels = np.zeros((h, w), np.int32)
    index: dict[int, int] = {}
    boxes: list[list[int]] = []
    for y, x0, x1, rid in runs:
        r = find(rid)
        if r not in index:
            index[r] = len(boxes)
            boxes.append([x0, y, x1, y, 0])
        b = boxes[index[r]]
        b[0], b[1], b[2], b[3] = min(b[0], x0), min(b[1], y), max(b[2], x1), max(b[3], y)
        b[4] += x1 - x0 + 1
        labels[y, x0:x1 + 1] = index[r] + 1
    return labels, [tuple(b) for b in boxes]


def text_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an ink mask into (kept, removed): borders, underlines and grid lines and solid image or
    avatar blocks are removed; glyphs and Khmer marks stay. (A stray-blob rule was dropped: it never
    helped the icon clips and deleted lone digits, review I3/I4.)"""
    kept, ruled, dropped = text_mask_parts(mask)
    return kept, ruled | dropped


def text_mask_parts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`text_mask` with the removed ink split into (kept, ruled, dropped): rule pixels (long thin
    strokes) and dropped components (blocks, short lines)."""
    none = np.zeros_like(mask)
    if mask.mean() > MAX_INK:
        return mask.copy(), none, none
    _, first = components(mask)
    if not first:
        return mask.copy(), none, none
    med0 = glyph_height(first)
    ruled = rules(mask, med0)
    labels, boxes = components(mask & ~ruled)
    if not boxes:
        return mask.copy(), none, none
    drop = set()
    thick = line_thick(med0)
    for k, (x0, y0, x1, y1, area) in enumerate(boxes):
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        # Long as well as thin: "=", "-" and the Khmer full stop "។" are thin but short.
        if min(bw, bh) <= thick and max(bw, bh) >= max(LINE_ASPECT * min(bw, bh), RULE_LENGTH * med0):
            drop.add(k)
    rest = [b for k, b in enumerate(boxes) if k not in drop]
    med = glyph_height(rest)
    h, w = mask.shape

    def on_edge(b) -> bool:  # background leaking in (gradients, textures) reaches the clip edge
        return b[0] == 0 or b[1] == 0 or b[2] == w - 1 or b[3] == h - 1

    for k, (x0, y0, x1, y1, area) in enumerate(boxes):
        if k in drop or not med or on_edge(boxes[k]):
            continue
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if area >= BLOCK_FILL * bw * bh and bh > BLOCK_HEIGHT * med:
            drop.add(k)
    if len(drop) == len(boxes):  # nothing text-like left: keep the clip as it was
        return mask.copy(), none, none
    dropped = np.isin(labels, [k + 1 for k in drop])
    if (mask & ~(ruled | dropped)).sum() < MIN_KEPT * mask.sum():
        return mask.copy(), none, none
    return mask & ~(ruled | dropped), ruled, dropped


EDGE_FRAGMENT = 0.6  # an edge-touching run shorter than this x the median other run is a cut-off line ...
EDGE_CUT = 0.5       # ... if its edge row is this dense vs its densest row (the cut sliced through glyphs)


def drop_edge_fragments(runs: list[list[int]], height: int, profile: np.ndarray | None = None) -> list[list[int]]:
    """Drop the first run when it touches the top edge, and the last when it touches the bottom,
    if it is shorter than EDGE_FRAGMENT x the median height of the runs not on an edge and (given
    the row-ink `profile`) its edge row is dense: what is left of a line the selection sliced
    through. A real short line in a tight crop ends in sparse stroke tips (review I2). Never drops
    the only line or a full-height one."""
    inner = [hi - lo + 1 for k, (lo, hi) in enumerate(runs)
             if not (k == 0 and lo == 0) and not (k == len(runs) - 1 and hi == height - 1)]
    if not inner:
        return runs
    cut = EDGE_FRAGMENT * float(np.median(inner))

    def sliced(lo: int, hi: int, edge: int) -> bool:
        return profile is None or profile[edge] >= EDGE_CUT * profile[lo:hi + 1].max()

    out = list(runs)
    if out[-1][1] == height - 1 and out[-1][1] - out[-1][0] + 1 < cut and sliced(*out[-1], height - 1):
        out.pop()
    if out and out[0][0] == 0 and out[0][1] - out[0][0] + 1 < cut and sliced(*out[0], 0):
        out.pop(0)
    return out or runs


def line_strips(image: Image.Image) -> list[Image.Image]:
    """Split a clip into horizontal text-line strips by row ink projection (mac lineStrips).

    Khmer subscripts often touch the next line's marks, leaving no blank row, so runs are first
    cut at the line pitch on a full-resolution mask, then detached marks are merged in."""
    mask = ink_mask(image, scan_max=max(image.size))
    if mask is None:
        return [image]
    mask = InkMask(text_mask(mask.ink)[0], mask.scale, mask.busy)  # borders, underlines, avatars, stray icons out
    counts = mask.ink.sum(axis=1)
    runs: list[list[int]] = []
    for y, n in enumerate(counts):
        if not n:
            continue
        if runs and runs[-1][1] == y - 1:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    profile = np.convolve(counts.astype(float), np.ones(3) / 3, "same")
    if runs:
        pitch = line_pitch(profile)
        split = []
        for lo, hi in runs:
            seg = profile[lo:hi + 1]
            if pitch:
                cuts = touching_cuts(seg, pitch) if hi - lo + 1 >= 1.5 * pitch else []
            else:
                cuts = [c] if (c := two_line_cut(seg)) is not None else []
            edges = [lo] + [lo + c for c in cuts] + [hi + 1]
            split += [[edges[k], edges[k + 1] - 1] for k in range(len(edges) - 1)]
        runs = split
    runs = merge_marks(runs)
    keep = drop_edge_fragments(runs, mask.ink.shape[0], counts)  # lines the selection sliced through
    if len(runs) > 1:
        letters = letter_runs(mask.ink, runs, mask.busy)
        keep = [r for r in keep if r in letters]
    if len(runs) <= 1:
        return [image]

    s, mh = mask.scale, mask.ink.shape[0]
    strips = []
    for k in range(len(runs)):
        top = 0 if k == 0 else (runs[k - 1][1] + runs[k][0] + 1) // 2
        bottom = mh if k == len(runs) - 1 else (runs[k][1] + runs[k + 1][0] + 1) // 2
        if runs[k] in keep:  # strips are still cut halfway to a dropped fragment
            strips.append(image.crop((0, top * s, image.width, min(image.height, bottom * s))))
    return strips


WIDE_SPLIT_ASPECT = 12.0  # mac TrOCREngine.wideSplitAspect


def split_wide(strip: Image.Image, max_aspect: float = WIDE_SPLIT_ASPECT) -> list[tuple[str, Image.Image]]:
    """Cut a strip whose line is wider than `max_aspect` x the strip height into ~equal pieces
    at blank-column runs (port of mac TrOCREngine.splitWide). The line's extent comes from ink in
    the strip's central band, so blank tails and neighbour-line marks never become pieces. Cuts
    snap to the nearest blank run within a quarter piece width or are skipped. Separator: " " for
    a gap >= a quarter of the text height, else ""."""
    w, h = strip.size
    if w <= max_aspect * h:
        return [("", strip)]
    m = ink_mask(strip, scan_max=max(w, h))
    if m is None:
        return [("", strip)]
    ink = m.ink
    top = int(0.3 * h)
    core = np.nonzero(ink[top:max(top + 1, int(0.7 * h))].any(axis=0))[0]
    if core.size == 0:
        return [("", strip)]
    pad = max(2, h // 5)
    x0, x1 = max(0, int(core[0]) - pad), min(w, int(core[-1]) + 1 + pad)
    if x1 - x0 <= max_aspect * h:
        return [("", strip)]
    rows = np.nonzero(ink.any(axis=1))[0]
    text_h = int(rows[-1] - rows[0] + 1)
    blank = ~ink.any(axis=0)
    gaps, x = [], x0
    while x < x1:
        if blank[x]:
            start = x
            while x < x1 and blank[x]:
                x += 1
            if start > x0 and x < x1:
                gaps.append((start, x))
        else:
            x += 1
    n = math.ceil((x1 - x0) / (max_aspect * h))
    piece_w = (x1 - x0) / n
    cuts: list[tuple[int, str]] = []
    for k in range(1, n):
        ideal = x0 + k * piece_w
        near = [g for g in gaps if abs((g[0] + g[1]) / 2 - ideal) <= 0.25 * piece_w
                and (not cuts or g[0] > cuts[-1][0])]
        if near:
            g = min(near, key=lambda g: abs((g[0] + g[1]) / 2 - ideal))
            cuts.append(((g[0] + g[1]) // 2, " " if g[1] - g[0] >= 0.25 * text_h else ""))
    if not cuts:
        return [("", strip)]
    pieces, left, sep = [], x0, ""
    for cx, next_sep in cuts + [(x1, "")]:
        pieces.append((sep, strip.crop((left, 0, cx, h))))
        left, sep = cx, next_sep
    return pieces


def content_crop(image: Image.Image) -> Image.Image:
    """Trim uniform background margin so the text fills the frame (TrOCR decodes garbage
    past the real text when given a loose crop)."""
    mask = ink_mask(image)
    if mask is None:
        return image
    mask = InkMask(text_mask(mask.ink)[0], mask.scale)
    ys, xs = np.nonzero(mask.ink)
    if xs.size == 0:
        return image
    sh, sw = mask.ink.shape
    min_x, max_x, min_y, max_y = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # A mis-detected background (corner on ink) collapses the box to a sliver: don't trust it.
    if (max_x - min_x + 1) * 20 < sw or (max_y - min_y + 1) * 20 < sh:
        return image
    pad_x = max(1, (max_x - min_x + 1) * 4 // 100)
    pad_y = max(1, (max_y - min_y + 1) * 4 // 100)
    s = mask.scale
    x0, y0 = max(0, (min_x - pad_x) * s), max(0, (min_y - pad_y) * s)
    x1 = min(image.width, (max_x + 1 + pad_x) * s)
    y1 = min(image.height, (max_y + 1 + pad_y) * s)
    return image.crop((x0, y0, x1, y1))


def pixel_values(image: Image.Image) -> np.ndarray:
    """HF ViTImageProcessor equivalent: bilinear resize to 384x384, mean = std = 0.5."""
    img = content_crop(image).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
