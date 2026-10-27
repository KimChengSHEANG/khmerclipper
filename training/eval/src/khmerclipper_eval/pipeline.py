"""End-to-end TrOCR pipeline as the apps run it, in PyTorch: Mac `TrOCREngine.lineStrips` split,
plain 384x384 bilinear resize (training preprocessing, no content crop), greedy decode capped at
128 ids with the repeat-run cap, lines joined by a space (as OCRBench prints them).

Candidate improvements are opt-in keyword flags on `recognize` so each can be gated alone:
  split_wide_at=R  cut strips wider than R x their height at blank-column gaps
  crop=True        desktop's content_crop before resizing
  beam_gate=T      re-decode a line with beam 4 when its least likely greedy token has p < T
  normalize=True   post-decode cleanup (see normalize_text)
  split_touching=True  cut runs of touching lines at the line pitch (line_strips)
  cycle_cap=True   also block a token continuing a 2-token cycle repeated 4 times
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 384
MAX_LENGTH = 128
MAX_RUN = 4
THRESHOLD = 40


def blocked_token(ids: list[int], max_run: int = MAX_RUN) -> int | None:
    """Token that may not extend a run of `max_run` identical ids (mac TrOCREngine.blockedToken)."""
    tail = ids[-max_run:]
    return tail[-1] if len(ids) >= max_run and len(set(tail)) == 1 else None


def cycle_blocked_token(ids: list[int], cycles: int = 4) -> int | None:
    """Token that would continue a 2-token cycle already repeated `cycles` times (greedy decode
    loops on " ។ ។ ។" past a line's end; the single-token repeat cap can't see it)."""
    tail = ids[-2 * cycles:]
    if len(tail) < 2 * cycles or tail[0] == tail[1]:
        return None
    return tail[0] if all(t == tail[i % 2] for i, t in enumerate(tail)) else None


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


def ink_mask(image: Image.Image, scan_max: int | None = 512,
             robust_bg: bool = False, local_bg: bool = False) -> tuple[np.ndarray, int] | None:
    """Bool ink mask (any channel > 40 from the background colour) at <= scan_max long edge.
    Background = mean corner colour; with robust_bg, `background_color` when the corners disagree."""
    w, h = image.size
    if w <= 1 or h <= 1:
        return None
    scale = 1 if scan_max is None else max(1, -(-max(w, h) // scan_max))
    sw, sh = w // scale, h // scale
    if sw <= 1 or sh <= 1:
        return None
    rgb = image.convert("RGB")
    px = np.asarray(rgb if scale == 1 else rgb.resize((sw, sh), Image.BILINEAR), dtype=np.int32)
    bg = clip_background(px, robust_bg)
    mask = (np.abs(px - bg) > THRESHOLD).any(axis=2)
    if local_bg and (mask.mean() > BUSY_INK or (mask.mean() > LOCAL_INK and band_varies(px))):
        mask = local_ink_mask(px, mask)
    return mask, scale


def is_busy(image: Image.Image, scan_max: int | None = 512, robust_bg: bool = False) -> bool:
    """A clip `ink_mask(local_bg=True)` measures against a local background."""
    m = ink_mask(image, scan_max=scan_max, robust_bg=robust_bg)
    return m is not None and m[0].mean() > BUSY_INK


SPECKLE = 0.35  # a busy clip's run whose tallest component is below this x the clip's tallest is texture


def letter_runs(mask: np.ndarray, runs: list[list[int]]) -> list[list[int]]:
    """Runs that hold a letter-sized component. On busy clips, paper grain and wood texture can
    form a run of their own, which the model fills with invented words (printed-087)."""
    _, boxes = components(mask)
    tall = [max([b[3] - b[1] + 1 for b in boxes if b[1] >= lo - 2 and b[3] <= hi + 2], default=0) for lo, hi in runs]
    return [r for r, t in zip(runs, tall) if t >= SPECKLE * max(tall)]


def band_varies(px: np.ndarray, share: float = 0.75) -> bool:
    """True when under `share` of the clip's edge band (as in background_color) sits within the
    ink threshold of the band's most common colour: a gradient or photo panel, not a flat
    background with glyphs touching the edge."""
    h, w = px.shape[:2]
    d = max(1, min(h, w) // 8)
    band = np.concatenate([px[:d].reshape(-1, 3), px[-d:].reshape(-1, 3),
                           px[d:-d, :d].reshape(-1, 3), px[d:-d, -d:].reshape(-1, 3)])
    bg = background_color(px)
    return float((np.abs(band - bg) <= THRESHOLD).all(axis=1).mean()) < share


LOCAL_INK = 0.15    # ... or past this share when the edge band also varies (band_varies)
BUSY_INK = 0.3      # share of "ink" past which one background colour can't describe the clip
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


def clip_background(px: np.ndarray, robust_bg: bool = False) -> np.ndarray:
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


FILL_GROW = 1  # grow the painted area by this many mask px to cover anti-aliased edges


def fill_removed(image: Image.Image, removed: np.ndarray, scale: int, bg: np.ndarray) -> Image.Image:
    """Paint the full-resolution pixels under `removed` (a mask at 1/scale, grown by one mask
    pixel to cover anti-aliasing) with the background colour, so the model never sees the noise."""
    grown = removed.copy()
    for _ in range(FILL_GROW):
        g = grown.copy()
        g[1:] |= grown[:-1]; g[:-1] |= grown[1:]; g[:, 1:] |= grown[:, :-1]; g[:, :-1] |= grown[:, 1:]
        grown = g
    full = np.repeat(np.repeat(grown, scale, axis=0), scale, axis=1)
    px = np.array(image.convert("RGB"))
    h, w = min(px.shape[0], full.shape[0]), min(px.shape[1], full.shape[1])
    px[:h, :w][full[:h, :w]] = np.asarray(bg, dtype=np.uint8)
    return Image.fromarray(px)


PITCH_STRENGTH = 0.1  # autocorrelation peak a line pitch needs
VALLEY = 0.2          # a cut between touching lines must be this sparse vs the run's densest row


def line_pitch(profile: np.ndarray) -> int | None:
    """Line pitch in rows: first autocorrelation peak of the row-ink profile above PITCH_STRENGTH,
    searched between n/12 and n/1.5 rows. None when the clip shows no repeating line structure."""
    x = profile - profile.mean()
    n = len(x)
    ac = np.correlate(x, x, "full")[n - 1:]
    if n < 8 or ac[0] <= 0:
        return None
    ac = ac / ac[0]
    for k in range(max(4, n // 12), min(n - 1, int(n / 1.5) + 1)):
        if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > PITCH_STRENGTH:
            return k
    return None


def touching_cuts(profile: np.ndarray, pitch: int) -> list[int]:
    """Rows to cut a run holding several touching lines: one valley per expected line boundary,
    kept only when it is sparse (<= VALLEY x the run's densest row)."""
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
            if profile[c] <= VALLEY * profile.max():
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
    """Swift lineStrips merge rule: a run < 35% of the tallest, or < 50% within a quarter of the
    tallest's height of a neighbour, is a detached mark and joins its nearer neighbour."""
    while len(runs) > 1:
        tallest = max(hi - lo + 1 for lo, hi in runs)

        def gap(k: int) -> float:
            return min(runs[k][0] - runs[k - 1][1] if k > 0 else math.inf,
                       runs[k + 1][0] - runs[k][1] if k < len(runs) - 1 else math.inf)

        i = next((k for k, (lo, hi) in enumerate(runs)
                  if (hi - lo + 1) * 100 < tallest * 35
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


def line_strips(image: Image.Image, split_touching: bool = False, robust_bg: bool = False,
                drop_noise: bool = False, drop_edge: bool = False, fill_noise: bool = False,
                local_bg: bool = False) -> list[Image.Image]:
    """Port of Swift `TrOCREngine.lineStrips`: runs of inked rows, detached marks merged in,
    strips extended halfway into the gaps.

    split_touching: first cut runs holding several lines whose marks touch (no blank row between
    them, common at Khmer line heights) at the line pitch, on a full-resolution mask. Splitting
    before the mark merge matters: a merged double line otherwise makes real lines look small."""
    m = ink_mask(image, scan_max=None if split_touching else 512, robust_bg=robust_bg, local_bg=local_bg)
    if m is None:
        return [image]
    mask, s = m
    if drop_noise:
        mask, _, removed = text_mask_parts(mask)
        # Paint out dropped components only. Rules stay for the model: painting underlines out
        # cost errors (noisy_underline +3-5%, results/noise_preprocessing.md), it reads them fine.
        if fill_noise and removed.any():
            small = image.convert("RGB")
            if s > 1:
                small = small.resize((image.width // s, image.height // s), Image.BILINEAR)
            image = fill_removed(image, removed, s, clip_background(np.asarray(small, dtype=np.int32), robust_bg))
    runs: list[list[int]] = []
    for y, inked in enumerate(mask.any(axis=1)):
        if inked:
            if runs and runs[-1][1] == y - 1:
                runs[-1][1] = y
            else:
                runs.append([y, y])
    if split_touching and runs:
        profile = np.convolve(mask.sum(axis=1).astype(float), np.ones(3) / 3, "same")
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
    keep = drop_edge_fragments(runs, mask.shape[0], mask.sum(axis=1)) if drop_edge else runs
    if local_bg and len(runs) > 1 and is_busy(image, scan_max=None if split_touching else 512, robust_bg=robust_bg):
        keep = [r for r in keep if r in letter_runs(mask, runs)]
    if len(runs) <= 1:
        return [image]
    mh = mask.shape[0]
    out = []
    for k in range(len(runs)):
        top = 0 if k == 0 else (runs[k - 1][1] + runs[k][0] + 1) // 2
        bottom = mh if k == len(runs) - 1 else (runs[k][1] + runs[k + 1][0] + 1) // 2
        if runs[k] in keep:  # strips are still cut halfway to a dropped fragment
            out.append(image.crop((0, top * s, image.width, min(image.height, bottom * s))))
    return out


def split_wide(strip: Image.Image, max_aspect: float, space_gap: float = 0.25,
               window: float = 0.25, robust_bg: bool = False, local_bg: bool = False) -> list[tuple[str, Image.Image]]:
    """Cut a line strip whose *line* is wider than `max_aspect` x the strip height into ~equal
    pieces at blank-column runs.

    The line's horizontal extent comes from ink in the strip's central band (rows 30-70%): a strip
    spans the whole clip width and its top / bottom edges hold the neighbour lines' marks, and
    a piece cut from that blank tail makes TrOCR hallucinate text. Each ideal cut point snaps to
    the centre of the nearest blank-column run within +-`window` of a piece width; with no blank
    run there, that cut is skipped (never cut ink). Returns (separator, piece): " " when the gap
    is >= `space_gap` x text height, else "".
    """
    w, h = strip.size
    if w <= max_aspect * h:
        return [("", strip)]
    m = ink_mask(strip, scan_max=None, robust_bg=robust_bg, local_bg=local_bg)
    if m is None:
        return [("", strip)]
    mask = m[0]
    core = np.nonzero(mask[int(0.3 * h): max(int(0.3 * h) + 1, int(0.7 * h))].any(axis=0))[0]
    if core.size == 0:
        return [("", strip)]
    pad = max(2, h // 5)
    x0, x1 = max(0, int(core[0]) - pad), min(w, int(core[-1]) + 1 + pad)
    if x1 - x0 <= max_aspect * h:
        return [("", strip)]
    rows = np.nonzero(mask.any(axis=1))[0]
    text_h = int(rows[-1] - rows[0] + 1)
    blank = ~mask.any(axis=0)
    gaps, x = [], x0  # (start, end_exclusive) of blank-column runs inside the line
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
        near = [g for g in gaps if abs((g[0] + g[1]) / 2 - ideal) <= window * piece_w
                and (not cuts or g[0] > cuts[-1][0])]
        if not near:
            continue
        g = min(near, key=lambda g: abs((g[0] + g[1]) / 2 - ideal))
        cuts.append(((g[0] + g[1]) // 2, " " if g[1] - g[0] >= space_gap * text_h else ""))
    if not cuts:
        return [("", strip)]
    pieces, left, sep = [], x0, ""
    for cx, next_sep in cuts:
        pieces.append((sep, strip.crop((left, 0, cx, h))))
        left, sep = cx, next_sep
    pieces.append((sep, strip.crop((left, 0, x1, h))))
    return pieces


def content_crop(image: Image.Image, robust_bg: bool = False, drop_noise: bool = False,
                 local_bg: bool = False) -> Image.Image:
    """Port of desktop `preprocess.content_crop`: trim background margin, 4% padding."""
    m = ink_mask(image, robust_bg=robust_bg, local_bg=local_bg)
    if m is None:
        return image
    mask, s = m
    if drop_noise:
        mask = text_mask(mask)[0]
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return image
    sh, sw = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    if (x1 - x0 + 1) * 20 < sw or (y1 - y0 + 1) * 20 < sh:
        return image
    px, py = max(1, (x1 - x0 + 1) * 4 // 100), max(1, (y1 - y0 + 1) * 4 // 100)
    return image.crop((max(0, (x0 - px) * s), max(0, (y0 - py) * s),
                       min(image.width, (x1 + 1 + px) * s), min(image.height, (y1 + 1 + py) * s)))


def normalize_text(text: str) -> str:
    """Post-decode cleanup; rules are added only when references follow them (accuracy_v3.md)."""
    return " ".join(text.replace("​", "").split())


def pixel_values(images: list[Image.Image]):
    import torch

    arr = np.stack([np.asarray(i.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
                    for i in images])
    return torch.from_numpy((arr / 255.0 - 0.5) / 0.5).permute(0, 3, 1, 2).contiguous()


@lru_cache(maxsize=2)
def load(model_dir: str, device: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "trocr/src"))
    from transformers import VisionEncoderDecoderModel
    from trocr_khmer import load_trocr_processor

    processor = load_trocr_processor(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    return processor, model


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def decode(images: list[Image.Image], model_dir: Path, device: str, beams: int = 1,
           cycle_cap: bool = False) -> list[tuple[str, float]]:
    """(text, min greedy token probability) per image."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class RepeatCap(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for row, ids in enumerate(input_ids.tolist()):
                for b in (blocked_token(ids), cycle_blocked_token(ids) if cycle_cap else None):
                    if b is not None:
                        scores[row, b] = -math.inf
            return scores

    processor, model = load(str(model_dir), device)
    tok = processor.tokenizer
    with torch.no_grad():
        out = model.generate(pixel_values(images).to(device), max_length=MAX_LENGTH, num_beams=beams,
                             do_sample=False, no_repeat_ngram_size=0, length_penalty=1.0, early_stopping=True,
                             logits_processor=LogitsProcessorList([RepeatCap()]),
                             decoder_start_token_id=model.config.decoder_start_token_id,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    if beams > 1:
        return [(t.strip(), 1.0) for t in texts]
    probs = torch.stack([s.float().softmax(-1).max(-1).values for s in out.scores], 1)  # (batch, steps)
    done = (out.sequences[:, 1:] == tok.pad_token_id) & (out.sequences[:, :-1] != model.config.decoder_start_token_id)
    probs = probs.masked_fill(done[:, : probs.shape[1]], 1.0)
    return [(t.strip(), float(p.min())) for t, p in zip(texts, probs)]


def recognize(image: Image.Image, model_dir: Path, device: str | None = None, *, split_wide_at: float | None = None,
              crop: bool = False, beam_gate: float | None = None, normalize: bool = False,
              split_touching: bool = False, cycle_cap: bool = False, robust_bg: bool = False,
              drop_noise: bool = False, drop_edge: bool = False, fill_noise: bool = False,
              local_bg: bool = False) -> str:
    device = device or default_device()
    units: list[tuple[str, Image.Image]] = []  # (separator before, image to decode)
    for k, strip in enumerate(line_strips(image, split_touching, robust_bg=robust_bg, drop_noise=drop_noise,
                                               drop_edge=drop_edge, fill_noise=fill_noise, local_bg=local_bg)):
        # No drop_noise here: noise filtering in split_wide changed no crop on the screen set and
        # doubled preprocessing time (results/noise_preprocessing.md).
        pieces = split_wide(strip, split_wide_at, robust_bg=robust_bg, local_bg=local_bg) if split_wide_at else [("", strip)]
        units += [((" " if k else "") if i == 0 else sep, p) for i, (sep, p) in enumerate(pieces)]
    # With fill_noise the strips are already clean, so the crop needs no second noise pass.
    images = [content_crop(p, robust_bg=robust_bg, drop_noise=drop_noise and not fill_noise, local_bg=local_bg) if crop else p
              for _, p in units]
    results = decode(images, model_dir, device, cycle_cap=cycle_cap)
    if beam_gate is not None:
        redo = [i for i, (_, p) in enumerate(results) if p < beam_gate]
        if redo:
            for i, r in zip(redo, decode([images[i] for i in redo], model_dir, device, beams=4, cycle_cap=cycle_cap)):
                results[i] = r
    text = "".join(sep + t for (sep, _), (t, _) in zip(units, results) if t or sep)
    return normalize_text(text) if normalize else " ".join(text.split())
