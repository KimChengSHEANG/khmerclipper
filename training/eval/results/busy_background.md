# Busy backgrounds: local background for line finding (2026-10-27)

Follow-up to [accuracy_v5.md](accuracy_v5.md): on `noisy_background` the model reads text on photos and gradients
fine when it gets the right line strips (CER 0.005 on the 26 clips split correctly vs 0.651 on the 24 that weren't).
The strips were wrong because `ink_mask` measures ink against one background colour, so on a gradient or photo most
of the clip is "ink": lines merge into one strip, or patches of panel become strips the model fills with words.

## Change (eval `pipeline.py` `local_bg`, desktop `preprocess.py`, Mac `TrOCREngine.swift`)

- A clip is **busy** when the single-colour mask marks more than 30% of it as ink (clean screen text peaks at 27%).
- Busy clips get a **local background**: the mean, over a window of half the clip's short side, of the pixels not
  marked ink, re-estimated for 4 rounds. It starts from the pixels the single-colour mask calls background, so dense
  clean text keeps its mask exactly (starting from the mean of everything merged solid shapes on clean clips).
- On busy clips, a line run whose tallest component is under 0.35x the clip's tallest is **texture** and is dropped
  (wood grain above a line became a strip the model filled with invented Khmer, printed-087).
- Applies wherever `ink_mask` is used: line strips, wide-line split, content crop.

Line counts on `noisy_background` (labels carry the line count): 26 -> 39 of 50 right. Strip counts on every other
set unchanged (frozen 200, screen 450, slices 320, other noisy kinds, real 30). Window radius 0.25-1.0 all gave
38-40; a 25% busy threshold gave 42 but sits too close to clean clips.

## On device: Mac (CoreML `OCRBench`, Latin fallback on), main vs this change

| Set | main errors | errors | change | better / worse | median ms |
|---|---:|---:|---:|---:|---|
| frozen 200 | 11 | 12 | +1 char | 1 / 1 | 134 -> 83 |
| screen 450 | 734 | 734 | 0 | 0 / 0 | 578 -> 316 |
| slices 320 | 165 | 165 | 0 | 0 / 0 | 258 -> 237 |
| **noisy 300** | 3233 | **2397** | **-25.9%** | 17 / 1 | 376 -> 333 |
| - background | 1439 | **603** | **-58.1%** | 17 / 1 | |
| - avatar, chrome, edgecut, icons, underline | | | 0 | 0 / 0 | |
| real 4 | 12 | 12 | 0 | 0 / 0 | |

Latency is no worse anywhere (main was measured earlier on a busier machine, so its times are inflated). The two
frozen changes are scanned printed clips (busy): printed-047 gains a mark, printed-049 reorders one vowel sign after a
slightly different crop.

Parity: desktop strips are byte-identical to the eval pipeline on all 1,300 fixtures; Mac piece counts match desktop
on 1,300 / 1,300 (boxes within 2 px on 1,250, the existing CoreGraphics vs PIL downscaling).

## Not solved

- 11 background clips still got the wrong line count; 3 of those are fixed by the follow-up below, 8 remain.
- A card inside a white margin (real-003, 4 lines) stays one strip: starting from the margin, the whole card is ink.
  Tried (2026-10-27): when the local mask leaves one component over half the clip, restart from the plain window
  mean and keep it if that breaks the panel up. It split real-003 into its 4 lines and changed no other strip
  count, but on device real-003 already read perfectly as one strip, and the frozen set lost a mark (printed-047,
  12 -> 13 errors). Dropped: nothing to gain.

## PyTorch app pipeline (3090), v3, gate5 run vs this change

frozen 35 -> 36 (1 / 1), screen 716 = 716, slices 164 = 164, noisy 3208 -> 2370 (-26.1%; background 1438 -> 600,
-58.3%, 17 better / 1 worse; every other noise kind identical).

## Cut-off lines (noisy_edgecut, CER 0.31): diagnosis, no change

With the right strip count (39 of 50) edge-cut clips still average CER 0.16: a sliver of the cut-off line is
shorter than 35% of the tallest run, so `merge_marks` folds it into the real line as "marks" and the model reads
it as extra words. Running `drop_edge_fragments` before the merge as well (edge_first) trimmed several slivers,
but the right-count tally fell 39 -> 36 (dropping a run changes which runs merge) and 12 of the 30 real captures
re-split, e.g. real-009, where a nearly whole top line and a sliced bottom line frame one clean line: whether to
read them depends on what the user meant to select. Reverted; needs labelled real captures to decide.

## Follow-up: gentle gradients under 30% ink (2026-10-27)

Several missed background clips were never "busy": a two-line clip on a soft gradient is 22-28% ink, so it kept the
single-colour mask and the gradient leaked into it. Lowering the busy threshold to 15% fixed 5 more line counts but
touched clean clips (frozen +4 errors on one clip, noisy_edgecut +3.4%). Gating the lower threshold on the corners
disagreeing still split printed-031 (one word whose glyph touches a corner). The kept rule: below 30% ink, a clip gets
the local background only when over 15% of it is ink **and** under 75% of its edge band (the band `background_color`
reads) is within the ink threshold of the band's colour, i.e. the background itself varies. Texture-only runs are
still dropped only above 30% ink.

Line counts: background 39 -> 42 of 50; every other set's strips unchanged (share 0.85 gave 44 but changed an edgecut
clip). Model, paired against the previous version:

| Run | frozen 200 | screen 450 | slices 320 | noisy background | other noisy kinds | real 4 |
|---|---|---|---|---|---|---|
| PyTorch pipeline (3090) | 36 = 36 | 716 = 716 | 164 = 164 | 600 -> 478 (-20.3%, 4 / 0) | identical | - |
| On device (Mac CoreML) | 12 = 12 | - | - | 603 -> 478 (-20.7%, 4 / 0) | identical | 12 = 12 |

Against main before this work, on device: noisy_background 1439 -> 478 (-66.8%). Parity: desktop strips byte-identical
to eval on 1,300 / 1,300; Mac piece counts equal desktop on 1,300 / 1,300.

## Stray symbols from icons (2026-10-27)

Icons and emoji at a line's ends (noisy_icons) come out as stray symbols ("... ។ > v", "* text &"). Both apps now drop
leading / trailing words made only of `< > * & @ ^ ~ | # `` ` `` + =` on lines with Khmer (never emptying a line, inner
spacing kept). Offline on the saved predictions, wider rules (any ASCII punctuation, bullets, single Latin letters)
gained more on icons but broke screen clips that start with a real "O" or end in quotes; this set touched no clip
outside the icon / chrome / avatar kinds.

On device (Mac), vs the previous commit: noisy_icons 355 -> 296 (-16.6%, 20 / 0), avatar 240 -> 235, chrome 175 -> 171;
frozen, screen, slices, background, edgecut, underline and real identical. (Medians rose on every set, including
the identical ones: the machine was busier during this run.)
