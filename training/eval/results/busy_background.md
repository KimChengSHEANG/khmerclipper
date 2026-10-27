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

## The 8 background clips still miscounted (2026-10-27): no safe rule found

- 038, 039: two tight lines on a warm gradient, 22-28% ink, edge band 81-83% one colour (just past the 0.75 cut-off).
  The local mask opens a blank gap between the lines, but no trigger separates them from scanned paper: the local
  / global ink ratio is 0.54-0.57 here and 0.29-0.63 on frozen printed clips, and share 0.85 changes an edgecut clip.
- 002, 023, 047, 050: busy (local mask on), lines still merged; 050 is 96% ink.
- 034: one Latin word on a texture split in two. 035: faint text on grey (the generator's low-contrast case).
Next lever is probably per-row background (the panels vary mostly along x) or labelled real captures to tell
which of these patterns users actually hit.

## Border-only strips on clean clips (2026-10-27)

noisy_chrome misses were extra strips holding only a button outline or divider piece ("| USB-C | lum", "USB-C ══ ឬ").
The letter-size check now runs on every clip: a run is dropped when its tallest component is under 0.35x the tallest
run's (busy clips) or the median run's (other clips, where a blob of leaked background can be the tallest; with the
tallest as reference noisy-background-032 lost two real lines). Strip counts: chrome 46 -> 48 of 50, real-007 5 -> 3
(its 3 text lines), nothing else changed.

| Run | frozen | screen | slices | noisy chrome | other noisy kinds | real 4 |
|---|---|---|---|---|---|---|
| PyTorch pipeline (3090) | 36 = 36 | 716 = 716 | 164 = 164 | 165 -> 156 (-5.5%, 2 / 0) | identical | - |
| On device (Mac) | 12 = 12 | - | - | 171 -> 162 (-5.3%, 2 / 0) | identical | 12 = 12 |

## Desktop end to end (ONNX, shipped `khmer-v3-kv-int8`), this morning vs all of today's preprocessing

| Set | errors | change | better / worse |
|---|---:|---:|---:|
| frozen 200 | 11 -> 12 | +1 char | 1 / 1 |
| screen 450 | 726 = 726 | 0 | 0 / 0 |
| slices 320 | 163 = 163 | 0 | 0 / 0 |
| noisy 300 | 3231 -> 2197 | **-32.0%** | 47 / 1 |
| - background | 1441 -> 484 | -66.4% | 21 / 1 |
| - icons | 350 -> 291 | -16.9% | 20 / 0 |
| - chrome | 170 -> 157 | -7.6% | 4 / 0 |
| - avatar | 246 -> 241 | -2.0% | 2 / 0 |
| - edgecut, underline | | 0 | 0 / 0 |
| real 4 | 12 = 12 | 0 | 0 / 0 |

## Mark merging cascade and edge-only clips (2026-10-27)

Found on the user's real captures. real-015 (three English lines with a toolbar widget 40 rows above) came out as one
strip and read as "the the the" plus Khmer digits: `merge_marks` merged the 10-row widget into the first line as a
mark, the merged run spanned the 40-row gap, so the "tallest" run grew to 80 rows and every real 29-row line then
looked like a mark (< 35% of the tallest) until the whole clip was one strip.

- `merge_marks` measures "tallest" in inked rows only (the gap inside a merged run no longer counts), capped at 2x
  the tallest run before merging. Measuring once before any merge also fixed it (and read the two-column real-008
  well) but broke two large Khmer lines in a box border, where a line's marks grow it to ~1.6x its base glyphs; caps
  of 1.0-1.5 broke that test too, so 2.0 it is, and real-008 (columns) stays open.
- That exposed noisy-edgecut-029: a sliver merged into the top line, a cut line at the bottom, every run on an edge,
  so `drop_edge_fragments` had no inner run to compare with and dropped nothing. Now edge runs are judged against the
  tallest run when no run is clear of the edges.

Strip counts: edgecut 39 -> 46 of 50; every other synthetic set unchanged.

| Run | frozen | screen | slices | noisy edgecut | other noisy kinds | real 4 |
|---|---|---|---|---|---|---|
| PyTorch pipeline (3090) | 36 = 36 | 716 = 716 | 164 = 164 | 880 -> 501 (-43.1%, 8 / 0) | identical | - |
| On device (Mac) | 12 = 12 | 734 = 734 | 165 = 165 | 889 -> 510 (-42.6%, 8 / 0) | identical | 12 = 12 |
| Desktop (ONNX) | 12 = 12 | 726 = 726 | 163 = 163 | 890 -> 511 (-42.6%, 8 / 0) | identical | 12 = 12 |

Unverified real captures, desktop, before -> after (labels not yet checked by the user):
- real-015: "se can get lost when another is frometizer ... the the the the the ៨០៤៨៨៨៨" -> "ge can get lost when another
  app is frontmost during a / ving the wrong box size, withng off text. I'll fix the HUD by / the cursor by
  reapplying it on mouse-down, drag, and app"
- real-017: "...read r2day-small.comkod–screnshots show the.44" -> "...read. r2drag-small. / The capture worked—screenshots
  show the 4"
- real-009: one squashed strip -> its two lines.

## Menu and table rows (2026-10-27)

real-021 (a row of six Khmer category links on a marketplace site) came out with three of six items: the 1258 px line
was split into three ~400 px pieces, each holding two items with a wide blank between them, and the model stopped at
the blank. `split_wide` now also cuts at every blank of at least 2x the text height, when both sides hold at least half
a text height of ink (a thin scrollbar after a wide gap otherwise became a piece the model read as "គឺ" on English
paragraphs). Pieces changed on no synthetic clip (frozen, screen, slices, noisy all identical on device), only on
real-008 and real-021, which now reads all six items on Mac and desktop.

## Real captures: what is real

Of the 26 unverified captures, 18 (real-007..015, 017..020, 022, 024, 026..028) are screenshots of the Claude session
that tested the capture tool on 2026-10-25/26, saved because "Keep capture images" was on; they are marked in
`fixtures_real/labels.jsonl` as not real use. The user's own Khmer captures with readable text (016, 021, 023, 029) got
drafts re-read from the image; 001 and 006 (blurry / stylised Khmer) are left for the user. Nothing was marked
verified.

real-006 (the user's photo of a printed notice: paper ~145, vignetted corners ~126, three lines) still reads as one
squashed strip: the corner colour leaks into every row, and the clip is under 30% ink with an edge band that looks
uniform, so the local background is not used. Also using it when no row of the mask is blank changed no real capture
(even the local mask keeps 1-3% ink in the gaps) and split printed-031; not kept. Needs a pitch-based cut on the local
mask's row profile, or photo-specific handling.

real-001 is faint handwriting on lined paper (275x40, one strip, segmented correctly); the app reads it mostly right
and misses the two greetings (ជំរាបសួរ / ជំរាបលា read as "ជាបសូវ" / "វែរាបណ"). A model limit on faint handwriting,
not preprocessing.
Pitch cuts on the local mask don't reach it either: `line_pitch` finds no period in real-006's local row profile
(three lines, one a short heading), and on printed-031 the local profile's pitch would cut one word into five pieces.

## Two columns (2026-10-27)

real-008 (a settings panel over a chat, ~30 lines in two columns whose lines interleave) read as Khmer gibberish:
no row is blank, so each "line" strip held several lines of both columns. When a clip has more than one strip, a
strip at least 5x the typical glyph height is now cut at its widest blank column band (if at least 2x glyph height
wide) and each side is split into lines again, left column first (up to two levels). Thresholds: 2.5x / 1x changed
dozens of synthetic clips (a Khmer strip with its marks is ~2.5x its glyphs); 4x / 2x still touched two noisy clips;
5x / 2x touches none. A single-strip clip is left alone: real-001 (faint handwriting, tiny glyph estimate) split its
word gaps as gutters.

real-008 on device now reads the panel and the chat as English lines ("Qwen 3 0.6B Download / Smallest and fastest.
Good word completion on any Mac. - 396,7 MB / Commit and push the capture cursor and readout changes. ..."); only the
top chat bubble, whose outline joins its rows, is still garbled. Every eval set is identical on device (frozen 12,
screen 734, slices 165, noisy 1816, real 12). Desktop strips equal eval on 1,300 / 1,300; Mac piece counts equal
desktop on 1,300 / 1,300.
Follow-up: the top chat bubble of real-008 (five separated lines) was still one strip: against the whole clip's
background its line gaps vanished, while on its own the strip splits into its five lines. A tall strip without a
gutter is now split into lines again on its own. Pieces changed only on real-008 (47 -> 54 on desktop; Mac and desktop
piece counts still equal on 1,300 / 1,300), which now reads in full from "The capture readout now uses small, thin
digits next to the macOS crosshair..." through the panel.
Adaptive thresholding (ink = darker than the local mean by 15-25, windows 8-16 px) doesn't separate real-006 either:
photo grain keeps every row inked (1-3% of a row in the gaps), and the gap between its 2nd and 3rd lines stays at
~23% of the densest row. A grain-tolerant row split would only free the heading; not worth a rule that touches every
scanned clip.
