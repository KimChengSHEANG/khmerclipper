# Accuracy v3: screen-capture evaluation and candidate gate (2026-10-17)

Plan: [`docs/superpowers/plans/2026-10-17-accuracy-v3.md`](../../../docs/superpowers/plans/2026-10-17-accuracy-v3.md).
Each candidate below was gated on paired per-fixture character errors (`scripts/compare.py`) and kept only if it helped.

## New evaluation sets (local only, not committed)

| Set | Built by | Clips | What it is |
|---|---|---:|---|
| `fixtures_screen/` `screen` | `build_screen_fixtures.py render` | 300 | 1-5 wrapped lines, 20-90 chars, Chrome-rendered |
| `fixtures_screen/` `screen_long` | same | 150 | one 60-120 char line |
| `fixtures_slices/` `slice_*` | `build_screen_fixtures.py slices` | 320 | single lines, 40 per style: plain, small (10-13 px 1x), bold, low contrast, coloured, display font, numbers, mixed |

- Text: Khmer Wikipedia (`20231101.km`). 1,907 candidate clips were dropped because a line shared a 16-char shingle with a training source's labels; the training sources contain a lot of Wikipedia text.
- Fonts: Khmer Sangam MN, Khmer MN, Khmer OS System / Battambang / Khmer OS (slices add Khmer OS Muol / Bokor / Metal Chrieng). None is in training; all stay eval-only.
- 12-20 css px at 1x and 2x, light and dark, 15% bold, line height 1.4-1.8.
- Labels are exact (rendered from text). Lines joined by a space, as OCRBench prints them.

`fixtures_screen/` and `fixtures_slices/` are gitignored: the text is CC BY-SA and the fonts aren't redistributable. Rebuild with the commands in the script docstring.

## Pipeline runner

`khmerclipper_eval.pipeline` runs the app pipeline in PyTorch (Mac `lineStrips`, plain 384x384 resize, greedy, repeat cap 4, max 128 ids). `bench.py --engine trocr-pipeline --variant k=v,...` toggles candidates. Parity: on the frozen 200 it reproduces the shipped v2 (`checkpoint-5/checkpoint-10000`) ui 0.0032 and mixed 0.0133 exactly.

## Baseline: v2 on the screen set

| Subset | v2 CER (mean per clip) |
|---|---:|
| frozen 200 | 0.0081 |
| screen (1-5 lines) | 0.1953 |
| screen_long | 0.0398 |
| 1-line clips | 0.0286 |
| 2 / 3 / 4 / 5-line clips | 0.282 / 0.366 / 0.483 / 0.535 |

Cause: `lineStrips` found too few strips for 93 of 153 multi-line clips. At Khmer line heights the subscripts of one line touch the above-marks of the next, so no blank row separates them, and the merge rule then folded real lines into the merged double line as "small marks". Two to five lines were squashed into one 384x384 input.

## Candidates

Character errors, paired against the previous kept configuration. Sets: frozen 200, screen 450.

| Candidate | frozen errs | screen errs | +/- (screen) | Verdict |
|---|---:|---:|---:|---|
| (e) split touching lines at the line pitch | 40 -> 40 | 9394 -> 1835 (-80.5%) | 87 / 5 | **keep** (Mac, desktop) |
| (f) 2-token cycle cap (on top of e) | 40 -> 40 | 1835 -> 1817 (-1.0%) | 1 / 0 | drop (below the 5% bar) |
| (b) normalize (strip ZWSP) | 40 -> 40 | 1817 -> 1817 | 0 / 0 | drop (the model never emits ZWSP) |
| (a) wide split, R = 8 | 40 -> 41 | 1817 -> 1373 (-24.4%) | 126 / 78 | drop: slices 194 -> 214 (+10%), splits 8-12:1 lines that read fine whole |
| (a) wide split, R = 12 | 40 -> 40 | 1817 -> 1456 (-19.9%) | 99 / 35 | **keep**; slices 194 -> 193 |
| (a) wide split, R = 16 | 40 -> 40 | 1817 -> 1600 (-11.9%) | 55 / 13 | weaker than 12 |
| (d) content crop | 40 -> 39 | 1817 -> 1417 (-22.0%) | 129 / 58 | **keep**; slices 194 -> 184. Desktop already cropped; Mac now does too |
| (c) beam 4 when min greedy token p < 0.5 | 40 -> 37 | 1817 -> 1768 (-2.7%) | 33 / 13 | drop (below the bar; needs several KV states in the CoreML decoder) |

(b)-(d) and (f) are paired against (e) + (f), the configuration they were swept on; (f) itself was then dropped, which changes one clip.

The first wide-split version cut blank tails and neighbour-line mark fragments of full-width strips into pieces TrOCR hallucinated text for (multi-line clips +26% errors at R = 8). The kept version measures the line from ink in the strip's central band (rows 30-70%) and splits only that.

### On device (CoreML, Mac `OCRBench`, v2 model, Latin fallback on)

| Set | v2 as shipped | + split touching | + wide split 12 + content crop |
|---|---:|---:|---:|
| frozen 200 CER | 0.0044 | 0.0044 | **0.0039** |
| screen 450 CER | ~0.14 (pipeline 0.1435) | 0.0401 | **0.0260** |
| slices 320 CER | - | 0.0138 | 0.0141 |
| screen median ms | - | 609 | 666 |

Paired, last column vs the middle one: frozen 15 -> 14 errors; screen 1850 -> 1177 (-36.4%, 160 better / 49 worse; long lines -72.5%); slices 195 -> 194 (24 / 19). Per slice, bold (+7 errors) and display (+8) got worse while small (-8), coloured (-5) and low contrast (-2) improved.

### Residual errors are partly the labels

The most common screen-set "error" is ដ -> ត after a coeng (111 times): the subscripts ្ដ and ្ត are the same glyph in these fonts, so pixels can't tell them apart; Wikipedia often writes ្ដ where training labels use ្ត. Next are deprecated characters in the references (ឤ U+17A4 read as អា, ឣ U+17A3 read as អ), which render identically. These cost both sides of every paired comparison equally, so no keep/drop decision depends on them.

(e) splits runs of touching lines at the row-profile autocorrelation pitch, cutting only sparse valleys (<= 0.2 x the run's densest row), before the mark merge. Clips segmented correctly: 357 -> 439 of 450; frozen fixtures stay one strip each. Swift and desktop strip counts match the Python reference on all 650 fixtures.

## v3 fine-tune (Task 4)

From v2 step 10k, lr 1e-5, 1 epoch, `--augment --max_aspect 32`. New sources: `synth_screen` (204k strips, `render_synth.py --screen`) and LazyGreed (179,916 of 179,988 rows after removing eval-text overlap). KhmerST was excluded: its upstream repository states no licence. All rows below use the final app config (split touching + wide split 12 + crop), paired against v2 in the same config.

| Model | frozen errs | screen errs | slices errs | English UI line acc | Verdict |
|---|---:|---:|---:|---:|---|
| v2 step 10k | 39 | 1153 | 188 | 0.743 | baseline |
| v3 (`synth_en` 40k), step 15242 | 36 | 921 (-20.1%) | 169 (-10.1%) | 0.720 | drop: English UI regressed |
| v3 (`synth_en` 40k), best mid-train (12k) | 36 | 971 (-15.8%) | 177 (-5.9%) | 0.727 | drop |
| **v3b (`synth_en` 120k), step 16000** | **35** | **946 (-18.0%, 83 / 35)** | **164 (-12.8%)** | **0.787** | **keep, shipped** |
| v3b, best mid-train (8k) | 38 | 1016 (-11.9%) | 175 (-6.9%) | 0.727 | drop |

Mid-train CER (mixed holdout) picked early steps both times, as it did for v2; the late checkpoint was better on every gate set.

On device (CoreML, `OCRBench`), v3b vs v2 with the same app changes: frozen CER 0.0039 -> **0.0034** (14 -> 11 errors), screen 0.0260 -> **0.0215** (-17.9%, 85 / 35), slices 0.0141 -> **0.0109** (-14.9%); frozen median 102 ms. Desktop ONNX export parity max abs diff 2.9e-5.

## Follow-ups (2026-10-21, on-device CoreML, v3 model unchanged)

| Change | frozen 200 | screen 450 | slices 320 |
|---|---|---|---|
| Cross-attention K/V once per line + up to 4 lines decoded concurrently | identical text; median 102 -> 100 ms | identical text; median 634 -> 401 ms | identical text; median 349 -> 292 ms |
| Two-line cut when no pitch is found (`twoLineCut`) | 15 -> 15 errors | CER 0.0215 -> **0.0163** (997 -> 755 errors, 6 better / 0 worse) | 171 -> 171 errors |

- Segmentation: 7 of 68 two-line screen clips came out as one strip because `linePitch` found no autocorrelation peak (the wrapped second line is often short). Strips now correct on 445 of 450 screen clips (was 439), still 200/200 frozen and 319/320 slices.
- Capture resolution: `SCDisplay.width` is in points, so Retina captures were taken at 1x. Paired on the 225 2x screen clips, downscaling to 1x gives CER 0.0232 vs 0.0199 at 2x. The app now captures the selection at the screen's backing scale.
- Residual errors (screen + slices, excluding ្ដ/្ត and deprecated-character label noise): CER 0.0178. Worst slices: font Khmer MN 0.0273, size <= 13 px 0.0244, bold 0.0228, dark 0.0210. Most common confusions are dropped/added vowel signs (ុ, េ, ែ, ើ->ី) and `:` read as `ៈ`. Next training data: more `synth_screen` strips in Khmer MN-like fonts at 11-13 px, bold and dark, and mixed Latin punctuation.

## v4 fine-tune (2026-10-23): dropped

`scripts/run_v4.sh`: from shipped v3 (step 16000), lr 5e-6, 1 epoch (15,884 steps, 2.2 h), adding 160k `synth_screen_v4` strips (10-14 px, half dark, a third faux-bold at >= 20 px, 10% with a Latin `:` lead-in) over a replay of the v3b mix. Gated with the PyTorch app pipeline (split touching + two-line cut + wide split 12 + crop), paired character errors vs v3 in the same run:

| Model | frozen 200 | screen 450 | slices 320 | Verdict |
|---|---:|---:|---:|---|
| v3 (step 16000, shipped) | 41 | 737 | 170 | baseline |
| v4 last (step 15884) | 41 | 732 (-0.7%, 34 / 26) | 172 (+1.2%) | drop: far below the 5% bar |
| v4 best mid-train (8k) | 42 | 742 (+0.7%) | 179 (+5.3%) | drop |

On the slices v4 targeted (screen + slices, v3 -> v4 last): bold 93 -> 102 (+9.7%), dark 285 -> 294 (+3.2%), size <= 13 px 275 -> 271, Khmer MN 296 -> 283, `:` read as `ៈ` 14 -> 12. More OFL-font synthetic strips in these styles don't move the residual errors; faux-bold likely hurts. The remaining screen errors look font-specific (Khmer MN and other Apple/Khmer OS faces that are eval-only and not redistributable), so the next lever is real captures from users' screens (the app's report/feedback path), not more synthetic data. `eval_english.py` needs macOS fonts and was not run on the box; it was moot after the screen gate.

Box note: `checkpoint-8/final` there is v3's best mid-train step (8k); the shipped model is `checkpoint-8/checkpoint-16000` (assembled with tokenizer files as `experiments/v3_16k`).
