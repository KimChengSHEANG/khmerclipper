# v7: Khmer OS fonts + CADT textbook lines (2026-10-28): v7 dropped; v7b shipped (2026-10-30)

Data added (user approved both, 2026-10-27):
- **Khmer OS family** (12 faces, Danh Hong / Open Forum of Cambodia, LGPL-2.1+), fetched from Debian
  `fonts-khmeros_5.0-9_all.deb` (SHA-256 pinned in `scripts/fetch_fonts.py`), in every synthetic render: 100k general
  lines, 100k screen strips, 60k handwriting strips. The eval keeps Apple's Khmer MN / Khmer Sangam MN as held-out
  fonts; `scripts/by_font.py` reports the two groups separately.
- **CADT-IDRI/khmer-sala-ocr-16k** (CC BY-NC 4.0, scanned textbook pages with Gemini page transcriptions):
  `scripts/cadt_lines.py` cut 6,000 train pages into lines with the app's splitter, read them with v3, and kept a line
  with the closest transcription line as its label when they agreed within CER 0.3. Its 100 human-verified test
  pages became `cadt_test` (3,254 aligned lines, eval only; alignment kept at any CER, so absolute CER is high).

| Model | handwriting_test | cadt_test | frozen 200 | screen 450 | slices 320 | noisy 300 |
|---|---:|---:|---:|---:|---:|---:|
| v3 (shipped) | 0.0071 | 0.5995 | 0.0070 | 0.0158 | 0.0108 | 0.0862 |
| v7 last (step ~12k, early stop) | 0.0064 | 0.5882 | 0.0076 | **0.0259** | 0.0142 | 0.0835 |
| v7 best mid-train | 0.0096 | 0.5891 | 0.0086 | 0.0215 | 0.0133 | 0.0787 |

Per font, screen errors v3 -> v7 last: Khmer OS (now trained) 329 -> 577 (+75%), held-out Apple 387 -> 470 (+21%).
Worse even on the fonts it trained on, so the fonts were not the cause. The new errors look like the Gemini labels:
inserted words that aren't in the image ("ពណ៌សម្រាប់ឲ្យ" for "ពណ៌សឲ្យ"), "…" written as "...", and one clip
(screen-268, Vietnamese diacritics) decoding into a run of broken bytes. Gemini's page transcriptions are fluent
rewrites, not exact transcriptions; matching within CER 0.3 still lets those rewrites through as labels.

v7b (`scripts/run_v7b.sh`): the same fonts and renders without CADT, synth_screen replay back to 60k.

## v7b: fonts and handwriting renders, no CADT (2026-10-29)

PyTorch pipeline gate (same flags as the apps). `v7b last` is the final step; `v7b final` is the checkpoint the
trainer picked as best on its synthetic eval.

| Model | handwriting_test | cadt_test | frozen 200 | screen 450 | slices 320 | noisy 300 |
|---|---:|---:|---:|---:|---:|---:|
| v3 (shipped) | 0.0071 | 0.5995 | 0.0070 | 0.0158 | 0.0108 | 0.0862 |
| **v7b last** | **0.0054 (-24%)** | 0.6015 | 0.0070 | 0.0158 | 0.0106 | 0.0875 (+1.5%) |
| v7b final | 0.0054 | 0.5955 | 0.0072 | 0.0170 (+7.6%) | 0.0114 (+5.6%) | 0.0823 |

Per font, errors v3 -> v7b last: screen Khmer OS 329 -> 332, held-out Apple 387 -> 367 (-5.2%); slices 109 -> 104 and
55 -> 57; noisy 794 -> 796 and 1042 -> 1078 (+3.5%). Without CADT the regression is gone: the fonts and new
handwriting renders cut real handwriting errors by a quarter and leave printed and screen text where they were.
`v7b final` fails the 3% bar on screen and slices. `v7b last` passes; next is the on-device (CoreML) gate.

On device (`trocr-coreml`, Mac, paired character errors, v3 -> v7b last): frozen 200 12 -> 11, screen 450 734 -> 710
(-3.3%), slices 320 165 -> 162 (-1.8%), noisy 300 1792 -> 1821 (+1.6%), real 4 12 -> 12. All within the 3% bar.
Per font on screen: Khmer OS 345 -> 338, held-out Apple 389 -> 372. One subset moved against it: noisy_avatar
235 -> 283 (+20%, 8 better / 17 worse); the other noisy subsets net slightly better.

The compiled-model cache (`TrOCRCompiled/khmer-xkv-*`) is keyed by size only, and a package newer than the cache
replaces it, so benching two models in turn silently reuses the newer one. Clear it before each model.

Desktop (`trocr-desktop`, int8 decoder, `khmer-v3-kvb-int8` -> v7b last built the same way, 670 MB class): frozen 200
11 -> 11, screen 450 729 -> 711 (-2.5%), slices 320 162 -> 163, noisy 300 1777 -> 1890 (+6.4%), real 4 12 -> 12. Median
ms within noise of v3. Noisy misses the 3% bar on one clip: noisy-chrome-014 (+93 errors) runs on into broken bytes after
a Vietnamese name, the same failure as v7's screen-268, on desktop only. Without it noisy is 1777 -> 1797 (+1.1%).
noisy_avatar is +18% here too: on those clips v7b reads the horizontal divider rules as "────" and text past them.

## Shipped: v7b last with the rule and broken-byte cleanup (2026-10-30)

`drop_stray_symbols` / `dropStraySymbols` now also remove "─" runs of 3+ and U+FFFD on Khmer lines. Both models
re-measured with that code, cache cleared before each model, paired character errors v3 -> v7b:

| Set | Mac (CoreML) | Desktop (int8) |
|---|---:|---:|
| frozen 200 | 12 -> 11 | 11 -> 11 |
| screen 450 | 731 -> 707 (-3.3%) | 726 -> 707 (-2.6%) |
| slices 320 | 165 -> 162 (-1.8%) | 162 -> 163 (+0.6%) |
| noisy 300 | 1727 -> 1706 (-1.2%) | 1705 -> 1689 (-0.9%) |
| real 4 | 12 -> 12 | 12 -> 12 |

The cleanup alone took v3's noisy errors from 1792 to 1727 on Mac and 1777 to 1705 on desktop. noisy_avatar is back
within noise (181 -> 182, 172 -> 180). Median ms is unchanged. Handwriting (-24%) is from the PyTorch gate above and has
no on-device set. Desktop: `khmer-v7b-kvb-int8`, HF commit 27bb4e7. The Mac model is built locally into the git-ignored
`Vendor/`, so it reaches users with the next DMG. The v7b PyTorch weights are not published.
