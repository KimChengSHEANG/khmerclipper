# v7: Khmer OS fonts + CADT textbook lines (2026-10-28): dropped; v7b retries without CADT

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
