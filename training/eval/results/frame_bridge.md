# English captures read as Khmer: frames, bridged lines, word-level Latin fallback (2026-11-01)

Report: a 6-line English terminal clip came back as Khmer gibberish while Tesseract read it fine. One strip went to
the model: every line squashed into one image. Three changes, in pipeline.py, apps/desktop and TrOCREngine.swift:

- **`trim_frame`.** A flat band on the clip's edge in a colour other than the clip's most common one is ink on every
  row, so no row is blank. It is cropped first. Flat within 12, and 3 lines past it the colour must have jumped by the
  ink threshold: the first version used 40 and no jump, trimmed gradients and took noisy-background-035 from 62 to 199.
- **`split_bridged`.** A run at least 1.6x the median run (3+ runs) gets `two_line_cut`. The clip's mouse pointer joined
  two lines, a blank line doubled the detected pitch so `touching_cuts` never ran, and the tall run made every other
  line look like a mark, collapsing the clip. The apps hide the pointer, but any tall icon does the same.
- **`latin_line`.** A Latin-only line used to need every Tesseract word at 70+. One weak word kept all the model's
  misspellings ("forr", "petterns"). Now, when both engines split the line into the same words, each word at 90+ is
  taken. At 70 handwriting got 4 clips worse (+5 errors): wrong words reach 78 there, right ones on clean text 93+.

Paired errors, v7b as released -> this change (Levenshtein totals, so frozen differs from compare.py's):

| Set | Mac | Windows/Linux |
|---|---:|---:|
| frozen 200 | 15 -> 15 | 15 -> 15 |
| screen 450 | 707 -> 707 | 707 -> 707 |
| slices 320 | 162 -> 163 | 163 -> 164 |
| noisy 300 | 1706 -> 1601 (-6.2%) | 1689 -> 1583 (-6.3%) |
| real 4 | 12 -> 12 | 12 -> 12 |
| handwriting 307 | 171 -> 171 | 170 -> 170 |

Screen and real were last measured at word bar 70; the bar change only removes swaps, and neither set had any. Noisy's
gain is almost all noisy-background-050, one framed clip (148 -> 43). The eval sets hold few framed clips: `trim_frame`
changes 2 of 300 noisy clips and none in frozen, screen or slices, but 9 of 30 real captures, of which only 4 are
labelled, so its real-world effect is not measured here.

The reported clip now reads as English on Mac with one slip, "1 want" for "i want" on a highlighted line.

## Capitals and a lone "1" (2026-11-02, after 0.4.1)

The clip's last slips came from Tesseract, not the model: "1" for "i" at 93 and "Change" for "change" at 79, both over the
line bar, so they replaced the model's correct words. Tesseract's word now gives way when it differs from the model's only
in capitals, or is a lone "1" where the model has i, I or l. Each word is compared counting from both ends of the line, since
the model often merges two words. Replaying the 108 logged Latin lines through the desktop rule changed one: the clip's. That
replay used the desktop model's lines on the frozen, screen, noisy, slices and real sets, not handwriting, and the Mac has
only its own tests and the clip behind it.
