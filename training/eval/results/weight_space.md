# Weight-space programming on M2: one bias edit cuts errors 3-17%, fused weights make captures 14% faster (2026-11-03)

Weight-space programming (MIT CSAIL, "Sparks of Weight-Space Programming", 2026-11-02): an LLM reads a network's
weights, finds the units behind a failure and edits them, with no gradients. Its headline win removed a reasoning loop
in Qwen3.5-0.8B; on a model without that kind of fault (Gemma 3 270M) it moved almost nothing. Tried here on M2 (v7b).

## Is there a mechanical fault to edit?

M2's errors on all sets (desktop, 0.4.0 baseline CSVs): 912 substitutions, 1094 deletions, 750 insertions.

- **ត for ដ after coeng, 174 (the largest single error).** ្ត and ្ដ render the same, so only spelling decides, and the
  labels disagree with themselves: ស្តេច 5 / ស្ដេច 10, ចិត្ត 18 / ចិត្ដ 3, 14 of 288 words in both spellings. Biasing
  the model toward ្ដ would fit label noise, not reading.
- **ៈ for ":", 16.** Labels use an ASCII colon in words like លក្ខណៈ; the model is right.
- **Edge-cut clips (mean CER 0.25, 9 of 50 output much longer than the label).** The label drops the half-cut line the
  model reads. A labelling policy, not a weight fault.
- **Loops: 4 clips.** The rest are scattered look-alike vowels and signs, no single direction to edit.

## Edits tried (no training)

| Edit | Result |
|---|---|
| Prune the output layer to used tokens | Only 923 of 50,414 tokens appear in 23k local texts, but an English word list needs 34,145. Output layer = 3.9% of desktop decoder time, ~1.1 of ~16 ms per Mac step. Not worth it. |
| Drop one decoder layer (frozen set probe) | Layers 2-5 each: 17-18 errors vs 18. Layers 0, 1: broken. 8-11: 34-44. |
| Drop two or more | (2,4) 21, (3,4) 37, (2,3,4) 108, (2,3,4,5) 779. |
| Drop layer 4, desktop int8, all sets | Screen 707 = 707, frozen 16 vs 15, hw 168 vs 170, but noisy 1842 vs 1583 (+16%) and slices 188 vs 164 (+15%). 2-4% faster per clip. Fails the gate. |
| Drop layer 3 | Worse: screen 841, noisy 2046, slices 249. |
| Extrapolate M2 away from M1, M1 + λ(M2 − M1), desktop int8, all sets | λ 1.15: frozen 15 = 15, screen 707 = 707, noisy 1610 vs 1583, slices 168 vs 164, real 12 = 12, hw 170 = 170. λ 1.10: 15, 710, 1607, 166, 12, 169. Its probe gain (slices 197 → 174, HF on MPS) didn't carry over. No better anywhere. |
| Soup: interpolate M1 and M2, λ 0.5 / 0.75, desktop int8 | noisy 1615 / 1606 vs 1583, screen 720 / 716 vs 707, slices 165 / 167 vs 164. Worse. |

Verdict: M2 has no single fault of the kind weight-space programming fixes; its largest "errors" are label noise.
Removal edits cost far more accuracy than they save time, and extrapolating along the M1 → M2 direction gains nothing.

## Accuracy: an end-of-text bias (desktop)

Of M2's noisy-set errors, 429 of 1583 come from 13 clips where the model keeps writing after the real text
(`…ចង្អៀត══ ចំណោយ ឯបហាល` for `…ចង្អៀត`); early stops are rare (2 clips). One weight edit: an output-layer bias of
+3 on `</s>` (the decoder's output layer has none), in `tools/export_onnx.py` as `EOS_LOGIT_BIAS`.

| Set (errors) | M2 | +2 | +3 | +4 | +5 |
|---|---|---|---|---|---|
| frozen | 15 | | 15 | | |
| screen | 707 | | 687 (−2.8%) | | 674 |
| noisy | 1583 | 1532 | 1489 (−5.9%) | 1694 | 2102 |
| slices | 164 | | 160 (−2.4%) | | |
| real | 12 | | 12 | | |
| hw | 170 | | 142 (−16.5%) | | 128 |

Changed clips at +3: hw 6 better / 1 worse, noisy 29 / 2, screen 12 / 5. The losses are lines cut short: a long number
loses digits (`១០០០០០០` → `១០០`), one noisy clip drops a middle phrase. Caveats: +3 was picked on these same sets (the
grid was coarse: 2-5) and noisy turns worse just past it, so the margin is narrow.

The Mac CoreML decoder (`apps/mac/tools/trocr/kv_decoder.py`, `make trocr-khmer`), same +3, OCRBench:

| Set (errors) | M2 | +3 |
|---|---|---|
| frozen | 15 | 15 |
| screen | 707 | 684 (−3.3%) |
| noisy | 1601 | 1473 (−8.0%) |
| slices | 163 | 160 (−1.8%) |
| real | 12 | 12 |
| hw | 171 | 140 (−18.1%) |

Speed unchanged (median ms per set within 1-2%, except real: 4 clips).

## Speed: per-layer cross K/V, fused Q/K/V weights (desktop)

Same weights, rewired graph. The desktop decoder took all 12 layers' cross-attention K/V as one stacked
`cross_k`/`cross_v` input and split it on every step: a copy of ~110 MB per step at batch 4, 23% of step time.
`tools/export_onnx.py` now gives the decoder `cross_k_0`, `cross_v_0`, … inputs; the runtime passes per-layer views
of the encoder output (no copy) and still accepts the stacked layout of models already downloaded.

| M2, int8 decoder, batch 4, 40 steps | Stacked | Per-layer |
|---|---|---|
| Decoder step | 15.2 ms | 12.1 ms (−20%) |
| `Split` share of step | 23% | 0 |
| Whole capture, 120 clips from 4 sets, A/B interleaved | 84.5 s | 79.4 s (−6%) |
| Same text | | 120 / 120 |
| Total errors, all six sets (frozen, screen, noisy, slices, real, hw) | 15, 707, 1583, 164, 12, 170 | identical |

Then a weight edit on top, same function: each self-attention's Q, K and V matrices concatenated into one
[3072, 1024] matrix with the 1/8 attention scaling folded into the Q rows (and into cross-attention Q). The int8
decoder then quantizes each layer's input once instead of three times and runs one matmul instead of three. Per-channel
int8 scales are per output row, so the fused matrix quantizes to the same integers.

| M2, int8 decoder | Baseline | Per-layer + fused Q/K/V |
|---|---|---|
| Decoder step, alternating runs; baseline = per-layer alone | 10.0 ms | 9.5 ms (−5%) |
| Whole capture, 120 clips, A/B interleaved; baseline = shipped | 217.0 s | 186.5 s (−14%) |
| Same text | | 120 / 120 |
| Total errors, all six sets | 15, 707, 1583, 164, 12, 170 | identical |

The two A/B runs differ in absolute time because of machine load; each compares models run alternately under the same
load. Export parity 3.0e-5 (fused), 4.7e-5 (per-layer). Shipping it needs a new ONNX upload and manifest pin, so desktop users would download the model
again (~700 MB). The Mac CoreML model has its own conversion and is untouched.

## KHOB (2026-11-11)

KHOB level 1 (EKYCSolutions/khmer-ocr-benchmark-dataset, 31-01-2026 release), 1,862 line crops from
`scripts/build_khob_fixtures.py`. An earlier local `fixtures_khob` predated the task-3/4 point scaling in that script:
its task-3/4 crops showed the wrong lines (pooled CER 0.82 / 1.20). Rebuilt before these runs.

Errors / label characters (pooled CER), M2 vs M2 with the +3 end-of-text bias:

| Task (chars) | Mac base | Mac +3 | Desktop base | Desktop +3 |
|---|---|---|---|---|
| task-1 (29,589) | 750 (0.025) | 741 (0.025) | 750 (0.025) | 744 (0.025) |
| task-2 (30,290) | 5,767 (0.190) | 5,683 (0.188) | 5,747 (0.190) | 5,657 (0.187) |
| task-3 (30,245) | 17,251 (0.570) | 16,946 (0.560) | 17,298 (0.572) | 16,962 (0.561) |
| task-4 (30,248) | 4,993 (0.165) | 4,738 (0.157) | 5,055 (0.167) | 4,779 (0.158) |
| all (120,372) | 28,761 (0.239) | 28,108 (0.234) | 28,850 (0.240) | 28,142 (0.234) |
| mean per-line CER | 0.283 | 0.234 | 0.284 | 0.234 |

Better on every task on both platforms, −2.3% errors overall; mean per-line CER drops more (−17%) because the bias
mostly fixes short lines the model ran past. Task 3 is small low-resolution page scans: correct crops, still 0.56.
Mac: CoreML via OCRBench (median 324 ms/line). Desktop: the exported decoder in fp32 on an RTX 3090 (CUDA provider,
cross K/V and the cache kept on the GPU; median 406 ms/line, mostly CPU pipeline time) without Tesseract, so
Latin-only lines keep the model's text; int8 matched fp32 within 0.5% on the noisy set.
