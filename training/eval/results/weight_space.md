# Weight-space programming on M2: no accuracy gain; fused weights and per-layer inputs make captures 14% faster (2026-11-03)

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

## The edits that work: per-layer cross K/V, fused Q/K/V weights (desktop)

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
