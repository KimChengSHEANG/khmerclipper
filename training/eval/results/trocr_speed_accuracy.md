# TrOCR CoreML speed + accuracy, by stage

Full frozen 200 fixtures, Mac CoreML via `OCRBench` (Release), same `bench.py`
run as Tesseract. Plan: `docs/superpowers/plans/2026-10-03-trocr-speed-accuracy.md`.
Printed fixtures leaked into this checkpoint's training (soyvitou testset), so
absolute printed CER is optimistic; ui fixtures are clean.

| Stage | CER | printed | ui | mixed | median ms | strips/img | enc ms | steps/img | ms/step | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tesseract | 0.2580 | 0.5370 | 0.0199 | 0.0894 | 107.9 | — | — | — | — | — |
| 0 baseline (FP32, no KV) | 0.5577 | 0.1982 | 1.0514 | 0.0598 | 1367.2 | 1.1 | 36.6 | 17.2 | 173.1 | FAIL |
| HF greedy + run cap (PyTorch CPU) | 0.0593 | 0.0433 | 0.0662 | 0.0950 | — | — | — | — | — | — |
| 1 parity (lines, PIL resize, no crop, cap 128, run cap 4) | 0.0593 | 0.0433 | 0.0662 | 0.0950 | 1286.8 | 1.0 | 31.9 | 13.5 | 148.5 | FAIL |
| 2 KV fp16 units=all | 0.0593 | 0.0433 | 0.0662 | 0.0950 | 105.9 | 1.0 | 27.4 | 13.5 | 7.1 | FAIL |
| 2 KV fp16 units=gpu | 0.0593 | 0.0433 | 0.0662 | 0.0950 | 95.3 | 1.0 | 18.2 | 13.5 | 7.5 | FAIL |
| 2 KV fp16 units=ane | — | — | — | — | — | — | — | — | — | N/A |

## Notes

OCRBench stdout (stage 0+): multi-strip recognition is flattened with
`split(whereSeparator: \.isNewline).joined(separator: " ")` so one image
always yields one text line for `bench.py` (plain `replacingOccurrences`
of `"\n"` left an extra line on at least one multi-strip fixture).

Stage 2 KV-cache FP16 (2026-10-03): default app compute units `.cpuAndGPU`
(`units=gpu`). ANE (`units=ane`) fails CoreML plan build (error -14) on the
stateful KV decoder; not benchmarked. Steps 3–4 (encoder FP32 / int8) skipped —
stage-2 CER matches stage 1 and median is well under 2× Tesseract. Gate fails
only on mixed-subset CER (see `mac_gate.md`).

## Int8 decoder weights on Mac (2026-10-27): dropped

`coremltools.optimize.coreml.linear_quantize_weights` (linear symmetric, per channel) on the shipped v3 KV decoder:
436 MB -> 219 MB and identical decode steps on 60 screen clips, but 21.9 ms per step vs 7.2 ms in fp16 on
`.cpuAndGPU` (median clip 852 ms vs 315 ms). The GPU dequantizes on every step, which costs more than the halved
weight reads save. (On desktop ONNX CPU the int8 decoder is a win: results/desktop_int8.md.)

## Encoder on the Neural Engine, decoder on the GPU (2026-10-27): no gain

Separate compute units per model (encoder `.cpuAndNeuralEngine` or `.all`, decoder `.cpuAndGPU`), 60 screen clips,
two runs each: median 346-381 ms vs 350-357 ms with both on the GPU, identical text. Encoding is not the
bottleneck the overlap would hide; not kept.

## Batched decoder on Mac (2026-10-27 spike): not built

A batch-4 stateful KV decoder (same conversion with a batch axis, fp16, `.cpuAndGPU`), timed with coremltools on
the khmer v3 checkpoint: 10.3 ms per step at batch 1, 29.7 ms at batch 4 (7.4 ms per line-step). The app's 4
concurrent batch-1 lines already overlap about 2x (60 screen clips: 39 s of summed decoder time in 20 s of wall time,
~3.6 ms per line-step), so a batch would be slower. Unlike desktop CPU decode (memory-bound; batching -27%,
results/desktop_int8.md), the GPU step is compute-bound: cross-attention over 577 patches per line scales with B.
