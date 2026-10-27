# Desktop TrOCR: int8 decoder (2026-10-26)

`bench.py --engine trocr-desktop` (ONNX CPU, Latin fallback on), v3 KV model, Apple Silicon Mac.
Paired character errors from `scripts/compare.py`; bar: no set more than 3% worse, and faster.
Latency is wall-clock on a laptop in use, so compare within a row.

| Candidate | download | frozen 200 | screen 450 | slices 320 | noisy 300 | real 4 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| fp32 (shipped `khmer-v3-kv`) | 1469 MB | 11 err, 454 ms | 727, 1619 ms | 164, 1165 ms | 3215, 1541 ms | 12, 912 ms | baseline |
| int8 encoder + decoder (per tensor) | 391 MB | CER 0.0034 -> 0.0111 | +13% CER, 706 ms | +16% CER, 386 ms | +8% CER, 1104 ms | = | drop: accuracy |
| int8 encoder + decoder (per channel) | 374 MB | CER 0.0034 -> 0.0101, 229 ms | - | - | - | - | drop: the encoder is the sensitive half |
| **int8 decoder (per channel), fp32 encoder** | **670 MB** | **11, 405 ms** | **726, 1187 ms** | **163, 596 ms** | **3231 (+0.5%), 1080 ms** | **12, 575 ms** | **keep** |
| fp32, 4 lines decoded at once (threads) | 1469 MB | identical text, 425 ms | 1 clip differs, 1466 ms | identical, 850 ms | 1 clip differs, 2959 ms | identical, 2328 ms | drop: concurrent ONNX runs fight over the CPU's cores |

Built with `apps/desktop/tools/quantize_model.py` (deterministic: same decoder SHA-256 on rebuild).
The decoder is ~70% of the fp32 weights: its 52M-weight token embedding is stored twice (input
embedding and output projection), and every decode step reads the projection.

## Lines decoded as one batch (2026-10-27)

`tools/export_onnx.py` now exports a dynamic batch axis (parity at batch 1 and 2: max abs diff 3.0e-5), and
`trocr.recognize_with_session` decodes up to 4 pieces of a capture together in lockstep. Batch-1 CPU decode is
memory-bound (every step reads all weights for one token), so a batch costs about one line. Models with a fixed
batch axis (`khmer-v3-kv-int8`, shipped) fall back to one piece at a time; on the frozen set that gives exactly the
old text apart from the two clips results/busy_background.md changed.

int8 decoder, unbatched (`khmer-v3-kv-int8`, measured before the local-background change) vs batched
(`khmer-v3-kvb-int8`, after it; the noisy row's gain is that change, see results/busy_background.md):

| Set | errors | change | median ms |
|---|---:|---:|---|
| frozen 200 | 11 -> 11 | 2 / 1 | 405 -> 344 |
| screen 450 | 726 -> 729 | +0.4% (3 / 5) | 1187 -> 869 (-27%) |
| slices 320 | 163 -> 162 | 2 / 1 | 596 -> 537 |
| noisy 300 | 3231 -> 2387 | -26.1% (mostly local background) | 1080 -> 945 |
| real 4 | 12 -> 12 | 0 | 575 -> 567 |

Batched texts differ slightly because dynamic int8 quantizes activations with one scale per batch; errors stay
within the bars. Not shipped until `khmer-v3-kvb-int8` is published to the model repo.

Partial int8 encoders (per channel, batched model, frozen 200 CER, fp32 encoder 0.0031-0.0035): MLP matmuls only
0.0113 (259 ms), MLP + attention projections of layers 2-9 0.0102 (231 ms). Dropped: any int8 in the ViT encoder
triples clean-text errors, though it is now 64% of desktop time on screen clips.
