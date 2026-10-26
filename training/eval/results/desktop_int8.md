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
