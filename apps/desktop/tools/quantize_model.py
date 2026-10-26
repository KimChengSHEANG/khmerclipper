"""Int8 decoder weights for an exported TrOCR ONNX model (download 1.47 GB -> 0.67 GB, decode ~30% faster).

    .venv/bin/python tools/quantize_model.py exported/khmer-v3-kv exported/khmer-v3-kv-int8

Dynamic quantization: weights stored as int8 (per output channel), activations quantized per run,
so no calibration data is needed. Only the decoder: on the eval sets an int8 decoder matched fp32
errors (+0.5% on the noisy set, 0 elsewhere), while an int8 encoder tripled clean-text errors.
Gate the result with training/eval `bench.py --engine trocr-desktop` before publishing it with
tools/publish_models.py.
"""

import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    # Gather covers the 52M-weight token embedding, as large as all the decoder layers together.
    quantize_dynamic(args.src / "decoder.onnx", args.dest / "decoder.onnx", weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"], per_channel=True)
    for name in ("encoder.onnx", "tokenizer.json"):
        shutil.copy2(args.src / name, args.dest / name)
    for p in sorted(args.dest.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
