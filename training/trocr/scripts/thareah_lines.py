#!/usr/bin/env python3
"""Thareah/khmer-handwriting (MIT, 147 scanned pages split by page) -> cache/thareah parquet (v8).

    python scripts/thareah_lines.py [local clone]

Pass a local clone (git clone https://huggingface.co/datasets/Thareah/khmer-handwriting) to skip the
hub's per-file downloads: ~2.9k images hit the anonymous rate limit.

train + validation go to cache/thareah/train (registry "thareah"); test goes to cache/thareah/test
("thareah_test"), eval only: its pages never reach training.
"""
import sys
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

OUT = Path("cache/thareah")


def main() -> None:
    ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "Thareah/khmer-handwriting")
    for part, splits in (("train", ["train", "validation"]), ("test", ["test"])):
        rows = concatenate_datasets([ds[s].select_columns(["image", "text"]) for s in splits])
        (OUT / part).mkdir(parents=True, exist_ok=True)
        rows.to_parquet(str(OUT / part / "lines.parquet"))
        print(part, len(rows))


if __name__ == "__main__":
    main()
