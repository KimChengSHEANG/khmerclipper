#!/usr/bin/env python3
"""Real scanned textbook lines from CADT-IDRI/khmer-sala-ocr-16k (CC BY-NC 4.0; page images + Gemini page
transcriptions; only its 100-page test split is human-verified).

Pages are cut into lines with the app's line splitter (training/eval pipeline, same flags as the apps), each
line is read with the current model, and the reading is matched to the closest line of the page's
transcription. A line is kept with the transcription's text as its label when they agree closely (CER <=
--max_cer): the label comes from the page, the match only aligns it. LaTeX lines (any backslash) and lines
without Khmer are skipped: the model reads plain text.

    python scripts/cadt_lines.py --split train --model experiments/v3_16k --out cache/cadt/train
    python scripts/cadt_lines.py --split test --model experiments/v3_16k --out cache/cadt/test --max_cer 1.0

On the test split use a loose --max_cer: it is an eval set, and filtering by the model's own agreement would
make it easier for that model.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from datasets import Image as DSImage
from PIL import Image
from rapidfuzz.distance import Levenshtein

EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL / "src"))
from khmerclipper_eval import pipeline  # noqa: E402

REPO = "hf://datasets/CADT-IDRI/khmer-sala-ocr-16k/data"
SPLITS = {"train": [f"{REPO}/train-{i:05d}-of-00010.parquet" for i in range(10)],
          "validation": [f"{REPO}/validation-{i:05d}-of-00002.parquet" for i in range(2)],
          "test": [f"{REPO}/test-00000-of-00001.parquet"]}
KW = dict(robust_bg=True, drop_noise=True, drop_edge=True, local_bg=True, letters_all=True, max_gap=True)
KHMER = re.compile("[ក-៿]")
FEATURES = Features({"image": DSImage(), "text": Value("string"), "book_id": Value("string"),
                     "page": Value("int64"), "cer": Value("float32")})


def page_lines(image: Image.Image) -> list[Image.Image]:
    strips = pipeline.line_strips(image, True, **KW)
    if len(strips) > 1:
        strips = [c for s in strips for c in pipeline.split_columns(s, **KW)]
    pieces = []
    for s in strips:  # the model reads one line of up to ~12:1; long lines go in as their split pieces
        pieces += [p for _, p in pipeline.split_wide(s, 12, robust_bg=True, local_bg=True, big_gap=True)]
    return [pipeline.content_crop(p, robust_bg=True, drop_noise=True, local_bg=True) for p in pieces]


def label_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line and "\\" not in line and KHMER.search(line):
            out.append(line)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=sorted(SPLITS), default="train")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max_cer", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=0, help="pages (0 = all)")
    p.add_argument("--shard_size", type=int, default=5000)
    a = p.parse_args()

    ds = load_dataset("parquet", data_files=SPLITS[a.split], split="train")
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    device = pipeline.default_device()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, shard, kept, seen = defaultdict(list), 0, 0, 0
    for n, ex in enumerate(ds):
        labels = label_lines(ex["text"])
        if not labels:
            continue
        lines = [l for l in page_lines(ex["image"].convert("RGB")) if l.width >= 8 and l.height >= 8]
        if not lines:
            continue
        preds = []  # batches of 32: a dense page in one batch ran the GPU out of memory
        for i in range(0, len(lines), 32):
            preds += [t for t, _ in pipeline.decode(lines[i:i + 32], Path(a.model), device)]
        for img, pred in zip(lines, preds):
            seen += 1
            if not pred.strip():
                continue
            best = min(labels, key=lambda t: Levenshtein.normalized_distance(pred, t))
            cer = Levenshtein.distance(pred, best) / max(1, len(best))
            if cer > a.max_cer:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            rows["image"].append({"bytes": buf.getvalue()})
            rows["text"].append(best)
            rows["book_id"].append(str(ex.get("book_id", "")))
            rows["page"].append(int(ex.get("page", 0) or 0))
            rows["cer"].append(cer)
            kept += 1
        if len(rows["text"]) >= a.shard_size:
            Dataset.from_dict(rows, features=FEATURES).to_parquet(str(a.out / f"shard-{shard:05d}.parquet"))
            shard, rows = shard + 1, defaultdict(list)
        if n % 50 == 0:
            print(f"page {n}/{len(ds)}: kept {kept} of {seen} lines", flush=True)
    if rows["text"]:
        Dataset.from_dict(rows, features=FEATURES).to_parquet(str(a.out / f"shard-{shard:05d}.parquet"))
    print(f"done: kept {kept} of {seen} lines from {len(ds)} pages")


if __name__ == "__main__":
    main()
