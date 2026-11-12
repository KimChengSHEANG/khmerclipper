#!/usr/bin/env python3
"""Open Development Cambodia law scans, pseudo-labelled by Chakkhu -> cache/odc parquet (v8b).

    python scripts/odc_pseudo.py LINES_DIR LABELS.csv [...] [--min_conf 0.97]

LINES_DIR holds labels.jsonl ({"id", "image"}) and the line crops cut from ODC's scanned laws and decrees by the
app's own line_strips / split_wide. Each CSV (id, conf, pred) is Chakkhu-OCR-Khmer (Apache-2.0) run on those
crops. Only lines it reads with conf >= min_conf are kept: on KHOB, conf >= 0.97 gave 1.2% CER.
One document in 20 (by name) goes to cache/odc/test ("odc_test"), eval only.
"""
import argparse
import csv
import json
import zlib
from pathlib import Path

from datasets import Dataset, Features, Image, Value

OUT = Path("cache/odc")


def doc(line_id: str) -> str:
    return line_id.rsplit("-", 2)[0]  # <doc>-<page>-<line>


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lines")
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--min_conf", type=float, default=0.97)
    a = ap.parse_args()
    root = Path(a.lines)
    image = {r["id"]: r["image"] for r in map(json.loads, open(root / "labels.jsonl"))}
    rows = {"train": [], "test": []}
    for f in a.labels:
        for r in csv.DictReader(open(f)):
            if float(r["conf"]) < a.min_conf or not r["pred"].strip():
                continue
            part = "test" if zlib.crc32(doc(r["id"]).encode()) % 20 == 0 else "train"
            png = (root / image[r["id"]]).read_bytes()
            rows[part].append({"image": {"bytes": png, "path": None}, "text": r["pred"].strip()})
    for part, rs in rows.items():
        (OUT / part).mkdir(parents=True, exist_ok=True)
        features = Features({"image": Image(), "text": Value("string")})
        Dataset.from_list(rs, features=features).to_parquet(str(OUT / part / "lines.parquet"))
        print(part, len(rs))


if __name__ == "__main__":
    main()
