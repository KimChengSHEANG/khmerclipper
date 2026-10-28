#!/usr/bin/env python3
"""Character errors per font group for paired bench CSVs: the held-out Apple fonts (Khmer MN, Khmer Sangam MN,
never trained on) vs the Khmer OS family (trained on from v7). A model that only learned the Khmer OS faces
improves the second group and not the first.

    python scripts/by_font.py fixtures_screen results/gate7_v3_fixtures_screen.csv results/gate7_v7_last_fixtures_screen.csv
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HELD_OUT = {"Khmer MN", "Khmer Sangam MN"}


def main() -> None:
    fixtures, base, cand = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    font = {json.loads(l)["id"]: json.loads(l).get("font", "") for l in open(fixtures / "labels.jsonl")}
    rows = [{r["id"]: r for r in csv.DictReader(open(f))} for f in (base, cand)]
    stats = defaultdict(lambda: [0, 0, 0, 0, 0])  # chars, base errs, cand errs, better, worse
    for k, b in rows[0].items():
        c = rows[1].get(k)
        if c is None:
            continue
        group = "held-out (Apple)" if font.get(k) in HELD_OUT else "Khmer OS (trained from v7)"
        eb, ec = Levenshtein.distance(b["pred"], b["ref"]), Levenshtein.distance(c["pred"], c["ref"])
        s = stats[group]
        s[0] += len(b["ref"]); s[1] += eb; s[2] += ec; s[3] += ec < eb; s[4] += ec > eb
    for g, (n, eb, ec, better, worse) in sorted(stats.items()):
        print(f"{g:28} chars {n:6}  errors {eb:5} -> {ec:5} ({(ec - eb) / max(1, eb):+.1%})  better {better} / worse {worse}")


if __name__ == "__main__":
    main()
