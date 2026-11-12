"""Crop KHOB (EKYCSolutions/khmer-ocr-benchmark-dataset) LabelMe polygons into line fixtures.

Usage: python scripts/build_khob_fixtures.py <khob-dataset/level-1 dir> [out dir, default fixtures_khob]

Download the level zips from the Google Drive folder linked in the KHOB README first.
Each polygon becomes its axis-aligned bounding box plus a small margin. Labels write a
space as a literal "\\t"; that becomes a real space, zero-width spaces and stray
backslashes are dropped. Ids keep the task (task-1..4) so results can be split by it.

Some pages' boxes drift down the page against the image (task-3 goc_2: right at the top, most of a line low at
the bottom), so a crop shows the next line instead of its label's. register() fits y' = a*y + b per page so box
middles land on ink and box edges on the gaps between lines; pages that already fit come out near (1, 0).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def clean_label(label: str) -> str:
    text = label.replace("\\t", " ").replace("\\", "").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def register(page: Image.Image, boxes: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """(a, b) maximising ink in each box's middle half minus ink at its top and bottom edges."""
    ink = np.asarray(page.convert("L")) < 128
    rows = [ink[:, max(0, int(x0)):max(1, int(x1))].mean(axis=1) for x0, _, x1, _ in boxes]
    n = len(ink)

    def band(r, lo, hi):
        lo, hi = max(0, int(round(lo))), min(n, int(round(hi)))
        return r[lo:hi].mean() if hi > lo else 0.0

    def score(a, b):
        t = 0.0
        for r, (_, y0, _, y1) in zip(rows, boxes):
            y0, y1 = a * y0 + b, a * y1 + b
            h = y1 - y0
            t += band(r, y0 + h / 4, y1 - h / 4) - band(r, y0 - h / 8, y0 + h / 8) - band(r, y1 - h / 8, y1 + h / 8)
        return t

    # ponytail: grid search, ~2.5k scores a page; a coarse-to-fine search if pages get big.
    return max(((a, b) for a in np.arange(0.94, 1.0601, 0.0025) for b in range(-30, 31)), key=lambda ab: score(*ab))


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / "fixtures_khob"
    (out / "khob").mkdir(parents=True, exist_ok=True)
    rows = []
    for task in sorted(p for p in src.iterdir() if p.is_dir()):
        for ann_path in sorted(task.glob("*.json")):
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            # Some imagePath values are "../name.tiff"; the image sits next to its JSON.
            page = Image.open(ann_path.parent / Path(ann["imagePath"]).name).convert("RGB")
            # The 2026 release downscaled task-3/4 pages after labelling; scale points to the real size.
            sx, sy = page.width / ann["imageWidth"], page.height / ann["imageHeight"]
            pts = [[(x * sx, y * sy) for x, y in s["points"]] for s in ann["shapes"]]
            a, b = register(page, [(min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p),
                                    max(y for _, y in p)) for p in pts])
            print(task.name, ann_path.stem, f"y' = {a:.4f} y {b:+d}")
            for i, shape in enumerate(ann["shapes"]):
                text = clean_label(shape["label"])
                if not text:
                    continue
                xs = [x for x, _ in pts[i]]
                ys = [a * y + b for _, y in pts[i]]
                # ponytail: fixed margin of 15% line height; the paper's processed crops aren't published.
                m = 0.15 * (max(ys) - min(ys))
                box = (
                    max(0, int(min(xs) - m)),
                    max(0, int(min(ys) - m)),
                    min(page.width, int(max(xs) + m + 1)),
                    min(page.height, int(max(ys) + m + 1)),
                )
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                fid = f"khob-{task.name}-{ann_path.stem}-{i:03d}"
                page.crop(box).save(out / "khob" / f"{fid}.png")
                mixed = re.search(r"[A-Za-z0-9]", text) is not None
                rows.append({"id": fid, "image": f"khob/{fid}.png", "text": text,
                             "subset": "mixed" if mixed else "printed", "task": task.name})
    with (out / "labels.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} lines -> {out}")


if __name__ == "__main__":
    main()
