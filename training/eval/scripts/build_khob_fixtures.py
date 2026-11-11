"""Crop KHOB (EKYCSolutions/khmer-ocr-benchmark-dataset) LabelMe polygons into line fixtures.

Usage: python scripts/build_khob_fixtures.py <khob-dataset/level-1 dir> [out dir, default fixtures_khob]

Download the level zips from the Google Drive folder linked in the KHOB README first.
Each polygon becomes its axis-aligned bounding box plus a small margin. Labels write a
space as a literal "\\t"; that becomes a real space, zero-width spaces and stray
backslashes are dropped. Ids keep the task (task-1..4) so results can be split by it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image


def clean_label(label: str) -> str:
    text = label.replace("\\t", " ").replace("\\", "").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


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
            for i, shape in enumerate(ann["shapes"]):
                text = clean_label(shape["label"])
                if not text:
                    continue
                xs = [p[0] * sx for p in shape["points"]]
                ys = [p[1] * sy for p in shape["points"]]
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
