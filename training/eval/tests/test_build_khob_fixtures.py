import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "build_khob_fixtures", Path(__file__).parents[1] / "scripts/build_khob_fixtures.py"
)
khob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(khob)


def test_register_undoes_boxes_drifting_down_the_page():
    # 30 lines, 14 px of ink every 22 px; boxes labelled 2% too tall, as on KHOB task-3 pages.
    px = np.full((700, 300), 255, np.uint8)
    lines = [(20 + 22 * i, 34 + 22 * i) for i in range(30)]
    for y0, y1 in lines:
        px[y0:y1, 20:280:3] = 0
    boxes = [(10, (y0 - 4) * 1.02 + 3, 290, (y1 + 4) * 1.02 + 3) for y0, y1 in lines]
    a, b = khob.register(Image.fromarray(px), boxes)
    for (_, y0, _, y1), (t0, t1) in zip(boxes, lines):
        assert abs((a * y0 + b + a * y1 + b) / 2 - (t0 + t1) / 2) <= 2
