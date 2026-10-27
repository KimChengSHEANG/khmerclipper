import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from helpers import FIXTURES
from khmerclipper.errors import Cancelled
from khmerclipper.ocr import trocr
from khmerclipper.ocr.tokenizer import TrOCRTokenizer

TOKENIZER = TrOCRTokenizer({"kind": "bpe", "bos_id": 0, "eos_id": 2, "pad_id": 1, "unk_id": 3,
                            "decoder_start_id": 0, "literal_ids": [4, 5],
                            "id_to_token": ["<s>", "<pad>", "</s>", "<unk>", "ក", "ខ"]})


class FakeEncoder:
    def run(self, names, feeds):
        b = feeds["pixel_values"].shape[0]
        assert feeds["pixel_values"].shape == (b, 3, 384, 384)
        cross = np.zeros((2, b, 2, 4, 3), np.float32)  # [L, B, H, S, D]
        return [cross, cross]


class ScriptedDecoder:
    """Emits script[position] as every row's next token (last entry repeats), growing the cache by one."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def run(self, names, feeds):
        self.calls += 1
        ids, pos, past = feeds["input_ids"], int(feeds["position"][0]), feeds["past_k"]
        b = feeds["cross_k"].shape[1]
        assert ids.dtype == np.int64 and ids.shape == (b, 1) and past.shape[1] == b
        assert past.shape[3] == pos  # one cached row per earlier token
        logits = np.zeros((b, 1, 6), np.float32)
        logits[:, -1, self.script[min(pos, len(self.script) - 1)]] = 1.0
        grown = np.zeros((*past.shape[:3], pos + 1, past.shape[4]), np.float32)
        return [logits, grown, grown]


def session(script, batch=trocr.BATCH):
    return SimpleNamespace(encoder=FakeEncoder(), decoder=ScriptedDecoder(script), tokenizer=TOKENIZER, batch=batch)


def lines_image(n):
    img = Image.new("RGB", (400, 100 * n), "white")
    for k in range(n):
        ImageDraw.Draw(img).rectangle((20, 100 * k + 30, 380, 100 * k + 60), fill="black")
    return img


def test_greedy_decode_until_eos():
    assert trocr.recognize_with_session(session([4, 5, 2]), lines_image(1)) == "កខ"


def test_each_line_decoded_and_joined_with_newline():
    s = session([4, 5, 2])
    assert trocr.recognize_with_session(s, lines_image(2)) == "កខ\nកខ"
    assert s.decoder.calls == 3  # both lines in one batch, in lockstep


def test_fixed_batch_model_decodes_one_line_at_a_time():
    s = session([4, 5, 2], batch=1)
    assert trocr.recognize_with_session(s, lines_image(2)) == "កខ\nកខ"
    assert s.decoder.calls == 6


def test_decode_stops_at_max_length_without_eos():
    s = session([4, 5] * trocr.MAX_LENGTH)
    assert trocr.recognize_with_session(s, lines_image(1)) == ("កខ" * trocr.MAX_LENGTH)[:trocr.MAX_LENGTH - 1]
    assert s.decoder.calls == trocr.MAX_LENGTH - 1


def test_blocked_token_caps_runs():
    # mirrors mac TrOCRDecodeGuardTests.testBlockedTokenCapsRuns
    assert trocr.blocked_token([0, 5, 5, 5]) is None
    assert trocr.blocked_token([0, 5, 5, 5, 5]) == 5
    assert trocr.blocked_token([0, 5, 6, 5, 6]) is None
    assert trocr.blocked_token([0]) is None


def test_run_cap_breaks_a_stuck_token():
    # decoder wants ក forever; every 5th step the cap forces the runner-up (<s>, decodes to "")
    out = trocr.recognize_with_session(session([4]), lines_image(1))
    assert out.count("ក") < trocr.MAX_LENGTH - 1 - 20


def test_cancel_raises():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        trocr.recognize_with_session(session([4, 2]), lines_image(1), cancel=cancel)


def test_progress_reports_token_count():
    seen = []
    trocr.recognize_with_session(session([4, 5, 2]), lines_image(1), progress=seen.append)
    assert seen[-1] == "Reading text… (4)"


def test_contains_khmer():
    assert not trocr.contains_khmer("Warranty 11 months")
    assert trocr.contains_khmer("ដក language")


@pytest.mark.parametrize("tess, expected", [
    (("Warranty 11 months", 95.0), "Warranty 11 months"),   # confident: Tesseract wins
    (("Warranty 11 rnonths", 40.0), "Warrantly 11 months"),  # unsure: keep TrOCR
    (("", 0.0), "Warrantly 11 months"),
])
def test_latin_fallback_uses_confident_tesseract(monkeypatch, tess, expected):
    from khmerclipper.ocr import tesseract
    monkeypatch.setattr(tesseract, "recognize_line", lambda img: tess)
    assert trocr.latin_fallback(Image.new("RGB", (10, 10)), "Warrantly 11 months") == expected


def test_latin_fallback_keeps_khmer_and_survives_missing_tesseract(monkeypatch):
    from khmerclipper.errors import EngineUnavailable
    from khmerclipper.ocr import tesseract

    def boom(img):
        raise EngineUnavailable("no tesseract")
    monkeypatch.setattr(tesseract, "recognize_line", boom)
    assert trocr.latin_fallback(Image.new("RGB", (10, 10)), "Price inbox") == "Price inbox"
    assert trocr.latin_fallback(Image.new("RGB", (10, 10)), "កខ") == "កខ"


@pytest.mark.skipif(not os.environ.get("KHMERCLIPPER_TROCR_DIR"), reason="set KHMERCLIPPER_TROCR_DIR to an exported model")
def test_real_model_reads_fixture():
    with Image.open(FIXTURES / "1.png") as img:
        text = trocr.recognize(img)
    assert text.strip()


def test_latin_fallback_gets_the_noise_free_crop(monkeypatch):
    from PIL import ImageDraw

    from khmerclipper.ocr import tesseract

    img = Image.new("RGB", (400, 60), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((6, 8, 50, 52), fill=(90, 120, 200))  # avatar left of an English line
    for x in range(120, 380, 14):
        d.rectangle((x, 22, x + 8, 38), fill="black")  # glyphs
    seen = []
    monkeypatch.setattr(tesseract, "recognize_line", lambda im: seen.append(im.size) or ("Hello", 99.0))
    assert trocr.latin_fallback(img, "Hel1o") == "Hello"
    assert seen and seen[0][0] < 330  # cropped to the text, avatar left out
    seen.clear()
    trocr.latin_fallback(img, "កខ")
    assert not seen  # Khmer lines never pay for the crop


def test_drop_stray_symbols_trims_icon_noise_at_line_ends():
    assert trocr.drop_stray_symbols("* ស្ដេច រាជរថ & >") == "ស្ដេច រាជរថ"
    assert trocr.drop_stray_symbols("ក = ខ") == "ក = ខ"  # inner symbols stay
    assert trocr.drop_stray_symbols("ខ្មែរ។ (១)") == "ខ្មែរ។ (១)"  # brackets and punctuation are text
    assert trocr.drop_stray_symbols(">") == ">"  # never empties a line
