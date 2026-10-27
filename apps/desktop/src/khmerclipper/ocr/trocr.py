"""TrOCR via onnxruntime: greedy KV-cache decode, a capture's text lines batched together (same
caps and Latin-line Tesseract fallback as apps/mac/Sources/OCR/TrOCREngine.swift)."""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from khmerclipper.errors import Cancelled, EngineUnavailable, KhmerClipperError
from khmerclipper.ocr import trocr_models
from khmerclipper.ocr.preprocess import content_crop, line_strips, pixel_values, split_wide
from khmerclipper.ocr.tokenizer import TrOCRTokenizer

# Total decoded ids incl. the start token; matches training MAX_LABEL_LEN and the mac
# TrOCREngine.maxLength (64 cut off long Khmer lines).
MAX_LENGTH = 128
# Greedy decode can lock onto one token; forbid extending a run past MAX_RUN identical ids
# (mac TrOCREngine.maxRun: greedy CER 0.172 -> 0.056). Labels never repeat a token > 3x.
MAX_RUN = 4
# Pieces decoded together in one batch (ONNX models exported with a dynamic batch axis).
BATCH = 4


def blocked_token(ids: list[int], max_run: int = MAX_RUN) -> int | None:
    tail = ids[-max_run:]
    return tail[-1] if len(ids) > max_run and len(set(tail)) == 1 else None

# Lowest Tesseract word confidence for it to replace a Latin-only TrOCR line.
# Measured (apps/mac TrOCREngine.tesseractMinConfidence): clean UI strips >= 75.9,
# all correct; scanned eval lines Tesseract gets wrong <= 62.7.
TESSERACT_MIN_CONFIDENCE = 70.0


def contains_khmer(text: str) -> bool:
    """Khmer (U+1780-17FF) or Khmer Symbols (U+19E0-19FF)."""
    return any("\u1780" <= c <= "\u17ff" or "\u19e0" <= c <= "\u19ff" for c in text)


def latin_fallback(strip: Image.Image, line: str) -> str:
    """The Khmer fine-tune misspells clean English UI text ("Warrantly", "PriceSnbox")
    that Tesseract reads perfectly, but Tesseract is worse on low-res scanned English.
    So a Latin-only line switches to Tesseract only when every word is confident.
    Lines with any Khmer stay with TrOCR. ponytail: one tesseract run per Latin-only line."""
    if contains_khmer(line):
        return line
    from khmerclipper.ocr import tesseract

    try:
        # The noise-free crop the model saw (avatars, borders left out); only for Latin lines.
        text, min_conf = tesseract.recognize_line(content_crop(strip))
    except KhmerClipperError:
        return line
    return text if text and min_conf >= TESSERACT_MIN_CONFIDENCE else line


@dataclass
class Session:
    encoder: Any
    decoder: Any
    tokenizer: TrOCRTokenizer
    batch: int = BATCH  # 1 for models exported with a fixed batch axis (khmer-v3-kv and older)


_cache: dict[Path, Session] = {}
_lock = threading.Lock()


def load_session(model_dir: Path) -> Session:
    with _lock:
        if model_dir not in _cache:
            import onnxruntime as ort

            try:
                encoder = ort.InferenceSession(str(model_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
                fixed = isinstance(encoder.get_inputs()[0].shape[0], int)
                _cache[model_dir] = Session(
                    encoder,
                    ort.InferenceSession(str(model_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]),
                    TrOCRTokenizer.from_file(model_dir / "tokenizer.json"),
                    1 if fixed else BATCH,
                )
            except Exception as e:  # onnxruntime raises several unrelated types for bad files
                raise EngineUnavailable(
                    f"TrOCR model files are unreadable ({e}). Delete {model_dir} and download again.") from e
        return _cache[model_dir]


def installed_model_dir() -> Path:
    if override := os.environ.get("KHMERCLIPPER_TROCR_DIR"):
        return Path(override)
    manifest = trocr_models.load_manifest()
    if not trocr_models.is_installed(manifest):
        raise EngineUnavailable("The TrOCR model isn't downloaded yet.")
    return trocr_models.model_dir(manifest)


def preload() -> None:
    """Warm the ONNX sessions while the user drags a selection. Errors surface on recognize()."""
    try:
        load_session(installed_model_dir())
    except KhmerClipperError:
        pass


def recognize(image: Image.Image, progress: Callable[[str], None] | None = None,
              cancel: threading.Event | None = None) -> str:
    if progress:
        progress("Loading model…")
    return recognize_with_session(load_session(installed_model_dir()), image, progress, cancel)


def recognize_with_session(session, image: Image.Image, progress: Callable[[str], None] | None = None,
                           cancel: threading.Event | None = None) -> str:
    tok = session.tokenizer
    strips = line_strips(image)
    # A long line squashed into 384x384 loses its thin marks; decode it in pieces.
    units = [(k, sep, piece) for k, strip in enumerate(strips) for sep, piece in split_wide(strip)]
    texts: list[str] = []
    done = 0
    # Up to BATCH pieces decode together, in lockstep: a batch-1 step on CPU is memory-bound (it
    # reads every weight for one token), so a batch costs about the same as one line.
    batch = getattr(session, "batch", BATCH)
    for b in range(0, len(units), batch):
        chunk = units[b:b + batch]
        # Cross-attention K/V once per piece; self-attention K/V carried step to step.
        cross_k, cross_v = session.encoder.run(None, {
            "pixel_values": np.concatenate([pixel_values(piece) for _, _, piece in chunk])})
        past_k = past_v = np.zeros((*cross_k.shape[:3], 0, cross_k.shape[4]), np.float32)
        ids = [[tok.decoder_start_id] for _ in chunk]
        live = [True] * len(chunk)
        for t in range(MAX_LENGTH - 1):
            if not any(live):
                break
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            logits, past_k, past_v = session.decoder.run(None, {
                "input_ids": np.array([[row[-1]] for row in ids], dtype=np.int64),
                "position": np.array([t], dtype=np.int64),
                "cross_k": cross_k, "cross_v": cross_v, "past_k": past_k, "past_v": past_v})
            for j, row in enumerate(ids):
                if not live[j]:
                    continue  # finished rows ride along; their output is ignored
                step = logits[j, -1].copy()
                if (blocked := blocked_token(row)) is not None:
                    step[blocked] = -np.inf
                row.append(int(np.argmax(step)))
                done += 1
                live[j] = row[-1] != tok.eos_id
            if progress:
                progress(f"Reading text… ({done + 1})")
        texts += [tok.decode(row).strip() for row in ids]
    lines = []
    for k, strip in enumerate(strips):
        text = ""
        for (unit_k, sep, _), piece_text in zip(units, texts):
            if unit_k == k and piece_text:
                text += (sep if text else "") + piece_text
        if line := latin_fallback(strip, text):
            lines.append(line)
    return "\n".join(lines)
