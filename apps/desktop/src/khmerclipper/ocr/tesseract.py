"""Tesseract subprocess engine (khm+eng), mirroring TesseractEngine.swift."""

import io
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from khmerclipper.errors import EngineUnavailable, KhmerClipperError
from khmerclipper.paths import resource_dir

LANGS = "khm+eng"
TIMEOUT_S = 60


def find_tesseract(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    if override := env.get("KHMERCLIPPER_TESSERACT"):
        return override
    exe = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    bundled = resource_dir() / "tesseract" / exe
    if bundled.is_file():
        return str(bundled)
    # Linux AppImage: AppRun puts the bundled binary first on PATH.
    return shutil.which("tesseract", path=env.get("PATH"))


def tessdata_dir() -> Path:
    return resource_dir() / "tesseract" / "tessdata"


def subprocess_env(env: Mapping[str, str], frozen: bool) -> dict[str, str]:
    """PyInstaller's Linux bootloader points LD_LIBRARY_PATH at its own bundled libs
    (saving the original in LD_LIBRARY_PATH_ORIG). Tesseract must not inherit that."""
    out = dict(env)
    if frozen:
        if "LD_LIBRARY_PATH_ORIG" in out:
            out["LD_LIBRARY_PATH"] = out["LD_LIBRARY_PATH_ORIG"]
        else:
            out.pop("LD_LIBRARY_PATH", None)
    return out


def ascii_path(path: Path) -> str:
    """Tesseract opens --tessdata-dir through the ANSI code page on Windows, so an
    install under a non-ASCII profile (e.g. a Khmer username) fails. Use the 8.3 short path."""
    s = str(path)
    if sys.platform != "win32" or s.isascii():
        return s
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, len(buf))
    return buf.value if 0 < n < len(buf) else s


def recognize(image: Image.Image) -> str:
    return _run(image)


def recognize_words(image: Image.Image) -> list[tuple[str, float]]:
    """A line's words, each with Tesseract's confidence (0-100), from its TSV output."""
    return parse_tsv(_run(image, ["-c", "tessedit_create_tsv=1"]))


def parse_tsv(tsv: str) -> list[tuple[str, float]]:
    """TSV columns: level page_num block_num par_num line_num word_num left top width height conf text."""
    words = []
    for row in tsv.splitlines()[1:]:
        f = row.split("\t")
        if len(f) > 11 and (word := "\t".join(f[11:]).strip()):
            words.append((word, float(f[10])))
    return words


def _run(image: Image.Image, extra_args: list[str] | None = None) -> str:
    exe = find_tesseract()
    if exe is None:
        raise EngineUnavailable("Tesseract was not found. Reinstall KhmerClipper (developers: install tesseract-ocr).")
    tessdata = tessdata_dir()
    if not (tessdata / "khm.traineddata").is_file():
        raise EngineUnavailable("Khmer language data is missing. Developers: run tools/fetch_tessdata.py.")

    png = io.BytesIO()
    image.convert("RGB").save(png, "PNG")
    cmd = [exe, "stdin", "stdout", "--tessdata-dir", ascii_path(tessdata), "-l", LANGS, "--psm", "6", *(extra_args or [])]
    try:
        proc = subprocess.run(
            cmd,
            input=png.getvalue(),
            capture_output=True,
            timeout=TIMEOUT_S,
            env=subprocess_env(os.environ, getattr(sys, "frozen", False)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console flash on Windows
        )
    except subprocess.TimeoutExpired as e:
        raise KhmerClipperError("Tesseract timed out.") from e
    except OSError as e:
        raise EngineUnavailable(f"Could not run Tesseract: {e}") from e
    if proc.returncode != 0:
        raise KhmerClipperError("Tesseract failed: " + proc.stderr.decode("utf-8", "replace").strip())
    return proc.stdout.decode("utf-8", "replace").strip()
