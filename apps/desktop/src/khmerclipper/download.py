"""Download pinned files with SHA-256 verification (tessdata, TrOCR models)."""

import hashlib
import os
import threading
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from khmerclipper import __version__
from khmerclipper.errors import Cancelled, KhmerClipperError


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    sha256: str
    size: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def is_present(files: Sequence[RemoteFile], dest_dir: Path) -> bool:
    # ponytail: size check only; files are SHA-verified before the rename that installs them,
    # so hashing ~670 MB again on every launch buys nothing.
    return all((dest_dir / f.name).is_file() and (dest_dir / f.name).stat().st_size == f.size for f in files)


def download(
    files: Sequence[RemoteFile],
    dest_dir: Path,
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    chunk_size: int = 1 << 20,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = sum(f.size for f in files)
    done = 0
    for f in files:
        final = dest_dir / f.name
        if final.is_file() and final.stat().st_size == f.size:
            done += f.size
            continue
        part = final.with_name(final.name + ".part")
        digest = hashlib.sha256()
        request = urllib.request.Request(f.url, headers={"User-Agent": f"KhmerClipper/{__version__}"})
        try:
            with urllib.request.urlopen(request, timeout=30) as resp, open(part, "wb") as out:
                while block := resp.read(chunk_size):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    digest.update(block)
                    out.write(block)
                    done += len(block)
                    if progress:
                        progress(done, total)
            if digest.hexdigest() != f.sha256:
                raise KhmerClipperError(f"Checksum mismatch for {f.name}; the download was corrupted. Try again.")
            os.replace(part, final)
        except (Cancelled, KhmerClipperError):
            raise
        except OSError as e:  # URLError, HTTPError, disk full, permission denied
            raise KhmerClipperError(f"Downloading {f.name} failed: {e}") from e
        finally:
            part.unlink(missing_ok=True)
