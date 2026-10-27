#!/usr/bin/env python3
"""Download the fonts used by the synthetic renderer into cache/fonts: the OFL Google Fonts Khmer + Latin
files, and the LGPL Khmer OS family from Debian's fonts-khmeros package (pinned by SHA-256).

    python scripts/fetch_fonts.py
"""
import argparse
import hashlib
import io
import lzma
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

from trocr_khmer.synth import ALL_FONTS, KHMER_OS_FONTS

BASE = "https://raw.githubusercontent.com/google/fonts/main/"
KHMEROS_DEB = "https://deb.debian.org/debian/pool/main/f/fonts-khmeros/fonts-khmeros_5.0-9_all.deb"
KHMEROS_SHA256 = "15bfe6ed4f08818b5d6ead58146a0280adeafd959c912d2a5069a4202713fc82"


def ar_members(data: bytes) -> dict[str, bytes]:
    """Members of a Debian .deb (an `ar` archive)."""
    assert data[:8] == b"!<arch>\n", "not an ar archive"
    out, pos = {}, 8
    while pos + 60 <= len(data):
        name, size = data[pos:pos + 16].decode().strip().rstrip("/"), int(data[pos + 48:pos + 58])
        out[name] = data[pos + 60:pos + 60 + size]
        pos += 60 + size + size % 2
    return out


def fetch_khmeros(out: Path) -> None:
    deb = urllib.request.urlopen(KHMEROS_DEB).read()
    if hashlib.sha256(deb).hexdigest() != KHMEROS_SHA256:
        raise SystemExit("fonts-khmeros .deb checksum mismatch")
    data = ar_members(deb)["data.tar.xz"]
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(data))) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and f"khmeros/{name}" in KHMER_OS_FONTS:
                (out / name).write_bytes(tar.extractfile(member).read())
                print(f"fetched {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("cache/fonts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for repo_path in ALL_FONTS:
        if repo_path in KHMER_OS_FONTS:
            continue
        dest = args.out / Path(repo_path).name
        if dest.is_file():
            continue
        urllib.request.urlretrieve(BASE + urllib.parse.quote(repo_path), dest)
        print(f"fetched {dest.name}")
    if not all((args.out / Path(p).name).is_file() for p in KHMER_OS_FONTS):
        fetch_khmeros(args.out)
    print(f"{len(ALL_FONTS)} fonts in {args.out}")


if __name__ == "__main__":
    main()
