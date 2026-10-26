#!/usr/bin/env python3
"""Pre-render synthetic Khmer line images to parquet shards (deterministic per --seed).

    python scripts/fetch_fonts.py
    python scripts/render_synth.py --part val --count 3000 --seed 2 --out cache/synth/val
    python scripts/render_synth.py --part train --count 400000 --seed 1 --out cache/synth/train

Text comes from existing datasets' labels, chunked to <= --max_chars, minus every eval fixture
and UI string. --part picks the train or val side of a stable content-hash split.

Screen strips (app line strips from tight paragraphs, small sizes; v3):

    for i in 1 2 3 4 5 6; do python scripts/render_synth.py --screen --max_chars 64 --part train --count 34000 --seed 4$i --out cache/synth_screen/train/p$i & done

Clean English set (Latin fonts, capitalised UI-style lines; eval: training/eval/en_ui_strings.txt):

    python scripts/render_synth.py --latin_only --english_texts 300000 --text_sources soyvitou mrrtmob \
        --part train --count 150000 --seed 3 --out cache/synth_en/train
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from datasets import Image as DSImage

from trocr_khmer.data_sources import resolve_dataset
from trocr_khmer.synth import (augment_case, build_corpus, corpus_part, is_latin_only, load_fonts, render_sample,
                               render_screen_sample, require_raqm)

EVAL = Path(__file__).resolve().parents[2] / "eval"
FEATURES = Features(
    {
        "image": DSImage(),
        "text": Value("string"),
        "font": Value("string"),
        "kind": Value("string"),
        "style": Value("string"),
        "italic": Value("bool"),
    }
)


def load_texts(name: str, k: int, seed: int) -> list[str]:
    """k random labels from a source's *training* side — text column only, images never decoded.

    Drops the same seeded 3% holdout that load_ocr_split carves (test_size=0.03, seed=42), so
    replay-holdout labels never reach training re-rendered in other fonts."""
    spec = resolve_dataset(name)
    ds = load_dataset(spec.path, split="train", **spec.load_kwargs()).select_columns([spec.text_col])
    ds = ds.train_test_split(test_size=0.03, seed=42)["train"]
    idx = random.Random(seed).sample(range(len(ds)), min(k, len(ds)))
    return ds.select(idx)[spec.text_col]


def english_texts(k: int) -> list[str]:
    """First k English Wikipedia paragraphs (streamed; normal punctuation spacing)."""
    out: list[str] = []
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for article in ds:
        out += [p.strip() for p in article["text"].split("\n") if len(p.strip()) > 20]
        if len(out) >= k:
            return out[:k]
    return out


def eval_texts() -> list[str]:
    labels = (EVAL / "fixtures/labels.jsonl").read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in labels if line.strip()]
    for name in ("ui_strings.txt", "en_ui_strings.txt"):
        texts += (EVAL / name).read_text(encoding="utf-8").splitlines()
    return texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--part", choices=["train", "val"], default="train")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--font_dir", type=Path, default=Path("cache/fonts"))
    p.add_argument("--text_sources", nargs="+", default=["hanuman", "sokheng", "soyvitou", "mrrtmob"])
    p.add_argument("--texts_per_source", type=int, default=200_000)
    p.add_argument("--max_chars", type=int, default=32)
    p.add_argument("--max_aspect", type=float, default=20.0)
    p.add_argument("--shard_size", type=int, default=10_000)
    p.add_argument("--english_texts", type=int, default=0, help="add N English Wikipedia paragraphs")
    p.add_argument("--latin_only", action="store_true", help="keep only Latin-only lines; vary their case")
    p.add_argument("--screen", action="store_true",
                   help="screen strips: 11-24px (1x/2x) text fonts cut from a tight paragraph (render_screen_sample)")
    p.add_argument("--screen_sizes", type=int, nargs=2, default=(11, 24), help="--screen px range")
    p.add_argument("--screen_dark", type=float, default=0.3, help="--screen share of dark panels")
    p.add_argument("--screen_bold", type=float, default=0.0, help="--screen share drawn bold")
    p.add_argument("--colon_prefix", type=float, default=0.0,
                   help="share of lines prefixed with a time or number and ':' (v3 read ':' as Khmer U+17C8)")
    args = p.parse_args()

    require_raqm()
    fonts = load_fonts(args.font_dir)
    if args.latin_only:
        # Khmer fonts' Latin glyphs include caps-only display faces (Bayon, Koulen) whose
        # pixels contradict the label's case; keep English in real Latin fonts.
        fonts = [f for f in fonts if f.kind == "latin"]
    texts = [t for name in args.text_sources for t in load_texts(name, args.texts_per_source, args.seed)]
    if args.english_texts:
        texts += english_texts(args.english_texts)
    corpus = [c for c in build_corpus(texts, eval_texts(), args.max_chars) if corpus_part(c) == args.part]
    if args.latin_only:
        corpus = [c for c in corpus if is_latin_only(c)]
    if not corpus:
        raise SystemExit("empty corpus")
    rng = random.Random(args.seed)
    rng.shuffle(corpus)
    print(f"{len(corpus)} {args.part} lines from {len(texts)} labels; rendering {args.count}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows, shard, made, tried = defaultdict(list), 0, 0, 0
    while made < args.count:
        if tried > 20 * args.count:
            raise SystemExit(f"only {made} renderable lines after {tried} tries; check fonts/corpus")
        text = corpus[tried % len(corpus)]
        tried += 1
        if args.latin_only:
            text = augment_case(text, rng)
        if args.colon_prefix and rng.random() < args.colon_prefix:
            text = colon_prefix(rng) + text
        if args.screen:
            # Neighbour lines exist ~75% of the time (strips at a clip's top / bottom edge have none).
            near = tuple(corpus[rng.randrange(len(corpus))] if rng.random() < 0.75 else None for _ in range(2))
            sample = render_screen_sample(text, near, fonts, rng, sizes=tuple(args.screen_sizes),
                                          dark=args.screen_dark, bold=args.screen_bold)
        else:
            sample = render_sample(text, fonts, rng)
        if sample is None:
            continue
        image, meta = sample
        if image.width / image.height > args.max_aspect:
            continue
        rows["image"].append(image)
        rows["text"].append(text)
        for key in ("font", "kind", "style", "italic"):
            rows[key].append(meta[key])
        made += 1
        if len(rows["text"]) == args.shard_size or made == args.count:
            Dataset.from_dict(rows, features=FEATURES).to_parquet(str(args.out / f"shard-{shard:05d}.parquet"))
            print(f"  shard {shard}: {made}/{args.count}")
            shard += 1
            rows = defaultdict(list)


def colon_prefix(rng: random.Random) -> str:
    """'08:30 ', '1: ' or '12:05:' style lead-ins: Latin colons next to Khmer text."""
    kind = rng.randrange(3)
    if kind == 0:
        return f"{rng.randrange(24):02d}:{rng.randrange(60):02d} "
    if kind == 1:
        return f"{rng.randrange(1, 100)}: "
    return f"{rng.randrange(1, 13)}:{rng.randrange(60):02d}:"


if __name__ == "__main__":
    main()
