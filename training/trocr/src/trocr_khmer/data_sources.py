"""Load and combine the Khmer OCR datasets we fine-tune / evaluate on.

Several interchangeable Hugging Face datasets are supported out of the box, each an
``image``/``text`` pair of printed or synthetic Khmer (some also English) lines. Train on one,
or combine several into a single set — they share the same column schema, so combining is just a
concatenation.

    python scripts/train.py --datasets soyvitou           # one dataset (the default)
    python scripts/train.py --datasets sokheng hanuman    # combined
    python scripts/train.py --datasets soyvitou sokheng hanuman mrrtmob kiteaether   # all

Registry aliases include ``handwriting``, ``handwriting_test``, ``darayut``, ``siemreap``,
``synth``, ``synth_val``, ``synth_en`` and ``synth_en_val``. An existing directory path loads as a local ``imagefolder``
(images plus ``metadata.jsonl`` with ``file_name`` and ``text``).

Any string that isn't a known alias is treated as a raw Hugging Face hub id (assumed to expose
``image``/``text`` columns), so new datasets work without a code change.

Note: ``mrrtmob`` (12M lines, Khmer+English) is a gated dataset — you must accept its terms on
the Hub and authenticate (``huggingface-cli login``) before it can be downloaded.
"""
from pathlib import Path
from typing import NamedTuple

from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets, load_dataset
from datasets import Image as DSImage

from trocr_khmer.dataset import MAX_LABEL_LEN

_SOYVITOU = "hf://datasets/SoyVitou/62k-images-khmer-printed-dataset"
_HANDWRITING = "hf://datasets/SoyVitou/khmer-handwritten-dataset-4.2k"

# Every source is normalized to exactly these columns/types so sources concatenate cleanly.
_FEATURES = Features({"image": DSImage(), "text": Value("string")})


class DatasetSpec(NamedTuple):
    """How to load one source. ``path`` is a hub id, or a builder name ("parquet",
    "imagefolder") when ``data_files`` / ``data_dir`` pin exactly which files to read."""

    path: str
    image_col: str = "image"
    text_col: str = "text"
    data_files: list[str] | None = None
    data_dir: str | None = None

    def load_kwargs(self) -> dict:
        return {k: v for k, v in (("data_files", self.data_files), ("data_dir", self.data_dir)) if v is not None}


DATASET_REGISTRY = {
    # The hub card lumps train/valid/test parquets into one "train" split. testset.parquet is
    # where training/eval/fixtures/printed comes from, so it must never be trained on.
    "soyvitou": DatasetSpec("parquet", data_files=[f"{_SOYVITOU}/trainset.parquet", f"{_SOYVITOU}/validset.parquet"]),
    "sokheng": DatasetSpec("Sokheng/khmer-synthetic-ocr-v1-100k"),
    "hanuman": DatasetSpec("seanghay/khmer-hanuman-100k"),
    "mrrtmob": DatasetSpec("mrrtmob/khmer_english_ocr_image_line"),  # gated: 12M Khmer+English lines
    "kiteaether": DatasetSpec("KiteAether/c_khmer_gemma_ocr_train"),
    # Real handwriting; testset.parquet is held out for evaluation (handwriting_test).
    "handwriting": DatasetSpec("parquet", data_files=[f"{_HANDWRITING}/trainset.parquet", f"{_HANDWRITING}/valset.parquet"]),
    "handwriting_test": DatasetSpec("parquet", data_files=[f"{_HANDWRITING}/testset.parquet"]),
    "darayut": DatasetSpec("Darayut/khmer-document-synthetic-low-res", text_col="label"),  # low-res document lines
    "siemreap": DatasetSpec("KimkosalYon/khmer-ocr-200k-siemreap-arial"),  # Siemreap/Arial font lines
    # Rendered locally by scripts/render_synth.py (paths relative to training/trocr).
    "synth": DatasetSpec("parquet", data_files=["cache/synth/train/*.parquet"]),
    "synth_val": DatasetSpec("parquet", data_files=["cache/synth/val/*.parquet"]),
    # Clean English lines in Latin fonts (render_synth.py --latin_only).
    "synth_en": DatasetSpec("parquet", data_files=["cache/synth_en/train/*.parquet"]),
    "synth_en_val": DatasetSpec("parquet", data_files=["cache/synth_en/val/*.parquet"]),
    # Screen strips cut from tight paragraphs at screen sizes (render_synth.py --screen, v3).
    "synth_screen": DatasetSpec("parquet", data_files=["cache/synth_screen/train/*/*.parquet"]),  # one dir per render process
    "synth_screen_val": DatasetSpec("parquet", data_files=["cache/synth_screen/val/*.parquet"]),
    # v4: screen strips aimed at v3's weak slices (small, dark, bold, ':' next to Khmer)
    "synth_screen_v4": DatasetSpec("parquet", data_files=["cache/synth_screen_v4/train/*/*.parquet"]),
    # LazyGreed/khmer_img2txt_200K (MIT) minus rows sharing text with eval fixtures (scripts/filter_source.py).
    "lazygreed": DatasetSpec("parquet", data_files=["cache/lazygreed/train/*.parquet"]),
}


# What the scripts train/eval on when --datasets isn't passed.
DEFAULT_DATASETS = ["soyvitou"]


def resolve_dataset(name: str) -> DatasetSpec:
    """Registry alias -> its spec; an existing directory -> a local imagefolder (images plus a
    ``metadata.jsonl`` with ``file_name`` and ``text``); anything else -> a raw hub id."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    if Path(name).is_dir():
        return DatasetSpec("imagefolder", data_dir=name)
    return DatasetSpec(name)


def _normalize_columns(ds: Dataset, image_col: str, text_col: str) -> Dataset:
    """Rename the image/text columns to the canonical names and drop everything else, so
    datasets with differing schemas concatenate cleanly."""
    rename = {}
    if image_col != "image":
        rename[image_col] = "image"
    if text_col != "text":
        rename[text_col] = "text"
    if rename:
        ds = ds.rename_columns(rename)
    extra = [c for c in ds.column_names if c not in ("image", "text")]
    if extra:
        ds = ds.remove_columns(extra)
    return ds


def _canonical(ds: Dataset) -> Dataset:
    """Cast to the shared features only when they differ. Call this on already-subsampled data:
    a cast rewrites every row, which is ruinous on the 12M-row mrrtmob."""
    return ds if ds.features == _FEATURES else ds.cast(_FEATURES)


def _load_one(spec: DatasetSpec, smoke_test: bool = False, smoke_samples: int = 200) -> Dataset:
    if smoke_test:
        stream = load_dataset(spec.path, split="train", streaming=True, **spec.load_kwargs())
        ds = Dataset.from_list(list(stream.take(smoke_samples)))
    else:
        ds = load_dataset(spec.path, split="train", **spec.load_kwargs())
    return _normalize_columns(ds, spec.image_col, spec.text_col)


def load_ocr_datasets(names, smoke_test: bool = False, smoke_samples: int = 200) -> Dataset:
    """Load every row of each named source and concatenate (no holdout carved).

    ``smoke_test`` streams only the first ``smoke_samples`` rows of each (no full download).
    """
    parts = []
    for name in names:
        spec = resolve_dataset(name)
        print(f"Loading dataset {name} ({spec.path})" + (f" [smoke: {smoke_samples}]" if smoke_test else ""))
        parts.append(_canonical(_load_one(spec, smoke_test, smoke_samples)))
    if len(parts) == 1:
        return parts[0]
    combined = concatenate_datasets(parts)
    print(f"Combined {len(names)} datasets -> {len(combined)} samples")
    return combined


def parse_source(entry: str) -> tuple[str, int | None]:
    """``"name:N"`` -> ``(name, N)`` to train on exactly N rows of that source per epoch;
    a plain name -> ``(name, None)`` for all rows."""
    name, sep, rows = entry.rpartition(":")
    if sep and rows.isdigit():
        return name, int(rows)
    return entry, None


def resample(ds: Dataset, n: int, seed: int) -> Dataset:
    """Deterministically take ``n`` rows: a shuffled subset when ``n < len(ds)``, whole shuffled
    copies (plus a remainder) when ``n > len(ds)`` — upsampling small sources like handwriting."""
    if len(ds) == 0:
        raise ValueError("cannot resample an empty dataset")
    shuffled = ds.shuffle(seed=seed)
    return shuffled.select([i % len(shuffled) for i in range(n)])


def filter_ocr_rows(ds: Dataset, tokenizer=None, max_label_len: int = MAX_LABEL_LEN,
                    max_aspect: float | None = None, num_proc: int | None = None) -> Dataset:
    """Drop rows the model can't learn from: labels longer than ``max_label_len`` tokens (they
    would be silently truncated, teaching early stops) and images wider than ``max_aspect``
    (unreadable after the 384x384 squash)."""
    if tokenizer is None and max_aspect is None:
        return ds

    def keep(example) -> bool:
        if tokenizer is not None and len(tokenizer(example["text"]).input_ids) > max_label_len:
            return False
        if max_aspect is not None:
            width, height = example["image"].size
            if height == 0 or width / height > max_aspect:
                return False
        return True

    return ds.filter(keep, num_proc=num_proc)


def load_ocr_split(names, test_size: float = 0.03, seed: int = 42,
                   smoke_test: bool = False, smoke_samples: int = 200,
                   max_eval_per_source: int | None = None, tokenizer=None,
                   max_label_len: int = MAX_LABEL_LEN, max_aspect: float | None = None,
                   num_proc: int | None = None) -> DatasetDict:
    """Per source: carve a seeded holdout, cap it, resample the train part (``name:N``), filter
    both, then concatenate across sources.

    Splitting before resampling keeps upsampled duplicates out of the holdout; per-source caps
    stop one huge source (mrrtmob) from being the whole eval set. Same names + seed -> same
    holdout, so ``scripts/evaluate_model.py`` reproduces training's eval rows.
    """
    trains, tests = [], []
    for entry in names:
        name, rows = parse_source(entry)
        spec = resolve_dataset(name)
        print(f"Loading dataset {name} ({spec.path})" + (f" [smoke: {smoke_samples}]" if smoke_test else ""))
        split = _load_one(spec, smoke_test, smoke_samples).train_test_split(test_size=test_size, seed=seed)
        train, test = split["train"], split["test"]
        if max_eval_per_source is not None:
            test = test.select(range(min(max_eval_per_source, len(test))))
        if rows is not None and not smoke_test:
            train = resample(train, rows, seed)
        train = _canonical(filter_ocr_rows(train, tokenizer, max_label_len, max_aspect, num_proc))
        test = _canonical(filter_ocr_rows(test, tokenizer, max_label_len, max_aspect, num_proc))
        print(f"  {name}: train {len(train)} / eval {len(test)}")
        trains.append(train)
        tests.append(test)
    return DatasetDict(train=concatenate_datasets(trains), test=concatenate_datasets(tests))
