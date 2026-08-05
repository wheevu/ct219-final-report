"""Pipeline settings: defaults, modes, and CLI argument mapping."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

# Number of accepted documents targeted by each run mode.
MODE_DEFAULTS: dict[str, dict[str, int]] = {
    "smoke": {"max_documents": 2000, "scan_limit": 50_000},
    "development": {"max_documents": 20_000, "scan_limit": 500_000},
    "final": {"max_documents": 50_000, "scan_limit": None},
}

MODE_ALIASES = {"dev": "development", "devtest": "development"}


@dataclass(frozen=True)
class Settings:
    dataset: str = "VTSNLP/vietnamese_curated_dataset"
    dataset_revision: str | None = None
    dataset_split: str = "train"
    streaming: bool = True
    mode: str = "smoke"
    max_documents: int = 2000
    scan_limit: int | None = None  # None = unlimited; resolved from mode when unset
    seed: int = 42
    min_chars: int = 20
    max_chars: int | None = None
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    output_dir: str = "data/processed"
    overwrite: bool = False
    sample_size: int = 10
    text_field: str = "text"
    id_field: str = "id"
    domain_field: str = "domain"
    audit_dir: str = "data/audit"
    review_dir: str = "data/review"
    near_duplicates: bool = False
    simhash_threshold: int = 3
    shingle_size: int = 4
    chart: str = "docs/assets/domain_distribution.svg"
    dedup_backend: str = "memory"
    dedup_dir: str = "data/dedup"
    keep_dedup_db: bool = False
    rejected_sample_size: int = 20
    reject_encoding_corruption: bool = True
    reject_binary_invalid: bool = True
    reject_foreign_script: bool = False
    reject_concatenated: bool = False
    max_replacement_chars: int = 3
    min_unusual_unicode_count: int = 8
    min_unusual_unicode_ratio: float = 0.005
    min_non_latin_share: float = 0.3
    max_line_length: int = 20_000

    def resolve_mode(self) -> "Settings":
        """Apply mode defaults for values the user did not override explicitly."""
        mode_key = MODE_ALIASES.get(self.mode, self.mode)
        if mode_key not in MODE_DEFAULTS:
            raise ValueError(
                f"unknown mode {self.mode!r}; choose from "
                + ", ".join(sorted(MODE_DEFAULTS))
            )
        defaults = MODE_DEFAULTS[mode_key]
        values = asdict(self)
        values["mode"] = mode_key
        values["max_documents"] = self.max_documents or defaults["max_documents"]
        if self.scan_limit == 0:
            values["scan_limit"] = None  # 0 = unlimited
        elif self.scan_limit is None:
            values["scan_limit"] = defaults["scan_limit"]
        return Settings(**values)

    def effective(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.max_documents <= 0:
            raise ValueError("--max-documents must be positive")
        if self.min_chars < 0:
            raise ValueError("--min-chars must be non-negative")
        if self.max_chars is not None and self.max_chars < self.min_chars:
            raise ValueError("--max-chars must be >= --min-chars")
        if self.train_ratio <= 0 or self.val_ratio < 0:
            raise ValueError("split ratios must be positive")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train + validation ratios must be < 1.0")

    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.val_ratio, 6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.preprocess",
        description=(
            "Streaming preprocessing of raw Vietnamese text: clean, deduplicate, "
            "sample, split, and export model-ready plain text."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=Settings.dataset,
                        help="Hugging Face dataset name or path")
    parser.add_argument("--dataset-revision", default=Settings.dataset_revision,
                        help="pin a Hugging Face dataset revision (commit sha or tag)")
    parser.add_argument("--dataset-split", default=Settings.dataset_split,
                        help="dataset split to read")
    parser.add_argument("--streaming", action="store_true", default=Settings.streaming,
                        help="stream the dataset instead of loading it fully")
    parser.add_argument("--no-streaming", action="store_true",
                        help="load the dataset fully into memory")
    parser.add_argument("--mode", default=Settings.mode,
                        choices=sorted(MODE_DEFAULTS) + sorted(MODE_ALIASES),
                        help="run mode; sets default targets (overridable below)")
    parser.add_argument("--max-documents", type=int, default=0,
                        help="target number of accepted documents (default: mode-specific)")
    parser.add_argument("--scan-limit", type=int, default=None,
                        help="max source documents to inspect; default: mode-specific, 0 = unlimited")
    parser.add_argument("--seed", type=int, default=Settings.seed,
                        help="random seed for sampling and splitting")
    parser.add_argument("--min-chars", type=int, default=Settings.min_chars,
                        help="reject documents shorter than this many characters")
    parser.add_argument("--max-chars", type=int, default=Settings.max_chars,
                        help="optionally reject documents longer than this many characters")
    parser.add_argument("--train-ratio", type=float, default=Settings.train_ratio,
                        help="fraction of accepted documents for training")
    parser.add_argument("--val-ratio", type=float, default=Settings.val_ratio,
                        help="fraction of accepted documents for validation")
    parser.add_argument("--output-dir", default=Settings.output_dir,
                        help="directory for all pipeline outputs")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing outputs instead of failing")
    parser.add_argument("--sample-size", type=int, default=Settings.sample_size,
                        help="number of before/after sample pairs to export")
    parser.add_argument("--dedup-backend", default=Settings.dedup_backend,
                        choices=("memory", "sqlite"),
                        help="exact-deduplication backend; sqlite scales to full scans")
    parser.add_argument("--dedup-dir", default=Settings.dedup_dir,
                        help="working directory for the sqlite dedup database")
    parser.add_argument("--keep-dedup-db", action="store_true",
                        help="keep the sqlite dedup database after a successful run")
    parser.add_argument("--rejected-sample-size", type=int, default=Settings.rejected_sample_size,
                        help="max rejected examples recorded per reason")
    parser.add_argument("--reject-encoding-corruption",
                        action=argparse.BooleanOptionalAction,
                        default=Settings.reject_encoding_corruption,
                        help="reject documents with replacement chars / VNI mojibake")
    parser.add_argument("--reject-binary-invalid",
                        action=argparse.BooleanOptionalAction,
                        default=Settings.reject_binary_invalid,
                        help="reject documents with control-format chars or heavy replacement soup")
    parser.add_argument("--reject-foreign-script",
                        action=argparse.BooleanOptionalAction,
                        default=Settings.reject_foreign_script,
                        help="reject documents dominated by a non-Latin script (audit-only by default)")
    parser.add_argument("--reject-concatenated",
                        action=argparse.BooleanOptionalAction,
                        default=Settings.reject_concatenated,
                        help="reject page-dump documents with a single enormous line (audit-only by default)")
    parser.add_argument("--max-replacement-chars", type=int, default=Settings.max_replacement_chars,
                        help="replacement chars before encoding_corruption triggers")
    parser.add_argument("--min-unusual-unicode-count", type=int,
                        default=Settings.min_unusual_unicode_count,
                        help="unusual Unicode char count floor before binary_or_invalid_content triggers")
    parser.add_argument("--min-unusual-unicode-ratio", type=float,
                        default=Settings.min_unusual_unicode_ratio,
                        help="unusual Unicode share before binary_or_invalid_content triggers")
    parser.add_argument("--min-non-latin-share", type=float, default=Settings.min_non_latin_share,
                        help="non-Latin script share before foreign_script_dominant triggers")
    parser.add_argument("--max-line-length", type=int, default=Settings.max_line_length,
                        help="max line length before concatenated_dump triggers")
    return parser


def settings_from_args(argv: list[str] | None = None) -> Settings:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings(
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        dataset_split=args.dataset_split,
        streaming=not args.no_streaming,
        mode=args.mode,
        max_documents=args.max_documents,
        scan_limit=args.scan_limit,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        sample_size=args.sample_size,
        dedup_backend=args.dedup_backend,
        dedup_dir=args.dedup_dir,
        keep_dedup_db=args.keep_dedup_db,
        rejected_sample_size=args.rejected_sample_size,
        reject_encoding_corruption=args.reject_encoding_corruption,
        reject_binary_invalid=args.reject_binary_invalid,
        reject_foreign_script=args.reject_foreign_script,
        reject_concatenated=args.reject_concatenated,
        max_replacement_chars=args.max_replacement_chars,
        min_unusual_unicode_count=args.min_unusual_unicode_count,
        min_unusual_unicode_ratio=args.min_unusual_unicode_ratio,
        min_non_latin_share=args.min_non_latin_share,
        max_line_length=args.max_line_length,
    )
    settings = settings.resolve_mode()
    settings.validate()
    return settings
