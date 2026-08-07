"""Streaming preprocessing pipeline: clean, deduplicate, sample, split, export.

Run as:

    python -m src.data.preprocess --mode smoke --max-documents 2000 --seed 42
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .cleaning import ALL_REASONS, inspect_document
from .config import Settings, settings_from_args
from .dedup import open_dedup
from .export import (
    RELEASE_COMPLETE,
    AtomicTextWriter,
    ParquetSplitWriter,
    SPLIT_NAMES,
    atomic_write_bytes,
    expected_output_files,
    sweep_stale_temp_files,
    truncate_for_display,
    write_csv,
    write_json,
)
from .manifest import (
    build_manifest,
    resolve_dataset_revision,
    write_manifest,
)
from .quality_rules import rejection_decision, rule_checks, rule_signals
from .sampling import DeterministicReservoir, assign_split, stable_hash
from .stats import DomainTracker, LengthStats, RemovalCounters


def peak_memory_mb() -> float | None:
    """Approximate peak RSS in MiB (ru_maxrss units differ by platform)."""
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ValueError, OSError):
        return None
    return round(value / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def _record_rejected(rejected_examples: dict[str, list[dict]], reason: str,
                     row: dict, cleaned: str, settings: Settings) -> None:
    """Keep a capped per-reason record of rejected documents for the audit."""
    bucket = rejected_examples[reason]
    if len(bucket) >= settings.rejected_sample_size:
        return
    bucket.append({
        "id": str(row.get(settings.id_field, "missing")),
        "domain": str(row.get(settings.domain_field, "unknown")),
        "reason": reason,
        "character_count": len(cleaned),
        "preview": truncate_for_display(cleaned),
    })


def _write_rejected_examples(path: Path, rejected_examples: dict[str, list[dict]]) -> None:
    rows = [
        [ex["id"], ex["domain"], ex["reason"], ex["character_count"], ex["preview"]]
        for reason in sorted(rejected_examples)
        for ex in rejected_examples[reason]
    ]
    write_csv(path, ["id", "domain", "reason", "character_count", "preview"], rows)


def load_documents(settings: Settings) -> Iterable[dict]:
    """Load the dataset iterable (streaming by default).

    Two loaders:
    - "datasets": Hugging Face `datasets` streaming (default).
    - "shards": shard-by-shard download/process/delete for very large
      datasets that do not fit on disk; requires a pinned revision.
    """
    if settings.loader == "shards":
        from .shard_loader import yield_rows

        return yield_rows(
            settings.dataset,
            settings.dataset_revision,
            Path(settings.cache_dir),
        )
    try:
        from datasets import load_dataset

        return load_dataset(
            settings.dataset,
            split=settings.dataset_split,
            streaming=settings.streaming,
            revision=settings.dataset_revision,
        )
    except Exception as exc:  # network errors, missing dataset, bad config
        raise RuntimeError(
            f"failed to load dataset {settings.dataset!r} "
            f"(split={settings.dataset_split!r}, streaming={settings.streaming}, "
            f"revision={settings.dataset_revision!r}): {exc}"
        ) from exc


def run_pipeline(settings: Settings) -> dict:
    settings = settings.resolve_mode()
    settings.validate()
    output_dir = Path(settings.output_dir)
    processed_dir = output_dir / "processed"
    stats_dir = output_dir / "statistics"
    samples_dir = output_dir / "samples"

    existing = [p for p in expected_output_files(output_dir) if p.exists()]
    if existing and not settings.overwrite:
        raise FileExistsError(
            "outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(p) for p in existing[:5])
        )
    for directory in (processed_dir, stats_dir, samples_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # A rerun starts from a clean slate: remove the previous completion marker
    # so an interrupted rerun can never look complete.
    marker = output_dir / RELEASE_COMPLETE
    if marker.exists():
        marker.unlink()
    sweep_stale_temp_files(processed_dir, stats_dir, samples_dir)

    started = time.time()
    resolved_revision = resolve_dataset_revision(
        settings.dataset, settings.dataset_revision
    )
    print(f"scanning {settings.dataset!r} (split={settings.dataset_split!r}, "
          f"streaming={settings.streaming}, revision={resolved_revision or 'unresolved'}, "
          f"dedup={settings.dedup_backend})")

    docs = load_documents(settings)
    reservoir = DeterministicReservoir(settings.max_documents, settings.seed)
    counters = RemovalCounters()
    lengths = LengthStats()
    domains = DomainTracker()
    dedup = open_dedup(settings.dedup_backend, settings.seed,
                       Path(settings.dedup_dir), keep=settings.keep_dedup_db)
    rule_policy = {
        "encoding_corruption": settings.reject_encoding_corruption,
        "binary_or_invalid_content": settings.reject_binary_invalid,
        "foreign_script_dominant": settings.reject_foreign_script,
        "concatenated_dump": settings.reject_concatenated,
    }
    rule_thresholds = {
        "max_replacement_chars": settings.max_replacement_chars,
        "min_unusual_unicode_count": settings.min_unusual_unicode_count,
        "min_unusual_unicode_ratio": settings.min_unusual_unicode_ratio,
        "min_non_latin_share": settings.min_non_latin_share,
        "max_line_length": settings.max_line_length,
    }
    samples: list[dict] = []
    rejected_examples: dict[str, list[dict]] = {reason: [] for reason in ALL_REASONS}

    scanned = 0
    candidates_seen = 0
    for index, row in enumerate(tqdm(docs, desc="scanning", unit="docs")):
        if settings.scan_limit is not None and index >= settings.scan_limit:
            break
        scanned = index + 1

        raw_text = row.get(settings.text_field)
        domains.record_inspected(str(row.get(settings.domain_field, "unknown")))
        cleaned, reason = inspect_document(raw_text, settings.min_chars, settings.max_chars)
        if reason is not None:
            counters[reason] += 1
            _record_rejected(rejected_examples, reason, row, cleaned, settings)
            continue

        doc_id = str(row.get(settings.id_field, f"missing_{index}"))
        if any(rule_policy.values()):
            checks = rule_checks(cleaned, rule_signals(cleaned), rule_policy, rule_thresholds)
            rule_reason = rejection_decision(checks, rule_policy)
            if rule_reason is not None:
                counters[rule_reason] += 1
                _record_rejected(rejected_examples, rule_reason, row, cleaned, settings)
                continue

        dup_reason = dedup.check_and_add(doc_id, cleaned)
        if dup_reason is not None:
            counters[dup_reason] += 1
            continue
        domain = str(row.get(settings.domain_field, "unknown"))
        domains.record_accepted(domain)
        item = {"id": doc_id, "domain": domain, "text": cleaned, "text_hash": stable_hash(cleaned)}
        candidates_seen += 1
        retained = reservoir.offer(item)
        if retained and len(samples) < settings.sample_size:
            samples.append({
                "id": doc_id,
                "domain": domain,
                "before": raw_text,
                "after": cleaned,
            })

    accepted = len(reservoir.items)
    if accepted < settings.max_documents:
        print(f"warning: only {accepted} documents accepted "
              f"(target was {settings.max_documents})")

    # Assign splits and stream accepted documents to output writers.
    parquet_writer = ParquetSplitWriter(processed_dir)
    txt_writers = {
        name: AtomicTextWriter(processed_dir / f"{name}.txt")
        for name in SPLIT_NAMES
    }
    split_counts = {name: 0 for name in SPLIT_NAMES}
    for item in reservoir.items:
        split = assign_split(
            item["id"], settings.seed, settings.train_ratio, settings.val_ratio
        )
        split_counts[split] += 1
        lengths.add(len(item["text"]))
        parquet_writer.add(split, {
            "id": item["id"],
            "domain": item["domain"],
            "text": item["text"],
            "text_hash": item["text_hash"],
            "character_count": len(item["text"]),
        })
        txt_writers[split].write_line(json.dumps(item["text"], ensure_ascii=False))

    parquet_writer.commit()
    for writer in txt_writers.values():
        writer.commit()

    elapsed = round(time.time() - started, 1)
    removal = counters.snapshot()
    dedup_stats = dedup.stats()
    dedup.close()
    # ------------------------- statistics outputs -------------------------
    stats = {
        "config": settings.effective(),
        "resolved_dataset_revision": resolved_revision,
        "source_documents_inspected": scanned,
        "accepted_candidates": candidates_seen,
        "accepted_documents": accepted,
        "requested_documents": settings.max_documents,
        "rejected_documents": removal["total"],
        "sampled_out": candidates_seen - accepted,
        "removal_counts": {k: v for k, v in removal.items() if k != "total"},
        "duplicate_count": removal["duplicate_id"] + removal["duplicate_text"],
        "split_counts": split_counts,
        "length_statistics": lengths.summary(),
        "elapsed_seconds": elapsed,
        "performance": {
            "source_documents_per_second": round(scanned / elapsed, 1) if elapsed else None,
            "accepted_documents_per_second": round(accepted / elapsed, 1) if elapsed else None,
            "peak_memory_mb": peak_memory_mb(),
        },
        "dedup": dedup_stats,
    }
    write_json(stats_dir / "preprocessing_stats.json", stats)
    write_json(stats_dir / "length_statistics.json", lengths.summary())
    write_csv(stats_dir / "removal_counts.csv",
              ["reason", "count"],
              [[reason, removal[reason]] for reason in sorted(removal) if reason != "total"])
    write_csv(stats_dir / "domain_distribution.csv",
              ["domain", "inspected", "accepted"],
              [[r["domain"], r["inspected"], r["accepted"]] for r in domains.rows()])
    write_csv(samples_dir / "before_after_examples.csv",
              ["id", "domain", "before_chars", "after_chars", "before", "after"],
              [[s["id"], s["domain"], len(s["before"]), len(s["after"]),
                truncate_for_display(s["before"]), truncate_for_display(s["after"])]
               for s in samples])
    _write_rejected_examples(samples_dir / "rejected_examples.csv", rejected_examples)

    # --------------------------- manifest ---------------------------------
    manifest = build_manifest(
        output_dir=output_dir,
        settings=settings,
        run_stats=stats,
        resolved_revision=resolved_revision,
        output_paths=[
            processed_dir / f"{split}.{ext}"
            for split in SPLIT_NAMES for ext in ("parquet", "txt")
        ] + [
            stats_dir / "preprocessing_stats.json",
            stats_dir / "length_statistics.json",
            stats_dir / "removal_counts.csv",
            stats_dir / "domain_distribution.csv",
            samples_dir / "before_after_examples.csv",
            samples_dir / "rejected_examples.csv",
        ],
        row_counts={
            f"{split}.parquet": parquet_writer.counts[split]
            for split in SPLIT_NAMES
        },
        runtime_seconds=elapsed,
        docs_per_second=stats["performance"]["source_documents_per_second"],
        accepted_per_second=stats["performance"]["accepted_documents_per_second"],
        peak_memory_mb=stats["performance"]["peak_memory_mb"],
        hashes_retained=dedup_stats["hashes_retained"],
    )
    write_manifest(manifest, output_dir)
    # Completion marker: written last, so a release is only "complete" when
    # every output AND the manifest are on disk. An interrupted run leaves
    # outputs without the marker and is detected as incomplete.
    atomic_write_bytes(marker, b"complete\n")

    # ------------------------------ summary -------------------------------
    print(f"\nscanning finished: {scanned} source documents inspected, "
          f"{accepted} accepted in {elapsed}s")
    perf = stats["performance"]
    print(f"  throughput: {perf['source_documents_per_second']} docs/s source, "
          f"{perf['accepted_documents_per_second']} docs/s accepted; "
          f"peak RSS ~{perf['peak_memory_mb']} MiB; "
          f"dedup: {dedup_stats['backend']} "
          f"({dedup_stats['hashes_retained']} hashes retained)")
    print(f"  dataset revision: {resolved_revision or 'unresolved'}")
    print(f"  quality/dedup rejections: {removal['total']} "
          f"(across {candidates_seen} accepted candidates)")
    if candidates_seen - accepted:
        print(f"  sampled out by reservoir: {candidates_seen - accepted}")
    for reason in sorted(removal):
        if reason != "total" and removal[reason]:
            print(f"  rejected [{reason}]: {removal[reason]}")
    print(f"  split counts: {split_counts}")
    print(f"  outputs written under {output_dir}")
    return stats


def main(argv: list[str] | None = None) -> int:
    settings = settings_from_args(argv)
    try:
        run_pipeline(settings)
    except (FileExistsError, RuntimeError, ValueError, OSError) as exc:
        # OSError covers disk-full and interrupted writes; report clearly.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
