"""Safe, streaming output writers and helpers.

Writers buffer data and write through a temporary file that is atomically
renamed into place on commit, so interrupted runs never leave half-written
outputs behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("text", pa.string()),
    pa.field("text_hash", pa.string()),
    pa.field("character_count", pa.int64()),
])

SPLIT_NAMES = ("train", "validation", "test")

# Name of the completion marker written last by a successful run. A release
# without this file (or with a manifest whose status is not "complete") is
# treated as interrupted and never served as valid.
RELEASE_COMPLETE = "RELEASE_COMPLETE"


def release_status(output_dir: Path) -> str:
    """'complete', 'incomplete', or 'missing' for a release directory.

    complete   -> marker file exists and manifest status is "complete".
    incomplete -> outputs or manifest exist without a valid marker.
    missing    -> nothing was written yet.
    """
    marker = output_dir / RELEASE_COMPLETE
    manifest_path = output_dir / "manifest.json"
    if marker.exists() and manifest_path.exists():
        try:
            import json

            status = json.loads(manifest_path.read_text("utf-8")).get("status")
        except (OSError, ValueError):
            status = None
        if status == "complete":
            return "complete"
    has_outputs = any(p.exists() for p in expected_output_files(output_dir))
    if has_outputs or manifest_path.exists() or marker.exists():
        return "incomplete"
    return "missing"


def truncate_for_display(text: str, limit: int = 200) -> str:
    """Truncate a sample for display without touching the exported data."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…[truncated]"


class AtomicTextWriter:
    """Appends lines to a temp file; commit() atomically renames it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        fd, self._tmp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        self._file = os.fdopen(fd, "w", encoding="utf-8")
        self.count = 0

    def write_line(self, line: str) -> None:
        self._file.write(line + "\n")
        self.count += 1

    def commit(self) -> None:
        self._file.close()
        os.chmod(self._tmp, 0o644)
        os.replace(self._tmp, self.path)

    def abort(self) -> None:
        self._file.close()
        if os.path.exists(self._tmp):
            os.remove(self._tmp)


class ParquetSplitWriter:
    """Incremental parquet writer per split using pyarrow's ParquetWriter."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._batches: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
        self._flush_at = 1024
        self.counts: dict[str, int] = {name: 0 for name in SPLIT_NAMES}

    def add(self, split: str, row: dict) -> None:
        self._batches[split].append(row)
        if len(self._batches[split]) >= self._flush_at:
            self._flush(split)

    def _writer(self, split: str) -> pq.ParquetWriter:
        if split not in self._writers:
            self._writers[split] = pq.ParquetWriter(
                self.directory / f"{split}.parquet", PARQUET_SCHEMA
            )
        return self._writers[split]

    def _flush(self, split: str) -> None:
        rows = self._batches[split]
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        self._writer(split).write_table(table)
        self.counts[split] += len(rows)
        self._batches[split] = []

    def commit(self) -> None:
        for split in SPLIT_NAMES:
            self._flush(split)
            if split not in self._writers:
                # Write an empty table so every split file always exists.
                self._writer(split).write_table(
                    pa.Table.from_pylist([], schema=PARQUET_SCHEMA)
                )
        for writer in self._writers.values():
            writer.close()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "wb") as fh:
        fh.write(content)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def write_csv(path: Path, header: list[str], rows: list[list[str | int]]) -> None:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def expected_output_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "processed" / f"{split}.parquet" for split in SPLIT_NAMES
    ] + [
        output_dir / "processed" / f"{split}.txt" for split in SPLIT_NAMES
    ] + [
        output_dir / "statistics" / "preprocessing_stats.json",
        output_dir / "statistics" / "domain_distribution.csv",
        output_dir / "statistics" / "removal_counts.csv",
        output_dir / "statistics" / "length_statistics.json",
        output_dir / "samples" / "before_after_examples.csv",
        output_dir / "samples" / "rejected_examples.csv",
        output_dir / RELEASE_COMPLETE,
    ]


def sweep_stale_temp_files(*directories: Path) -> int:
    """Remove leftover writer temp files ('.<name>.<rand>.tmp') from crashed runs."""
    removed = 0
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob(".*.tmp"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed
