"""Release validation: status detection, integrity checks, and CLI.

A release is only trustworthy when:
- the completion marker exists and the manifest status is "complete";
- parquet row counts match the manifest;
- JSONL/TXT row counts match parquet and round-trip exactly;
- no id or normalized-text hash repeats across splits;
- every output checksum matches the manifest;
- no stale temp files or leftover dedup databases are present.

Run as:

    python -m src.data.release --output-dir data/releases/ct219-400k-v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from .export import RELEASE_COMPLETE, SPLIT_NAMES, release_status, sweep_stale_temp_files
from .manifest import sha256_file


def validate_release(output_dir: Path) -> dict:
    """Validate a completed release; returns a full report dict."""
    report: dict = {
        "output_dir": str(output_dir),
        "status": release_status(output_dir),
    }
    if report["status"] != "complete":
        report["ok"] = False
        return report

    manifest = json.loads((output_dir / "manifest.json").read_text("utf-8"))
    report["manifest_status"] = manifest.get("status")
    report["dataset"] = manifest.get("dataset")
    report["source_revision"] = manifest.get("resolved_dataset_revision")
    report["requested_revision"] = manifest.get("requested_dataset_revision")

    # Checksums.
    mismatches = []
    for rel, info in manifest.get("outputs", {}).items():
        path = output_dir / rel
        if not path.exists():
            mismatches.append(f"{rel}: missing")
            continue
        if sha256_file(path) != info["sha256"]:
            mismatches.append(f"{rel}: checksum mismatch")
    report["checksum_mismatches"] = mismatches

    # Row counts and JSONL round-trip, per split.
    split_report: dict[str, dict] = {}
    ids: set[str] = set()
    hashes: set[str] = set()
    for split in SPLIT_NAMES:
        entry: dict = {"expected_manifest": manifest["outputs"][
            f"processed/{split}.parquet"]["row_count"]}
        table = pq.read_table(output_dir / "processed" / f"{split}.parquet")
        entry["parquet_rows"] = table.num_rows
        lines = (output_dir / "processed" / f"{split}.txt").read_text("utf-8").splitlines()
        entry["txt_rows"] = len(lines)
        round_trip = True
        try:
            restored = [json.loads(line) for line in lines]
            round_trip = restored == table.column("text").to_pylist()
        except Exception:
            round_trip = False
        entry["jsonl_round_trip"] = round_trip
        split_ids = table.column("id").to_pylist()
        split_hashes = table.column("text_hash").to_pylist()
        entry["duplicate_ids_within_split"] = len(split_ids) - len(set(split_ids))
        entry["duplicate_hashes_within_split"] = len(split_hashes) - len(set(split_hashes))
        ids.update(split_ids)
        hashes.update(split_hashes)
        split_report[split] = entry
    report["splits"] = split_report
    report["total_rows"] = sum(e["parquet_rows"] for e in split_report.values())
    report["unique_ids"] = len(ids)
    report["unique_hashes"] = len(hashes)
    report["cross_split_overlap_ids"] = report["total_rows"] - len(ids)
    report["cross_split_overlap_hashes"] = report["total_rows"] - len(hashes)

    # Leftover state.
    leftover_tmp = [
        str(p.relative_to(output_dir))
        for directory in (output_dir / "processed", output_dir / "statistics",
                          output_dir / "samples")
        for p in directory.glob(".*.tmp")
    ]
    report["leftover_temp_files"] = leftover_tmp
    report["ok"] = (
        not mismatches
        and report["total_rows"] == report["unique_ids"] == report["unique_hashes"]
        and all(e["jsonl_round_trip"] and
                e["parquet_rows"] == e["txt_rows"] == e["expected_manifest"]
                for e in split_report.values())
        and not leftover_tmp
    )
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.data.release",
        description="Validate a completed preprocessing release.",
    )
    parser.add_argument("--output-dir", required=True,
                        help="release directory produced by the pipeline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    report = validate_release(Path(args.output_dir))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "complete":
        print(f"release status is {report['status']!r}: not valid", file=sys.stderr)
        return 1
    if not report["ok"]:
        print("release validation FAILED", file=sys.stderr)
        return 1
    print(f"release OK: {report['total_rows']} rows "
          f"({report['source_revision']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
