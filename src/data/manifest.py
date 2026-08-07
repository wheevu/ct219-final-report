"""Reproducibility manifest: environment, checksums, and dataset revision.

The manifest is written beside every completed output dataset so experiments
can be reproduced exactly. It never includes secrets or machine-identifying
details beyond the Python interpreter version and package versions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .export import write_json

DEPENDENCIES = ("datasets", "pyarrow", "pandas", "tqdm")
REVISION_TIMEOUT_SECONDS = 15.0


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed (safe for large outputs)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(directory: Path) -> str | None:
    """Head commit of the repository containing `directory`, if any."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def resolve_dataset_revision(dataset: str, revision: str | None) -> str | None:
    """Resolve the dataset revision to a concrete commit sha when possible.

    Without an explicit pin, this records the current head sha of the dataset
    repository so later runs can be compared against the exact same data.
    Returns None when the dataset is local, the hub is unreachable, or the
    lookup exceeds the timeout (never blocks the pipeline).
    """
    if dataset.startswith(("/", ".", "~")) or ":" in dataset and "/" not in dataset:
        return None

    def _resolve() -> str | None:
        try:
            from huggingface_hub import HfApi

            info = HfApi().dataset_info(dataset, revision=revision)
            return str(info.sha)
        except Exception:
            return None

    result: list[str | None] = []
    worker = threading.Thread(target=lambda: result.append(_resolve()), daemon=True)
    worker.start()
    worker.join(timeout=REVISION_TIMEOUT_SECONDS)
    if worker.is_alive():
        return None
    return result[0] if result else None


def environment_info() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system() + " " + platform.release(),
        "dependency_versions": {
            name: (importlib.metadata.version(name) if _is_installed(name) else None)
            for name in DEPENDENCIES
        },
    }


def _is_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def build_manifest(
    output_dir: Path,
    settings: Settings,
    run_stats: dict,
    resolved_revision: str | None,
    output_paths: list[Path],
    row_counts: dict[str, int],
    runtime_seconds: float,
    docs_per_second: float,
    accepted_per_second: float,
    peak_memory_mb: float | None,
    hashes_retained: int,
) -> dict:
    """Assemble the full reproducibility manifest."""
    manifest = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": settings.dataset,
        "requested_dataset_revision": settings.dataset_revision,
        "resolved_dataset_revision": resolved_revision,
        "source_split": settings.dataset_split,
        "configuration": settings.effective(),
        "seed": settings.seed,
        "scan_limit": settings.scan_limit,
        "accepted_document_target": settings.max_documents,
        "git_commit": git_commit(output_dir),
        "environment": environment_info(),
        "performance": {
            "runtime_seconds": round(runtime_seconds, 1),
            "source_documents_per_second": (
                round(docs_per_second, 1) if docs_per_second is not None else None),
            "accepted_documents_per_second": (
                round(accepted_per_second, 1) if accepted_per_second is not None else None),
            "peak_memory_mb": peak_memory_mb,
            "hashes_retained_in_memory": hashes_retained,
        },
        "outputs": {
            str(path.relative_to(output_dir)): {
                "row_count": row_counts.get(path.name, row_counts.get(str(path), None)),
                "sha256": sha256_file(path),
            }
            for path in output_paths
            if path.exists()
        },
        "run_summary": {
            key: run_stats[key]
            for key in (
                "source_documents_inspected",
                "accepted_candidates",
                "accepted_documents",
                "rejected_documents",
                "duplicate_count",
                "split_counts",
            )
            if key in run_stats
        },
        "dedup": run_stats.get("dedup"),
    }
    return manifest


def write_manifest(manifest: dict, output_dir: Path) -> Path:
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    return path
