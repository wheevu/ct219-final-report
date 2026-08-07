"""Shard-by-shard streaming loader for very large datasets.

The full VTSNLP/vietnamese_curated_dataset is ~35 GB across 132 parquet
shards, which does not fit on this machine alongside the release outputs.
Instead of caching everything, this loader downloads one shard at a time
(resumable `hf_hub_download` with retries), iterates its row groups, and
deletes the shard as soon as it has been consumed. Peak disk usage is one
shard (~260 MB) plus the release outputs.

Determinism: shards are processed in sorted filename order, which matches
the order `datasets` streaming would use for the same revision.

The loader only reads the pinned revision; an unpinned run fails loudly
because reproducibility requires a concrete file list.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator

import pyarrow.parquet as pq


def list_shard_files(repo_id: str, revision: str) -> list[str]:
    """Sorted list of `data/train-*.parquet` paths at the pinned revision."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo_id, repo_type="dataset", revision=revision
    )
    shards = sorted(f for f in files
                    if f.startswith("data/") and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards found in {repo_id} @ {revision}")
    return shards


def download_shard(
    repo_id: str, rel: str, revision: str, cache_dir: Path
) -> Path:
    """Download one shard (resumable); errors propagate to the caller."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id, rel, repo_type="dataset", revision=revision,
        local_dir=str(cache_dir),
    ))


def iter_shard_rows(
    shard_paths: list[Path],
    delete_after: bool = True,
) -> Iterator[dict]:
    """Yield {id, domain, text} dicts from parquet shards in order."""
    for path in shard_paths:
        try:
            table = pq.ParquetFile(path)
            for batch in table.iter_batches(batch_size=2000):
                ids = batch.column("id").to_pylist()
                domains = batch.column("domain").to_pylist()
                texts = batch.column("text").to_pylist()
                for i in range(len(ids)):
                    yield {"id": ids[i], "domain": domains[i], "text": texts[i]}
        finally:
            if delete_after:
                path.unlink(missing_ok=True)


def yield_rows(
    repo_id: str,
    revision: str,
    cache_dir: Path,
    retries: int = 5,
    _list_files: Callable[[str, str], list[str]] = list_shard_files,
    _download: Callable[[str, str, str, Path], Path] = download_shard,
    _iter_rows: Callable[[list[Path], bool], Iterator[dict]] = iter_shard_rows,
) -> Iterator[dict]:
    """Stream all rows of the dataset, one shard at a time.

    Each shard download is retried with backoff; `hf_hub_download` itself is
    resumable, so an interrupted transfer resumes rather than restarts.
    """
    if not revision:
        raise RuntimeError(
            "the shard loader requires a pinned dataset revision "
            "(--dataset-revision); reproducibility needs a concrete file list"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    shards = _list_files(repo_id, revision)
    print(f"[shard-loader] {len(shards)} shards at revision {revision[:12]}...",
          flush=True)
    for index, rel in enumerate(shards, start=1):
        last_error: Exception | None = None
        path: Path | None = None
        for attempt in range(1, retries + 1):
            try:
                path = _download(repo_id, rel, revision, cache_dir)
                break
            except Exception as exc:  # network errors, hub 5xx
                last_error = exc
                if attempt < retries:
                    delay = 10 * attempt
                    print(f"[shard-loader] attempt {attempt}/{retries} failed "
                          f"for {rel}: {type(exc).__name__}; retrying in "
                          f"{delay}s", flush=True)
                    time.sleep(delay)
        if path is None:
            raise RuntimeError(
                f"failed to download {rel} after {retries} attempts: "
                f"{last_error}") from last_error
        print(f"[shard-loader] {index}/{len(shards)} processing {rel}",
              flush=True)
        yield from _iter_rows([path], delete_after=True)
        print(f"[shard-loader] {index}/{len(shards)} done", flush=True)
