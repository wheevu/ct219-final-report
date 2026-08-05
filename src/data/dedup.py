"""Exact deduplication backends.

Two backends with identical decisions:

- `memory`: two Python sets. Fast, but retains every accepted candidate's id
  and text hash in RAM, growing linearly with the scanned population
  (measured ~126 MiB per 500k candidates). Fine for bounded scans; risky for
  a full-dataset scan.
- `sqlite`: an on-disk SQLite database with PRIMARY KEY columns, so
  uniqueness is enforced by the engine and the hash population never lives
  in Python memory. Batched transactions keep it fast. The temporary
  database lives in a configurable working directory and is removed after a
  successful run unless retention is requested.

Both backends decide identically: a document is a duplicate when its id or
its normalized-text SHA-256 was already seen in the scanned window.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from .sampling import stable_hash


class MemoryDedup:
    """In-memory exact deduplication (default; unchanged behavior)."""

    name = "memory"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._ids: set[str] = set()
        self._hashes: set[str] = set()

    def check_and_add(self, doc_id: str, text: str) -> str | None:
        """Register a document; returns the duplicate reason, or None when new."""
        if doc_id in self._ids:
            return "duplicate_id"
        text_hash = stable_hash(text)
        if text_hash in self._hashes:
            return "duplicate_text"
        self._ids.add(doc_id)
        self._hashes.add(text_hash)
        return None

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "ids_retained": len(self._ids),
            "hashes_retained": len(self._hashes),
        }

    def close(self) -> None:
        pass


class SqliteDedup:
    """On-disk exact deduplication with engine-enforced uniqueness.

    The database is always created fresh: any leftover file from an
    interrupted run is replaced, so stale dedup state can never leak into a
    new run. Each batch of inserts commits atomically; a crash loses at most
    the open batch and leaves a consistent database.
    """

    name = "sqlite"
    _batch_size = 10_000
    _schema = """
        CREATE TABLE seen_ids (id TEXT PRIMARY KEY);
        CREATE TABLE seen_hashes (hash TEXT PRIMARY KEY);
    """

    def __init__(self, seed: int, directory: Path, keep: bool = False) -> None:
        self.seed = seed
        self.keep = keep
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "dedup.sqlite3"
        if self.path.exists():
            self.path.unlink()  # fresh state for every run
        try:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(self._schema)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"failed to initialize dedup database at {self.path}: {exc}"
            ) from exc
        self._ops = 0
        self._duplicates = 0
        self._rows_ids = 0
        self._rows_hashes = 0

    def check_and_add(self, doc_id: str, text: str) -> str | None:
        """Register a document; returns the duplicate reason, or None when new.

        Mirrors the memory backend exactly: the id is checked first (so a
        document with a duplicate id and a duplicate hash reports
        `duplicate_id`), and a text-duplicate never registers its id.
        """
        if self._ops == 0:
            self._conn.execute("BEGIN")
        self._ops += 1
        if self._exists("seen_ids", doc_id):
            self._duplicates += 1
            self._flush_if_needed()
            return "duplicate_id"
        text_hash = stable_hash(text)
        if self._exists("seen_hashes", text_hash):
            self._duplicates += 1
            self._flush_if_needed()
            return "duplicate_text"
        # Fresh values: the inserts cannot collide with existing rows.
        self._insert("seen_ids", doc_id)
        self._insert("seen_hashes", text_hash)
        self._flush_if_needed()
        return None

    def _exists(self, table: str, value: str) -> bool:
        column = "id" if table == "seen_ids" else "hash"
        try:
            cursor = self._conn.execute(
                f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)
            )
        except sqlite3.Error as exc:
            raise RuntimeError(f"dedup database read failed: {exc}") from exc
        return cursor.fetchone() is not None

    def _insert(self, table: str, value: str) -> bool:
        """INSERT OR IGNORE; True when the row was actually inserted."""
        try:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO {table} VALUES (?)", (value,)
            )
        except sqlite3.Error as exc:
            raise RuntimeError(f"dedup database write failed: {exc}") from exc
        inserted = cursor.rowcount == 1
        if inserted:
            if table == "seen_ids":
                self._rows_ids += 1
            else:
                self._rows_hashes += 1
        return inserted

    def _flush_if_needed(self) -> None:
        if self._ops >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if self._ops:
            self._conn.execute("COMMIT")
            self._ops = 0

    def stats(self) -> dict:
        self._flush()
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "backend": self.name,
            "database_bytes": size,
            "ids_retained": self._rows_ids,
            "hashes_retained": self._rows_hashes,
            "duplicate_hits": self._duplicates,
            "journal_mode": "wal",
        }

    def close(self) -> None:
        try:
            self._flush()
            self._conn.close()
        except sqlite3.Error:
            pass
        if not self.keep:
            for suffix in ("", "-wal", "-shm"):
                path = Path(str(self.path) + suffix)
                if path.exists():
                    path.unlink()


def open_dedup(backend: str, seed: int, directory: Path, keep: bool = False):
    """Factory; unknown backend names fail loudly."""
    if backend == "memory":
        return MemoryDedup(seed)
    if backend == "sqlite":
        return SqliteDedup(seed, directory, keep=keep)
    raise ValueError(
        f"unknown --dedup-backend {backend!r}; choose from 'memory', 'sqlite'"
    )
