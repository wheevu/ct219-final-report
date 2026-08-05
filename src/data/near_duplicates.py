"""Optional near-duplicate audit using deterministic SimHash fingerprints.

Disabled by default (see --near-duplicates). Finds documents whose character
shingle sets are near-identical under Hamming distance on 64-bit SimHash
fingerprints. Purely advisory: near-duplicates are reported, never removed.

Limitations (documented in README and docs/data_contract.md):

- Character-shingle SimHash measures surface similarity, not semantics.
  Documents sharing heavy boilerplate (menus, headers, footers) can collide
  even when their content differs (false positives).
- Documents that are near-duplicates after light paraphrasing or with little
  shared surface text can be missed (false negatives).
- The threshold is Hamming distance on 64 bits: threshold 3 means fingerprints
  agreeing on at least 61 of 64 bits.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .export import write_csv, write_json
from .sampling import stable_hash


def shingles(text: str, n: int) -> set[str]:
    """Character n-gram shingles of the text (unique)."""
    if n <= 0 or len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def simhash_fingerprint(text: str, seed: int, shingle_size: int = 4) -> int:
    """64-bit SimHash of the document's unique shingles, deterministic.

    The per-bit accumulation is vectorized over shingles with numpy; the
    result is identical to a scalar implementation.
    """
    grams = shingles(text, shingle_size)
    if not grams:
        return 0
    values = np.array(
        [int(stable_hash("simhash", seed, gram)[:16], 16) for gram in grams],
        dtype=np.uint64,
    )
    bits = (values[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)
    weights = (bits.astype(np.int8) * 2 - 1).sum(axis=0)
    fingerprint = 0
    for bit in range(64):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class SimHashIndex:
    """Banded 64-bit SimHash index: 4 buckets of 16 bits each.

    Candidate lookup keeps memory at O(n) fingerprints (8 bytes each) plus
    bucket lists, with no quadratic scans.
    """

    def __init__(self) -> None:
        self._buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        self._items: list[tuple[str, str, int]] = []
        self._fingerprints: dict[str, int] = {}

    def add(self, doc_id: str, split: str, fingerprint: int) -> None:
        self._items.append((doc_id, split, fingerprint))
        self._fingerprints[doc_id] = fingerprint
        for band in range(4):
            key = (band, (fingerprint >> (16 * band)) & 0xFFFF)
            self._buckets[key].append(doc_id)

    def candidates(self, fingerprint: int) -> set[str]:
        """All ids sharing at least one 16-bit band with the fingerprint."""
        found: set[str] = set()
        for band in range(4):
            key = (band, (fingerprint >> (16 * band)) & 0xFFFF)
            found.update(self._buckets.get(key, ()))
        return found


def find_near_duplicates(
    documents: Iterable[tuple[str, str, str]],
    seed: int,
    threshold: int = 3,
    shingle_size: int = 4,
    out_dir: Path | None = None,
) -> dict:
    """Report near-duplicate pairs and clusters above `threshold`.

    `documents` yields (id, split, text) tuples. When `out_dir` is given,
    writes near_duplicate_pairs.csv, near_duplicate_clusters.csv, and
    near_duplicate_summary.json. Returns a summary dict.
    """
    index = SimHashIndex()
    shingle_sets: dict[str, set[str]] = {}
    by_id: dict[str, str] = {}
    for doc_id, split, text in documents:
        index.add(doc_id, split, simhash_fingerprint(text, seed, shingle_size))
        shingle_sets[doc_id] = shingles(text, shingle_size)
        by_id[doc_id] = split

    pairs: list[tuple[str, str, int, float]] = []
    for doc_id, split, fingerprint in index._items:
        for other_id in index.candidates(fingerprint):
            if other_id <= doc_id:
                continue
            distance = hamming_distance(fingerprint, index._fingerprints[other_id])
            if distance > threshold:
                continue
            overlap = _jaccard(shingle_sets[doc_id], shingle_sets[other_id])
            pairs.append((doc_id, other_id, distance, overlap))

    pairs.sort(key=lambda p: (p[2], -p[3], p[0], p[1]))
    cross_split = [p for p in pairs if by_id[p[0]] != by_id[p[1]]]

    clusters = _clusters([(p[0], p[1]) for p in pairs])

    summary = {
        "enabled": True,
        "documents_indexed": len(index._items),
        "pairs": len(pairs),
        "cross_split_pairs": len(cross_split),
        "clusters": len(clusters),
        "largest_cluster_size": max((len(c) for c in clusters), default=0),
        "threshold_hamming_bits": threshold,
        "shingle_size": shingle_size,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            out_dir / "near_duplicate_pairs.csv",
            ["id_a", "id_b", "hamming_distance", "shingle_jaccard", "cross_split"],
            [[a, b, d, round(o, 4), by_id[a] != by_id[b]] for a, b, d, o in pairs],
        )
        write_csv(
            out_dir / "near_duplicate_clusters.csv",
            ["cluster_id", "size", "ids"],
            [[i + 1, len(c), "|".join(sorted(c))] for i, c in enumerate(clusters)],
        )
        write_json(out_dir / "near_duplicate_summary.json", summary)
    return summary


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _clusters(pairs: list[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, set[str]] = defaultdict(set)
    for doc_id in list(parent):
        groups[find(doc_id)].add(doc_id)
    return [g for g in groups.values() if len(g) > 1]
