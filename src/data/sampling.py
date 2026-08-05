"""Deterministic sampling, stable hashing, and split assignment.

All randomness is derived from SHA-256, so results are identical across runs,
machines, and Python versions (never the process-randomized built-in hash()).
"""

from __future__ import annotations

import hashlib
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def stable_hash(*parts: Any) -> str:
    """SHA-256 hex digest of the parts joined with '|'.

    Accepts any hashable-safe value; everything is stringified first.
    """
    key = "|".join(str(p) for p in parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def stable_float(*parts: Any) -> float:
    """Uniform float in [0, 1) derived deterministically from the parts."""
    digest = stable_hash(*parts)
    return int(digest[:16], 16) / 2**64


def assign_split(
    doc_id: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    """Deterministically assign a document to train/validation/test.

    A document with the same id and seed always lands in the same split, so
    splits are stable across runs and across pipeline executions.
    """
    value = stable_float("split", seed, doc_id)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "validation"
    return "test"


class DeterministicReservoir(Generic[T]):
    """Streaming reservoir sampler using hash-derived randomness.

    Keeps at most `capacity` items, uniformly chosen among every candidate
    offered, without knowing the total candidate count in advance. Selecting
    only the first N documents would bias the sample; this avoids that bias
    while still streaming. When the stream is truncated (scan limit), the
    sample is uniform over the scanned prefix; see README limitations.
    """

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = capacity
        self.seed = seed
        self.items: list[T] = []
        self._offered = 0

    @property
    def offered(self) -> int:
        return self._offered

    def offer(self, item: T) -> bool:
        """Offer one candidate; returns True if it is retained."""
        self._offered += 1
        index = self._offered  # 1-based
        if len(self.items) < self.capacity:
            self.items.append(item)
            return True
        value = stable_float("reservoir", self.seed, index)
        slot = int(value * index)
        if slot < self.capacity:
            self.items[slot] = item
            return True
        return False
