"""In-memory accumulation of statistics over a streaming run."""

from __future__ import annotations

from collections import Counter
from statistics import median, quantiles

from .cleaning import ALL_REASONS


class RemovalCounters(Counter):
    """Counts every rejection reason; always includes all known reasons."""

    def snapshot(self) -> dict[str, int]:
        out = {reason: int(self.get(reason, 0)) for reason in ALL_REASONS}
        out["total"] = int(sum(self.values()))
        return out


class LengthStats:
    """Collects accepted document lengths; percentiles computed at the end."""

    def __init__(self) -> None:
        self._lengths: list[int] = []

    def add(self, length: int) -> None:
        self._lengths.append(length)

    def __len__(self) -> int:
        return len(self._lengths)

    def summary(self) -> dict[str, float | int | None]:
        if not self._lengths:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "p99": None,
            }
        lengths = sorted(self._lengths)
        if len(lengths) == 1:
            value = lengths[0]
            return {
                "count": 1,
                "min": value,
                "max": value,
                "mean": float(value),
                "median": float(value),
                "p90": value,
                "p95": value,
                "p99": value,
            }
        p90, p95, p99 = quantiles(lengths, n=100, method="inclusive")[89:92]
        return {
            "count": len(lengths),
            "min": lengths[0],
            "max": lengths[-1],
            "mean": round(sum(lengths) / len(lengths), 2),
            "median": median(lengths),
            "p90": p90,
            "p95": p95,
            "p99": p99,
        }


class DomainTracker:
    """Counts documents per domain for inspected and accepted sets."""

    def __init__(self) -> None:
        self.inspected: Counter[str] = Counter()
        self.accepted: Counter[str] = Counter()

    def record_inspected(self, domain: str) -> None:
        self.inspected[domain] += 1

    def record_accepted(self, domain: str) -> None:
        self.accepted[domain] += 1

    def rows(self) -> list[dict[str, str | int]]:
        domains = sorted(set(self.inspected) | set(self.accepted))
        return [
            {
                "domain": domain,
                "inspected": int(self.inspected.get(domain, 0)),
                "accepted": int(self.accepted.get(domain, 0)),
            }
            for domain in domains
        ]
