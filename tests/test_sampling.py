"""Tests for stable hashing, reservoir sampling, and split assignment."""

from data.sampling import DeterministicReservoir, assign_split, stable_float, stable_hash


def test_stable_hash_deterministic_across_calls():
    assert stable_hash("a", 1) == stable_hash("a", 1)
    assert stable_hash("a", 1) != stable_hash("a", 2)
    assert stable_hash("a", 1) != stable_hash("b", 1)


def test_stable_float_in_unit_interval_and_deterministic():
    value = stable_float("split", 42, "doc-1")
    assert 0.0 <= value < 1.0
    assert value == stable_float("split", 42, "doc-1")


def test_split_assignment_is_deterministic():
    first = assign_split("doc-123", 42, 0.9, 0.05)
    second = assign_split("doc-123", 42, 0.9, 0.05)
    assert first == second


def test_split_assignment_covered_all_splits_across_many_ids():
    splits = {assign_split(f"doc-{i}", 42, 0.9, 0.05) for i in range(5000)}
    assert splits == {"train", "validation", "test"}


def test_split_ratios_follow_configured_boundaries():
    # Ratios 1.0/0.0 would be invalid; use 0.99/0.005 to keep sanity checks.
    counts = {"train": 0, "validation": 0, "test": 0}
    for i in range(20000):
        counts[assign_split(f"id-{i}", 7, 0.9, 0.05)] += 1
    total = sum(counts.values())
    assert 0.85 < counts["train"] / total < 0.95
    assert 0.02 < counts["validation"] / total < 0.08
    assert 0.02 < counts["test"] / total < 0.08


def test_split_boundaries_never_outside_splits():
    for i in range(5000):
        split = assign_split(f"x-{i}", 1, 0.5, 0.25)
        assert split in ("train", "validation", "test")


def test_reservoir_keeps_first_items_up_to_capacity():
    reservoir = DeterministicReservoir(3, seed=1)
    accepted = [reservoir.offer(i) for i in range(3)]
    assert accepted == [True, True, True]
    assert sorted(reservoir.items) == [0, 1, 2]
    assert reservoir.offered == 3


def test_reservoir_capacity_bounded_after_overflow():
    reservoir = DeterministicReservoir(10, seed=99)
    for i in range(1000):
        reservoir.offer(i)
    assert len(reservoir.items) == 10
    assert reservoir.offered == 1000


def test_reservoir_is_deterministic():
    def run():
        r = DeterministicReservoir(50, seed=123)
        for i in range(10_000):
            r.offer(i)
        return sorted(r.items)

    assert run() == run()


def test_reservoir_does_not_keep_naive_prefix():
    reservoir = DeterministicReservoir(100, seed=42)
    for i in range(10_000):
        reservoir.offer(i)
    items = set(reservoir.items)
    assert len(items) == 100
    assert items != set(range(100))  # naive first-100 prefix was replaced
    assert max(items) > 100  # later stream items made it in
