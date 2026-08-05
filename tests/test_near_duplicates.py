"""Tests for the SimHash near-duplicate audit."""

from pathlib import Path

from data.near_duplicates import (
    SimHashIndex,
    find_near_duplicates,
    hamming_distance,
    shingles,
    simhash_fingerprint,
)

VI_BASE = (
    "Ngày nay, việc học tiếng Việt ngày càng trở nên quan trọng. "
    "Mỗi người cần rèn luyện kỹ năng đọc viết mỗi ngày. "
    "Chúng ta hãy cùng nhau giữ gìn sự trong sáng của tiếng Việt."
)


def _near_copy(text: str, tweak: str) -> str:
    return text.replace("Ngày nay,", tweak)


def test_fingerprint_deterministic():
    assert simhash_fingerprint(VI_BASE, 42) == simhash_fingerprint(VI_BASE, 42)
    assert simhash_fingerprint(VI_BASE, 42) != simhash_fingerprint(VI_BASE, 43)


def test_identical_texts_have_zero_distance():
    a = simhash_fingerprint(VI_BASE, 42)
    b = simhash_fingerprint(VI_BASE, 42)
    assert hamming_distance(a, b) == 0


def test_near_copy_within_threshold():
    near = _near_copy(VI_BASE, "Ngày nay,")
    tweaked = _near_copy(VI_BASE, "Hôm nay,")
    distance = hamming_distance(
        simhash_fingerprint(near, 42), simhash_fingerprint(tweaked, 42)
    )
    assert distance <= 3


def test_unrelated_texts_far_apart():
    other = (
        "Công thức nấu phở gồm nước dùng xương bò, hành, gừng, quế, hồi. "
        "Thời gian ninh khoảng bốn tiếng. Bánh phở tươi mua ở chợ."
    )
    distance = hamming_distance(
        simhash_fingerprint(VI_BASE, 42), simhash_fingerprint(other, 42)
    )
    assert distance > 3


def test_shingles_unique():
    grams = shingles("abab", 2)
    assert grams == {"ab", "ba"}


def test_index_finds_candidates_and_cross_split_flag(tmp_path):
    docs = [
        ("d1", "train", VI_BASE),
        ("d2", "validation", _near_copy(VI_BASE, "Hôm nay,")),
        ("d3", "test", "văn bản hoàn toàn khác nhau hoàn toàn khác nhau hoàn toàn khác"),
    ]
    summary = find_near_duplicates(
        docs, seed=42, threshold=3, shingle_size=4, out_dir=tmp_path
    )
    assert summary["documents_indexed"] == 3
    assert summary["pairs"] == 1
    assert summary["cross_split_pairs"] == 1

    pairs = (tmp_path / "near_duplicate_pairs.csv").read_text("utf-8").splitlines()
    assert len(pairs) == 2  # header + one pair
    assert "True" in pairs[1]  # cross-split

    clusters = (tmp_path / "near_duplicate_clusters.csv").read_text("utf-8").splitlines()
    assert "d1|d2" in clusters[1]


def test_clusters_merge_chains(tmp_path):
    docs = [
        ("a", "train", VI_BASE),
        ("b", "train", _near_copy(VI_BASE, "Hôm nay,")),
        ("c", "train", _near_copy(_near_copy(VI_BASE, "Hôm nay,"), "ngày càng")),
    ]
    summary = find_near_duplicates(docs, seed=42, threshold=5, out_dir=tmp_path)
    assert summary["clusters"] == 1
    assert summary["largest_cluster_size"] == 3


def test_index_bucket_membership():
    index = SimHashIndex()
    fp = simhash_fingerprint(VI_BASE, 42)
    index.add("d1", "train", fp)
    assert "d1" in index.candidates(fp)
    other = simhash_fingerprint("một văn bản khác hoàn toàn, không liên quan gì cả", 42)
    assert index.candidates(other) == set() or "d1" not in index.candidates(other)
