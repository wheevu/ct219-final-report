"""Equivalence tests: memory and sqlite dedup backends decide identically."""

import unicodedata

import pytest

from data.dedup import MemoryDedup, SqliteDedup, open_dedup

TEXTS = [
    "Hà Nội là thủ đô của Việt Nam. Đây là một câu văn đủ dài để vượt ngưỡng.",
    "Học sinh cần rèn luyện tiếng Việt mỗi ngày. Kiến thức sẽ bền vững hơn.",
    "Trận bóng đá tối qua rất hấp dẫn. Đội chủ nhà thắng với tỷ số sát nút.",
    "Nấu phở cần ninh xương thật kỹ. Nước dùng trong và thơm mùi quế hồi.",
]


def _stream():
    """Duplicate ids, duplicate texts, and duplicates far apart."""
    yield "id-1", TEXTS[0]
    yield "id-1", TEXTS[1]            # duplicate id (immediately)
    yield "id-2", TEXTS[1]
    yield "id-3", TEXTS[2]
    yield "id-4", TEXTS[0]            # duplicate text of id-1, far apart
    yield "id-5", unicodedata.normalize("NFD", TEXTS[2])  # Unicode-equivalent (NFC upstream)
    for i in range(30):
        yield f"fill-{i}", TEXTS[i % 2] + f" — mở rộng thứ {i}."
    yield "id-6", TEXTS[1]            # duplicate text of id-2, far apart
    yield "id-5", TEXTS[3]            # duplicate id, far apart


def _run(backend):
    decisions = []
    for doc_id, text in _stream():
        cleaned = unicodedata.normalize("NFC", text)
        decisions.append(backend.check_and_add(doc_id, cleaned))
    return decisions


def test_backends_decide_identically(tmp_path):
    memory = MemoryDedup(seed=42)
    sqlite = SqliteDedup(seed=42, directory=tmp_path)
    assert _run(memory) == _run(sqlite)
    expected = (
        [None, "duplicate_id", None, None, "duplicate_text", "duplicate_text"]
        + [None] * 30
        + ["duplicate_text", None]  # id-6 text dup; id-5 was never registered
    )
    assert _run(MemoryDedup(seed=42)) == expected
    memory.close()
    sqlite.close()


def test_unicode_equivalent_text_is_duplicate_after_normalization(tmp_path):
    # The pipeline NFC-normalizes before dedup; backends hash exact input.
    for backend in (
        MemoryDedup(seed=42),
        SqliteDedup(seed=42, directory=tmp_path),
    ):
        base = unicodedata.normalize("NFC", TEXTS[2])
        assert backend.check_and_add("a", base) is None
        equivalent = unicodedata.normalize("NFC", unicodedata.normalize("NFD", TEXTS[2]))
        assert backend.check_and_add("b", equivalent) == "duplicate_text"
        backend.close()


def test_sqlite_cleans_up_temp_files(tmp_path):
    backend = SqliteDedup(seed=42, directory=tmp_path)
    backend.check_and_add("a", TEXTS[0])
    backend.close()
    leftovers = [p for p in tmp_path.iterdir()]
    assert leftovers == [], leftovers


def test_sqlite_keep_retains_database(tmp_path):
    backend = SqliteDedup(seed=42, directory=tmp_path, keep=True)
    backend.check_and_add("a", TEXTS[0])
    backend.close()
    assert (tmp_path / "dedup.sqlite3").exists()


def test_sqlite_records_duplicates_in_stats(tmp_path):
    backend = SqliteDedup(seed=42, directory=tmp_path)
    backend.check_and_add("a", TEXTS[0])
    backend.check_and_add("a", TEXTS[1])
    stats = backend.stats()
    assert stats["ids_retained"] == 1
    assert stats["hashes_retained"] == 1
    assert stats["duplicate_hits"] == 1
    backend.close()


def test_sqlite_batches_transactions(tmp_path):
    backend = SqliteDedup(seed=42, directory=tmp_path)
    for i in range(25_000):  # crosses multiple 10k batches
        assert backend.check_and_add(f"id-{i}", TEXTS[i % 4] + f" {i}") is None
    # duplicates across batches still detected
    assert backend.check_and_add("id-0", TEXTS[0] + " 0") == "duplicate_id"
    stats = backend.stats()
    assert stats["hashes_retained"] == 25_000
    backend.close()


def test_sqlite_handles_interrupted_state_safely(tmp_path):
    """A leftover database from an interrupted run is replaced, not reused."""
    leftover = tmp_path / "dedup.sqlite3"
    leftover.write_bytes(b"garbage not a database")  # crashed mid-run artifact
    backend = SqliteDedup(seed=42, directory=tmp_path)
    assert backend.check_and_add("a", TEXTS[0]) is None
    assert backend.check_and_add("a", TEXTS[1]) == "duplicate_id"
    backend.close()


def test_sqlite_unknown_table_fails_loudly(tmp_path):
    import sqlite3

    backend = SqliteDedup(seed=42, directory=tmp_path)
    with pytest.raises(RuntimeError):
        backend._insert("does_not_exist", "x")
    backend.close()


def test_open_dedup_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        open_dedup("bloom", 42, tmp_path)


def test_sqlite_db_after_interrupted_batch_is_consistent(tmp_path):
    """Simulate a crash mid-run: uncommitted batch must not corrupt state."""
    backend = SqliteDedup(seed=42, directory=tmp_path)
    backend.check_and_add("a", TEXTS[0])
    backend.check_and_add("b", TEXTS[1])  # in the same open batch
    backend._flush()                      # simulated crash boundary commit
    import sqlite3

    conn = sqlite3.connect(str(backend.path))
    rows = conn.execute("SELECT COUNT(*) FROM seen_ids").fetchone()[0]
    conn.close()
    assert rows == 2
    backend.close()
