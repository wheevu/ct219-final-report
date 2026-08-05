"""Tests for safe exports: JSONL document boundaries and parquet output."""

import json

import pyarrow.parquet as pq

from data.export import AtomicTextWriter, ParquetSplitWriter, truncate_for_display
from data.sampling import stable_hash

SPLITS = ("train", "validation", "test")


def test_txt_export_preserves_document_boundaries(tmp_path):
    docs = [
        "Dòng một.\nDòng hai.\n\nDòng ba.",  # multiline document
        'Có dấu "ngoặc kép" và \\ gạch chéo.',
        "Xin chào, đây là văn bản có dấu phẩy, chấm; và ký tự đặc biệt.",
    ]
    writer = AtomicTextWriter(tmp_path / "docs.txt")
    for doc in docs:
        writer.write_line(json.dumps(doc, ensure_ascii=False))
    writer.commit()

    lines = (tmp_path / "docs.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(docs)
    decoded = [json.loads(line) for line in lines]
    assert decoded == docs


def test_round_trip_with_unicode_diacritics(tmp_path):
    doc = "Đây là văn bản tiếng Việt: ế ệ ỹ ả à ạ\nKết thúc."
    writer = AtomicTextWriter(tmp_path / "u.txt")
    writer.write_line(json.dumps(doc, ensure_ascii=False))
    writer.commit()
    restored = json.loads((tmp_path / "u.txt").read_text(encoding="utf-8"))
    assert restored == doc


def test_parquet_writer_writes_all_splits(tmp_path):
    writer = ParquetSplitWriter(tmp_path)
    for i in range(1500):  # forces multiple flush batches
        writer.add("train", {
            "id": f"id-{i}",
            "domain": "news",
            "text": f"Văn bản mẫu số {i}.",
            "text_hash": stable_hash(f"id-{i}"),
            "character_count": len(f"Văn bản mẫu số {i}."),
        })
    writer.add("validation", {
        "id": "val-1",
        "domain": "wiki",
        "text": "Văn bản kiểm tra.",
        "text_hash": stable_hash("val-1"),
        "character_count": 17,
    })
    writer.commit()

    for split in SPLITS:
        path = tmp_path / f"{split}.parquet"
        assert path.exists()
        table = pq.read_table(path)
        assert set(table.column_names) == {
            "id", "domain", "text", "text_hash", "character_count"
        }

    train_table = pq.read_table(tmp_path / "train.parquet")
    assert train_table.num_rows == 1500
    assert train_table.column("id").to_pylist()[1499] == "id-1499"


def test_truncate_for_display_only_affects_display():
    long_text = "x" * 500
    shown = truncate_for_display(long_text, limit=200)
    assert len(shown) == 200 + len("…[truncated]")
    assert long_text not in shown
    short = "ngắn"
    assert truncate_for_display(short) == short
