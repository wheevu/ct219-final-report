"""End-to-end pipeline test: manifest, checksums, split disjointness.

Runs the real pipeline on synthetic streamed documents (no network) and
verifies the manifest, output files, and cross-split guarantees.
"""

import json

import pyarrow.parquet as pq

from data.config import Settings
from data.preprocess import run_pipeline

VI_DOCS = [
    "Hà Nội là thủ đô của Việt Nam. Đây là một câu văn khá dài để vượt ngưỡng tối thiểu.",
    "Học sinh cần chăm chỉ rèn luyện tiếng Việt mỗi ngày. Kiến thức sẽ bền vững hơn.",
    "Sáng sớm, chợ quê đã đông vui. Người bán rau quả cười nói rôm rả.",
    "Trận bóng đá tối qua rất hấp dẫn. Đội chủ nhà thắng với tỷ số sát nút.",
    "Nấu phở cần ninh xương thật kỹ. Nước dùng trong và thơm mùi quế hồi.",
    "Mùa xuân về, hoa đào nở rộ khắp phố phường. Ai cũng rộn ràng chào đón năm mới.",
    "Bảo vệ môi trường là trách nhiệm của tất cả mọi người. Rác thải cần phân loại.",
    "Trí tuệ nhân tạo đang thay đổi thế giới. Các nhà nghiên cứu hợp tác chặt chẽ.",
    "Đêm trăng sáng vằng vặc, lũ trẻ chơi trò trốn tìm quanh sân đình làng.",
    "Sách là người bạn tốt của con người. Đọc sách mở mang trí tuệ và tâm hồn.",
]


def test_pipeline_end_to_end_with_manifest(tmp_path, monkeypatch):
    def fake_docs(settings):
        for i in range(120):
            text = VI_DOCS[i % len(VI_DOCS)] + f" — bản sao thứ {i}."
            yield {"id": f"id-{i}", "domain": "news", "text": text}

    monkeypatch.setattr("data.preprocess.load_documents", fake_docs)

    settings = Settings(
        output_dir=str(tmp_path / "out"),
        max_documents=40,
        scan_limit=120,
        seed=42,
        overwrite=False,
    )
    stats = run_pipeline(settings)

    assert stats["accepted_documents"] == 40
    assert sum(stats["split_counts"].values()) == 40

    # Manifest exists and is complete.
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text("utf-8"))
    assert manifest["dataset"] == "VTSNLP/vietnamese_curated_dataset"
    assert manifest["seed"] == 42
    assert manifest["run_summary"]["accepted_documents"] == 40
    assert "outputs" in manifest
    assert "environment" in manifest
    assert "python_version" in manifest["environment"]

    # Checksums verify.
    for rel, info in manifest["outputs"].items():
        from data.manifest import sha256_file

        path = tmp_path / "out" / rel
        assert path.exists()
        assert sha256_file(path) == info["sha256"]

    # No id or text_hash overlap across splits.
    ids, hashes = set(), set()
    for split in ("train", "validation", "test"):
        table = pq.read_table(tmp_path / "out" / "processed" / f"{split}.parquet")
        for i in range(table.num_rows):
            did = table.column("id").to_pylist()[i]
            text_hash = table.column("text_hash").to_pylist()[i]
            assert did not in ids
            assert text_hash not in hashes
            ids.add(did)
            hashes.add(text_hash)

    # JSONL boundaries round-trip and match parquet.
    for split in ("train", "validation", "test"):
        table = pq.read_table(tmp_path / "out" / "processed" / f"{split}.parquet")
        lines = (tmp_path / "out" / "processed" / f"{split}.txt").read_text(
            "utf-8").splitlines()
        assert len(lines) == table.num_rows
        restored = [json.loads(line) for line in lines]
        assert restored == table.column("text").to_pylist()


def test_pipeline_refuses_overwrite_without_flag(tmp_path, monkeypatch):
    def fake_docs(settings):
        yield {"id": "id-1", "domain": "news", "text": VI_DOCS[0]}

    monkeypatch.setattr("data.preprocess.load_documents", fake_docs)
    settings = Settings(output_dir=str(tmp_path / "out"), max_documents=5,
                        scan_limit=10, seed=1)
    run_pipeline(settings)
    try:
        run_pipeline(settings)
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
