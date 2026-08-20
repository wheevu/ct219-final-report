"""Tests for the 400k release configuration and long-run reliability."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from data.config import Settings, settings_from_args
from data.export import RELEASE_COMPLETE, release_status
from data.preprocess import run_pipeline
from data.release import validate_release

VI_DOCS = [
    "Hà Nội là thủ đô của Việt Nam. Đây là một câu văn khá dài để vượt ngưỡng tối thiểu.",
    "Học sinh cần chăm chỉ rèn luyện tiếng Việt mỗi ngày. Kiến thức sẽ bền vững hơn.",
    "Sáng sớm, chợ quê đã đông vui. Người bán rau quả cười nói rôm rả.",
    "Trận bóng đá tối qua rất hấp dẫn. Đội chủ nhà thắng với tỷ số sát nút.",
    "Nấu phở cần ninh xương thật kỹ. Nước dùng trong và thơm mùi quế hồi.",
]


def _fake_docs(settings):
    for i in range(300):
        yield {"id": f"id-{i}", "domain": "news",
               "text": VI_DOCS[i % len(VI_DOCS)] + f" — bản mở rộng {i}."}


def _run(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr("data.preprocess.load_documents", _fake_docs)
    settings = Settings(output_dir=str(tmp_path), max_documents=40,
                        scan_limit=200, seed=42, **overrides)
    return run_pipeline(settings)


# ---------------------------------------------------------------- release config

def test_release_400k_mode_defaults():
    settings = settings_from_args(["--mode", "release-400k"])
    assert settings.max_documents == 400_000
    assert settings.scan_limit is None          # full stream, no accidental limit
    assert settings.dedup_backend == "sqlite"   # on-disk for ~12M candidates
    assert settings.seed == 42
    assert settings.train_ratio == 0.90 and settings.val_ratio == 0.05


def test_release_mode_does_not_change_smoke_and_development():
    smoke = settings_from_args(["--mode", "smoke"])
    dev = settings_from_args(["--mode", "development"])
    assert smoke.dedup_backend == "memory"
    assert smoke.max_documents == 2000
    assert dev.max_documents == 20_000
    assert dev.scan_limit == 500_000


def test_explicit_flags_override_release_mode():
    settings = settings_from_args(
        ["--mode", "release-400k", "--max-documents", "500000",
         "--scan-limit", "0", "--dedup-backend", "memory"])
    assert settings.max_documents == 500_000
    assert settings.scan_limit is None  # 0 = unlimited
    assert settings.dedup_backend == "memory"


def test_config_serialization_is_deterministic():
    settings = settings_from_args(["--mode", "release-400k"])
    first = json.dumps(settings.effective(), sort_keys=True)
    second = json.dumps(Settings(**json.loads(first)).effective(), sort_keys=True)
    assert first == second
    assert "scan_limit" in first and "dedup_backend" in first


# ---------------------------------------------------------------- completion marker

def test_run_writes_completion_marker(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    assert (tmp_path / RELEASE_COMPLETE).exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "complete"
    assert release_status(tmp_path) == "complete"


def test_missing_marker_means_incomplete(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    (tmp_path / RELEASE_COMPLETE).unlink()
    assert release_status(tmp_path) == "incomplete"


def test_partial_outputs_detected_as_incomplete(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    # Simulate a crashed rerun: marker gone, one output replaced.
    (tmp_path / RELEASE_COMPLETE).unlink()
    (tmp_path / "processed" / "train.parquet").unlink()
    pq.write_table(
        pq.read_table(tmp_path / "processed" / "validation.parquet"),
        tmp_path / "processed" / "train.parquet")
    assert release_status(tmp_path) == "incomplete"


def test_rerun_without_overwrite_fails_safely(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    try:
        _run(tmp_path, monkeypatch)
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    assert release_status(tmp_path) == "complete"  # untouched by failed rerun


def test_rerun_with_overwrite_restarts_clean(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    _run(tmp_path, monkeypatch, overwrite=True)
    assert release_status(tmp_path) == "complete"
    # No stale temp files.
    leftovers = [p for d in ("processed", "statistics", "samples")
                 for p in (tmp_path / d).glob(".*.tmp")]
    assert leftovers == []


def test_stale_temp_files_swept_on_run(tmp_path, monkeypatch):
    (tmp_path / "processed").mkdir(parents=True)
    stale = tmp_path / "processed" / ".train.txt.abc123.tmp"
    stale.write_text("partial")
    _run(tmp_path, monkeypatch)
    assert not stale.exists()


def test_revision_recorded_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "data.preprocess.resolve_dataset_revision",
        lambda dataset, revision: "b81fcce58945970117a1b56d50ec81be2628a5c3")
    _run(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert manifest["resolved_dataset_revision"] == "b81fcce58945970117a1b56d50ec81be2628a5c3"
    stats = json.loads(
        (tmp_path / "statistics" / "preprocessing_stats.json").read_text("utf-8"))
    assert stats["resolved_dataset_revision"] == "b81fcce58945970117a1b56d50ec81be2628a5c3"


# ---------------------------------------------------------------- validation

def test_validate_release_full_check(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    report = validate_release(tmp_path)
    assert report["status"] == "complete"
    assert report["ok"] is True
    assert report["total_rows"] == 40
    assert report["cross_split_overlap_ids"] == 0
    assert report["cross_split_overlap_hashes"] == 0
    assert report["checksum_mismatches"] == []


def test_validate_release_detects_tampering(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    with open(tmp_path / "processed" / "train.txt", "a", encoding="utf-8") as fh:
        fh.write("\"tampered\"\n")
    report = validate_release(tmp_path)
    assert report["ok"] is False
    assert any("checksum mismatch" in m for m in report["checksum_mismatches"])


def test_validate_release_detects_cross_split_overlap(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    # Copy train row 0 into test (same id/hash) to simulate leakage.
    import pyarrow as pa

    train = pq.read_table(tmp_path / "processed" / "train.parquet")
    row0 = train.slice(0, 1)
    test_path = tmp_path / "processed" / "test.parquet"
    test = pq.read_table(test_path)
    pq.write_table(pa.concat_tables([test, row0]), test_path)
    report = validate_release(tmp_path)
    assert report["ok"] is False
    assert report["cross_split_overlap_ids"] == 1
