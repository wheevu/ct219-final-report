"""Tests for the shard-by-shard loader (synthetic local parquet files)."""

import pyarrow as pa
import pyarrow.parquet as pq

from data.shard_loader import iter_shard_rows, yield_rows

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("text", pa.string()),
])


def _make_shard(path, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def test_iter_shard_rows_order_and_cleanup(tmp_path):
    shard_a = tmp_path / "train-00000.parquet"
    shard_b = tmp_path / "train-00001.parquet"
    _make_shard(shard_a, [{"id": "a1", "domain": "news", "text": "Văn bản một."},
                          {"id": "a2", "domain": "wiki", "text": "Văn bản hai."}])
    _make_shard(shard_b, [{"id": "b1", "domain": "news", "text": "Văn bản ba."}])

    rows = list(iter_shard_rows([shard_a, shard_b], delete_after=True))
    assert [r["id"] for r in rows] == ["a1", "a2", "b1"]
    assert rows[0] == {"id": "a1", "domain": "news", "text": "Văn bản một."}
    assert not shard_a.exists() and not shard_b.exists()


def test_iter_shard_rows_keeps_files_when_asked(tmp_path):
    shard = tmp_path / "keep.parquet"
    _make_shard(shard, [{"id": "k1", "domain": "news", "text": "Văn bản giữ lại."}])
    list(iter_shard_rows([shard], delete_after=False))
    assert shard.exists()


def test_yield_rows_streams_in_order_and_cleans(tmp_path):
    shard_a = tmp_path / "train-00000.parquet"
    shard_b = tmp_path / "train-00001.parquet"
    _make_shard(shard_a, [{"id": "a1", "domain": "news", "text": "Văn bản một."}])
    _make_shard(shard_b, [{"id": "b1", "domain": "wiki", "text": "Văn bản hai."}])
    cache = tmp_path / "cache"

    fake_list = lambda repo, revision: ["data/train-00000.parquet",
                                        "data/train-00001.parquet"]

    def fake_download(repo, rel, revision, cache_dir):
        return shard_a if "00000" in rel else shard_b

    rows = list(yield_rows("repo", "rev123", cache,
                           _list_files=fake_list, _download=fake_download))
    assert [r["id"] for r in rows] == ["a1", "b1"]


def test_yield_rows_requires_pinned_revision(tmp_path):
    try:
        list(yield_rows("repo", None, tmp_path))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "pinned" in str(exc)


def test_yield_rows_retries_then_fails(tmp_path):
    calls = {"n": 0}

    def fake_download(repo, rel, revision, cache_dir):
        calls["n"] += 1
        raise ConnectionError("hub down")

    try:
        list(yield_rows("repo", "rev", tmp_path, retries=2,
                        _list_files=lambda a, b: ["data/train-00000.parquet"],
                        _download=fake_download))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "attempts" in str(exc)
    assert calls["n"] == 2


def test_release_mode_uses_shard_loader():
    from data.config import settings_from_args

    settings = settings_from_args(["--mode", "release-400k"])
    assert settings.loader == "shards"
    assert settings.dedup_backend == "sqlite"
    smoke = settings_from_args(["--mode", "smoke"])
    assert smoke.loader == "datasets"
