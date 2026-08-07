"""Tests for Hugging Face publication helpers (no network required).

Covers: repo-type distinction, private-by-default, allow/ignore patterns,
secret exclusion, dataset card rendering, model directory validation, and
dry-run upload argument wiring.
"""

import json

from data.hf_publish import (
    DATASET_IGNORE_PATTERNS,
    dataset_allow_patterns,
    propose_repo_id,
    render_dataset_card,
    scan_for_secrets,
    upload_dataset,
    upload_model,
    validate_model_dir,
)


def test_propose_repo_ids_distinguish_dataset_and_model():
    assert propose_repo_id("wheevu", "dataset") == "wheevu/ct219-vietnamese-raw-400k"
    assert propose_repo_id("wheevu", "model") == "wheevu/ct219-vietnamese-next-token-model"


def test_dataset_allow_patterns_curated():
    allow = dataset_allow_patterns()
    for expected in ("processed/train.parquet", "processed/test.parquet",
                     "manifest.json", "statistics/removal_counts.csv",
                     "checksums.txt", "README.md"):
        assert expected in allow
    # No duplicate representation, no rejected previews, no db.
    assert "processed/train.txt" not in allow
    assert "rejected_examples.csv" not in " ".join(allow)
    assert "sqlite" not in " ".join(allow)


def test_ignore_patterns_exclude_secrets_and_state():
    ignores = " ".join(DATASET_IGNORE_PATTERNS)
    assert ".venv" in ignores and "*.sqlite3*" in ignores and "*.tmp" in ignores


def test_secret_scan_finds_tokens_and_local_paths():
    text = "token hf_abcdefghijklmnopqrstuvwxyz123456 and /Users/nguyenhuyvu/x"
    hits = scan_for_secrets(text)
    assert len(hits) == 2
    assert any("token-like" in h for h in hits)
    assert any("local path" in h for h in hits)


def test_secret_scan_clean_text():
    assert scan_for_secrets("plain Vietnamese text, no secrets here") == []


def test_dataset_card_contains_reproducibility_info():
    card = render_dataset_card({
        "repo_id": "wheevu/ct219-vietnamese-raw-400k",
        "source_dataset": "VTSNLP/vietnamese_curated_dataset",
        "source_split": "train",
        "source_revision": "b81fcce58945970117a1b56d50ec81be2628a5c3",
        "seed": 42,
        "source_documents_inspected": 12_200_000,
        "dedup_backend": "sqlite",
        "duplicate_count": 5,
        "train_count": 360_000,
        "validation_count": 20_000,
        "test_count": 20_000,
        "release_dir": "ct219-400k-v1",
        "release_version": "v1",
    })
    assert "VTSNLP/vietnamese_curated_dataset" in card
    assert "b81fcce58945970117a1b56d50ec81be2628a5c3" in card
    assert "360,000" in card
    assert "private" in card
    assert "next-token" in card


# ------------------------------------------------------------------ model dir

def _fixture_model(tmp_path, with_token_in_config=False, missing_weights=False):
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    config = {"architectures": ["GPT2LMHeadModel"]}
    if with_token_in_config:
        config["_secret"] = "hf_abcdefghijklmnopqrstuvwxyz123456"
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if not missing_weights:
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 128)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (model_dir / "vocab.txt").write_text("vocab", encoding="utf-8")
    return model_dir


def test_model_validate_accepts_complete_checkpoint(tmp_path):
    report = validate_model_dir(_fixture_model(tmp_path))
    assert report["ok"] is True
    assert report["errors"] == []


def test_model_validate_rejects_missing_weights(tmp_path):
    report = validate_model_dir(_fixture_model(tmp_path, missing_weights=True))
    assert report["ok"] is False
    assert any("weights" in e for e in report["errors"])


def test_model_validate_rejects_secrets_in_config(tmp_path):
    report = validate_model_dir(_fixture_model(tmp_path, with_token_in_config=True))
    assert report["ok"] is False
    assert any("config.json" in e for e in report["errors"])


def test_model_validate_rejects_empty_dir(tmp_path):
    report = validate_model_dir(tmp_path / "nothing")
    assert report["ok"] is False


# ------------------------------------------------------------------ upload wiring

def test_upload_dataset_uses_dataset_repo_type_and_private(tmp_path, monkeypatch):
    calls = {}

    def fake_create_repo(self, **kwargs):
        calls["create"] = kwargs

    def fake_upload_folder(**kwargs):
        calls["upload"] = kwargs

    class FakeInfo:
        sha = "abc123"

    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", fake_create_repo)
    monkeypatch.setattr("huggingface_hub.upload_folder", fake_upload_folder)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.repo_info",
        lambda self, repo_id, repo_type: FakeInfo())

    release_dir = tmp_path / "release"
    (release_dir / "processed").mkdir(parents=True)
    (release_dir / "manifest.json").write_text("{}", encoding="utf-8")
    sha = upload_dataset("wheevu/ct219-vietnamese-raw-400k", release_dir, "# card",
                         private=True)
    assert sha == "abc123"
    assert calls["create"]["repo_type"] == "dataset"
    assert calls["create"]["private"] is True
    assert calls["upload"]["repo_type"] == "dataset"
    assert "rejected_examples.csv" not in calls["upload"]["allow_patterns"]
    assert (release_dir / "README.md").read_text("utf-8") == "# card"


def test_upload_model_uses_model_repo_type(tmp_path, monkeypatch):
    calls = {}

    def fake_create_repo(self, **kwargs):
        calls["create"] = kwargs

    def fake_upload_folder(**kwargs):
        calls["upload"] = kwargs

    class FakeInfo:
        sha = "def456"

    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", fake_create_repo)
    monkeypatch.setattr("huggingface_hub.upload_folder", fake_upload_folder)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.repo_info",
        lambda self, repo_id, repo_type: FakeInfo())

    model_dir = _fixture_model(tmp_path)
    upload_model("wheevu/ct219-vietnamese-next-token-model", model_dir, "# card")
    assert calls["create"]["repo_type"] == "model"
    assert calls["upload"]["repo_type"] == "model"


def test_dry_run_performs_no_network(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network call in dry-run")

    monkeypatch.setattr("huggingface_hub.HfApi.create_repo", boom)
    monkeypatch.setattr("huggingface_hub.upload_folder", boom)
    assert upload_dataset("x/y", tmp_path, "# card", dry_run=True) is None
    assert upload_model("x/y", tmp_path, "# card", dry_run=True) is None
