"""Hugging Face publication helpers: cards, uploads, and verification.

The 400k processed corpus is a DATASET (repo_type="dataset"); a trained
checkpoint would be a MODEL (repo_type="model"). These are never conflated
here. Everything defaults to private repositories, nothing is claimed about
licensing (the source card has none), and no secrets or local paths are ever
uploaded. Pure logic lives here so tests can run without network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .sampling import stable_float

# ------------------------------------------------------------------ repo ids

DATASET_REPO_SUFFIX = "ct219-vietnamese-raw-400k"
MODEL_REPO_SUFFIX = "ct219-vietnamese-next-token-model"

TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"api_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
]
LOCAL_PATH_PATTERNS = [
    re.compile(r"/Users/[\w.-]+"),
    re.compile(r"\.venv/"),
]


def propose_repo_id(namespace: str, kind: str) -> str:
    """f'{namespace}/{suffix}' for kind in ('dataset', 'model')."""
    suffix = DATASET_REPO_SUFFIX if kind == "dataset" else MODEL_REPO_SUFFIX
    return f"{namespace}/{suffix}"


# ------------------------------------------------------------------ patterns

def dataset_allow_patterns() -> list[str]:
    """Only useful release artifacts; never raw dumps, db, caches, or samples."""
    return [
        "processed/train.parquet",
        "processed/validation.parquet",
        "processed/test.parquet",
        "manifest.json",
        "statistics/preprocessing_stats.json",
        "statistics/domain_distribution.csv",
        "statistics/removal_counts.csv",
        "statistics/length_statistics.json",
        "checksums.txt",
        "README.md",
        "data_contract.md",
    ]


DATASET_IGNORE_PATTERNS = [
    "**/.venv/**",
    "**/__pycache__/**",
    "**/*.tmp",
    "**/*.sqlite3*",
    "**/rejected_examples.csv",
    "**/before_after_examples.csv",
    "**/RELEASE_COMPLETE",
]


def scan_for_secrets(text: str) -> list[str]:
    """Return every secret-like or local-path pattern found in the text."""
    hits: list[str] = []
    for pattern in TOKEN_PATTERNS:
        for match in pattern.findall(text):
            hits.append(f"token-like: {match[:6]}...{match[-4:]}")
    for pattern in LOCAL_PATH_PATTERNS:
        for match in pattern.findall(text):
            hits.append(f"local path: {match}")
    return hits


# ------------------------------------------------------------------ model dir

def model_required_files() -> dict[str, list[str]]:
    """File sets required for a reloadable Transformers-style checkpoint."""
    return {
        "config": ["config.json"],
        "weights": ["model.safetensors", "pytorch_model.bin", "tf_model.h5"],
        "tokenizer": ["tokenizer.json", "vocab.txt", "tokenizer_config.json",
                      "special_tokens_map.json"],
        "generation": ["generation_config.json"],
    }


def validate_model_dir(model_dir: Path) -> dict:
    """Structural validation of a checkpoint directory (no weights loading).

    Checks required files exist and are non-empty, and that no config or
    tokenizer file contains secrets or local paths. Loading-based checks
    (transformers) are reported as optional and skipped when unavailable.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not model_dir.exists() or not model_dir.is_dir():
        return {"ok": False, "errors": [f"model directory does not exist: {model_dir}"],
                "warnings": []}
    required = model_required_files()
    for group, names in required.items():
        if group == "weights":
            present = [n for n in names if (model_dir / n).exists()]
            if not present:
                errors.append("no model weights "
                              f"(expected one of {', '.join(names)})")
            else:
                for name in present:
                    if (model_dir / name).stat().st_size == 0:
                        errors.append(f"weight file {name} is empty")
        elif group == "generation":
            continue  # optional
        else:
            for name in names:
                if (model_dir / name).exists():
                    break
            else:
                errors.append(f"missing {group} files "
                              f"(expected one of {', '.join(names)})")
    for path in model_dir.iterdir():
        if not path.is_file() or path.name.endswith((".bin", ".safetensors", ".h5")):
            continue
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        hits = scan_for_secrets(text)
        if hits:
            errors.append(f"{path.name}: {hits[0]}")
    try:
        import transformers  # noqa: F401

        warnings.append("transformers installed; loading checks possible")
    except ImportError:
        warnings.append("transformers not installed; loading checks skipped")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


# ------------------------------------------------------------------ cards

DATASET_CARD_TEMPLATE = """\
---
language:
- vi
tags:
- vietnamese
- next-token
- language-modeling
license: unknown
---

# {repo_id}

Bộ dữ liệu văn bản tiếng Việt thô đã tiền xử lý, dùng để huấn luyện
next-token language model (CT219 - NLP final project).

## Nguồn dữ liệu

- Source dataset: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- Source split: `{source_split}`
- Pinned source revision: `{source_revision}`
- Source licence: không công bố - repo này được tạo ở chế độ **private** vì lý do đó.

## Mục đích

Huấn luyện next-token language model cho tiếng Việt.

## Cách chọn 400k văn bản

- Seed: `{seed}`
- Deterministic reservoir sampling trên toàn bộ stream nguồn ({source_documents_inspected:,} văn bản đã duyệt).
- Không chọn N văn bản đầu tiên; mẫu đồng đều trên phần đã quét.

## Tiền xử lý và chuẩn hóa

- NFC Unicode, giữ nguyên dấu tiếng Việt, chữ hoa, số, dấu câu, cấu trúc đoạn.
- Kết thúc dòng về LF, khoảng trắng lặp gộp lại, tối đa một dòng trống liên tiếp.
- Không tokenization, không hạ chữ thường, không bỏ dấu.

## Khử trùng lặp

- Trùng `id` và trùng văn bản chính xác sau chuẩn hóa (SHA-256) đều bị loại.
- Backend: {dedup_backend}. Duplicate count: {duplicate_count}.

## Chia tập

- Phương pháp: stable hash của `id` và seed.
- train: {train_count:,} (90%), validation: {validation_count:,} (5%), test: {test_count:,} (5%).
- Không có văn bản nào xuất hiện ở hai tập (đã kiểm chứng theo id và text_hash).

## Schema

Các tệp parquet có cột: `id`, `domain`, `text`, `text_hash`, `character_count`.

## Chính sách lọc chất lượng

- Loại bỏ (mặc định): `encoding_corruption` (>= 3 ký tự U+FFFD),
  `binary_or_invalid_content` (Unicode bất thường >= 8 ký tự và >= 0.5%).
- Chỉ audit (không loại): `foreign_script_dominant`, `concatenated_dump`,
  tín hiệu tiếng Việt thấp, VNI patterns. Xem `data_contract.md` trong repo mã nguồn.
- Ngưỡng ngôn ngữ chưa được chốt bằng dữ liệu có nhãn; cần duyệt thủ công trước khi thay đổi.

## Hạn chế đã biết

- Mẫu đại diện cho phần stream đã quét ({source_documents_inspected:,} văn bản), không đảm bảo đại diện toàn bộ bộ dữ liệu gốc.
- Có thể chứa văn bản không phải tiếng Việt (ước lượng < 1% qua audit) và boilerplate từ web.
- Licence nguồn không rõ: dữ liệu này chỉ dùng trong phạm vi đồ án học tập.

## Tái lập

```bash
python -m src.data.preprocess --mode release-400k \
  --dataset {source_dataset} --dataset-split {source_split} \
  --dataset-revision {source_revision} --seed {seed} \
  --dedup-backend {dedup_backend} --output-dir data/releases/{release_dir}
```

Mã nguồn: https://github.com/wheevu/ct219-final-report (nhánh feat/400k-huggingface-release)
Checksums và row counts: xem `manifest.json` và `checksums.txt` trong repo này.
Release version: {release_version}
"""


def render_dataset_card(values: dict) -> str:
    return DATASET_CARD_TEMPLATE.format(**values)


# ------------------------------------------------------------------ uploads

def upload_dataset(
    repo_id: str,
    release_dir: Path,
    card: str,
    private: bool = True,
    dry_run: bool = False,
) -> str | None:
    """Create the private dataset repo and upload curated artifacts."""
    if dry_run:
        return None
    from huggingface_hub import HfApi, upload_folder

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private,
                    exist_ok=True)
    (release_dir / "README.md").write_text(card, encoding="utf-8")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(release_dir),
        allow_patterns=dataset_allow_patterns(),
        ignore_patterns=DATASET_IGNORE_PATTERNS,
        repo_type="dataset",
        commit_message=f"Release ct219-vietnamese-raw-400k v1 (400k docs)",
    )
    return str(HfApi().repo_info(repo_id, repo_type="dataset").sha)


def upload_model(
    repo_id: str,
    model_dir: Path,
    card: str,
    private: bool = True,
    dry_run: bool = False,
) -> str | None:
    """Create the private model repo and upload a validated checkpoint."""
    if dry_run:
        return None
    from huggingface_hub import HfApi, upload_folder

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private,
                    exist_ok=True)
    (model_dir / "README.md").write_text(card, encoding="utf-8")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        commit_message="Upload trained next-token model checkpoint",
    )
    return str(HfApi().repo_info(repo_id, repo_type="model").sha)


# ------------------------------------------------------------------ verify

def deterministic_sample_ids(rows: list[dict], seed: int, take: int = 5) -> list[dict]:
    """Pick `take` documents deterministically from the given rows."""
    ranked = sorted(rows, key=lambda r: stable_float("release_sample", seed, r["id"]))
    return ranked[:take]


def verify_dataset(
    repo_id: str,
    local_dir: Path,
    seed: int = 42,
    max_sample_rows: int = 3000,
) -> dict:
    """Remote verification: files, privacy, row counts, deterministic samples.

    Requires network and returns a dict; every check is reported, nothing is
    claimed without evidence.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    result: dict = {}
    result["repo_id"] = repo_id
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
        result["exists"] = True
        result["private"] = info.private
        result["sha"] = info.sha
    except Exception as exc:
        result["exists"] = False
        result["error"] = str(exc)[:200]
        return result
    result["files"] = sorted(api.list_repo_files(repo_id, repo_type="dataset"))

    from datasets import load_dataset

    manifest = json.loads((local_dir / "manifest.json").read_text("utf-8"))
    result["splits"] = {}
    all_mismatches = []
    for split in ("train", "validation", "test"):
        local_rows = []
        import pyarrow.parquet as pq

        table = pq.read_table(local_dir / "processed" / f"{split}.parquet")
        for i in range(min(table.num_rows, max_sample_rows)):
            local_rows.append({
                "id": table.column("id").to_pylist()[i],
                "text_hash": table.column("text_hash").to_pylist()[i],
                "text": table.column("text").to_pylist()[i],
            })
        entry: dict = {}
        ds = load_dataset(repo_id, split=split, streaming=True)
        entry["remote_rows"] = sum(1 for _ in ds)
        entry["expected_rows"] = manifest["outputs"][f"processed/{split}.parquet"]["row_count"]
        entry["row_count_matches"] = entry["remote_rows"] == entry["expected_rows"]
        # Deterministic sample comparison on the first rows (order is preserved).
        remote_rows = []
        ds = load_dataset(repo_id, split=split, streaming=True)
        for row in ds:
            remote_rows.append({
                "id": str(row["id"]),
                "text_hash": str(row["text_hash"]),
                "text": str(row["text"]),
            })
            if len(remote_rows) >= max_sample_rows:
                break
        local_sample = deterministic_sample_ids(local_rows, seed)
        remote_sample = deterministic_sample_ids(remote_rows, seed)
        mismatches = []
        for lr, rr in zip(local_sample, remote_sample):
            if lr != rr:
                mismatches.append(f"{lr['id']}: local vs remote differ")
        entry["sample_mismatches"] = mismatches
        all_mismatches.extend(mismatches)
        result["splits"][split] = entry
    result["all_row_counts_match"] = all(
        e["row_count_matches"] for e in result["splits"].values()
    )
    result["all_samples_match"] = not all_mismatches

    # Secret and local-path scan of remote text files.
    secrets: list[str] = []
    for filename in result["files"]:
        if filename.endswith((".md", ".json", ".txt", ".csv")):
            try:
                content = api.hf_hub_download(
                    repo_id, filename, repo_type="dataset"
                )
                text = Path(content).read_text("utf-8", errors="ignore")
                secrets.extend(scan_for_secrets(text))
            except Exception:
                pass
    result["remote_secret_hits"] = secrets
    result["ok"] = (
        result["exists"]
        and result["private"] is True
        and result["all_row_counts_match"]
        and result["all_samples_match"]
        and not secrets
    )
    return result
