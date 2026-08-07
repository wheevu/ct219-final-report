#!/usr/bin/env python3
"""Hugging Face release CLI: dataset upload, model upload, verification.

Dataset and model repositories are separate concepts here; a processed
corpus is a dataset, a trained checkpoint is a model, and the two are never
conflated.

Usage:
  python scripts/hf_release.py dataset --release-dir data/releases/ct219-400k-v1 [--repo-id NS/NAME] [--dry-run]
  python scripts/hf_release.py verify-dataset --release-dir data/releases/ct219-400k-v1 [--repo-id NS/NAME]
  python scripts/hf_release.py model-validate --model-dir PATH/TO/CHECKPOINT
  python scripts/hf_release.py model --model-dir PATH/TO/CHECKPOINT [--repo-id NS/NAME] [--dry-run]

Everything defaults to private repositories. Never pass a token on the
command line; the authenticated huggingface_hub session is used.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hf_publish import (  # noqa: E402
    DATASET_REPO_SUFFIX,
    MODEL_REPO_SUFFIX,
    propose_repo_id,
    render_dataset_card,
    upload_dataset,
    upload_model,
    validate_model_dir,
    verify_dataset,
)
from src.data.manifest import sha256_file  # noqa: E402


def _namespace() -> str:
    from huggingface_hub import HfApi

    return HfApi().whoami()["name"]


def _write_checksums(release_dir: Path) -> Path:
    """checksums.txt: relative path, sha256, bytes for every uploaded artifact."""
    lines = ["# ct219-vietnamese-raw-400k checksums (sha256  size  path)"]
    for rel in ("processed/train.parquet", "processed/validation.parquet",
                "processed/test.parquet", "manifest.json",
                "statistics/preprocessing_stats.json",
                "statistics/domain_distribution.csv",
                "statistics/removal_counts.csv",
                "statistics/length_statistics.json"):
        path = release_dir / rel
        if path.exists():
            lines.append(f"{sha256_file(path)} {path.stat().st_size} {rel}")
    out = release_dir / "checksums.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def cmd_dataset(args) -> int:
    release_dir = Path(args.release_dir)
    repo_id = args.repo_id or propose_repo_id(_namespace(), "dataset")
    print(f"dataset repo: {repo_id} (private={not args.public})")
    if args.public:
        print("WARNING: --public requested; the source card has no licence.")
    manifest = json.loads((release_dir / "manifest.json").read_text("utf-8"))
    stats = json.loads(
        (release_dir / "statistics" / "preprocessing_stats.json").read_text("utf-8"))
    values = {
        "repo_id": repo_id,
        "source_dataset": manifest["dataset"],
        "source_split": manifest["source_split"],
        "source_revision": manifest["resolved_dataset_revision"],
        "seed": manifest["seed"],
        "source_documents_inspected": stats["source_documents_inspected"],
        "dedup_backend": (manifest.get("dedup") or {}).get("backend", "?"),
        "duplicate_count": stats["duplicate_count"],
        "train_count": stats["split_counts"]["train"],
        "validation_count": stats["split_counts"]["validation"],
        "test_count": stats["split_counts"]["test"],
        "release_dir": release_dir.name,
        "release_version": "v1",
    }
    card = render_dataset_card(values)
    _write_checksums(release_dir)
    sha = upload_dataset(repo_id, release_dir, card, private=not args.public,
                         dry_run=args.dry_run)
    if sha:
        print(f"uploaded: {repo_id} @ {sha}")
    else:
        print("dry-run: no upload performed")
    return 0


def cmd_verify(args) -> int:
    release_dir = Path(args.release_dir)
    repo_id = args.repo_id or propose_repo_id(_namespace(), "dataset")
    result = verify_dataset(repo_id, release_dir, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_model_validate(args) -> int:
    report = validate_model_dir(Path(args.model_dir))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def cmd_model(args) -> int:
    model_dir = Path(args.model_dir)
    report = validate_model_dir(model_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        print("model directory failed validation; not uploading", file=sys.stderr)
        return 1
    repo_id = args.repo_id or propose_repo_id(_namespace(), "model")
    sha = upload_model(repo_id, model_dir, "# placeholder card", private=not args.public,
                       dry_run=args.dry_run)
    if sha:
        print(f"uploaded: {repo_id} @ {sha}")
    else:
        print("dry-run: no upload performed")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dataset", help="upload the 400k processed corpus")
    p.add_argument("--release-dir", required=True)
    p.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO_ID"))
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("verify-dataset", help="verify a remote dataset repo")
    p.add_argument("--release-dir", required=True)
    p.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO_ID"))
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("model-validate", help="validate a checkpoint directory")
    p.add_argument("--model-dir", required=True)
    p.set_defaults(func=cmd_model_validate)

    p = sub.add_parser("model", help="upload a validated checkpoint")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--repo-id", default=os.environ.get("HF_MODEL_REPO_ID"))
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_model)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
