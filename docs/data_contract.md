# Downstream data contract

This document is the handoff contract for the tokenizer and model teammates.
It describes exactly what the preprocessing outputs contain, what guarantees
hold, and what responsibilities remain with the consuming side.

## 1. Output files and formats

Everything is written UTF-8, under the run's `--output-dir` (default
`data/processed/`).

| File | Format | Content |
| --- | --- | --- |
| `processed/train.parquet` | Apache Parquet | one row per accepted document |
| `processed/validation.parquet` | Apache Parquet | same schema |
| `processed/test.parquet` | Apache Parquet | same schema |
| `processed/train.txt` | JSONL | one JSON-escaped text per line |
| `processed/validation.txt` | JSONL | same |
| `processed/test.txt` | JSONL | same |
| `statistics/*.json, *.csv` | JSON / CSV | run statistics (see README) |
| `samples/before_after_examples.csv` | CSV | truncated display pairs |
| `manifest.json` | JSON | reproducibility manifest (see README) |

The `.txt` files are **JSONL, not plain text**. This is a hard contract: raw
text contains newlines and blank lines, so only a JSON-escaped-per-line format
preserves document boundaries unambiguously. `json.loads` each line to recover
one full document, newlines and paragraphs included.

## 2. Parquet schema

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | string | source document id (deduplicated) |
| `domain` | string | source domain, `"unknown"` when absent |
| `text` | string | cleaned raw text, NFC-normalized |
| `text_hash` | string | SHA-256 hex of `text` |
| `character_count` | int64 | `len(text)` in Unicode code points |

The JSONL files contain only the text (one `json.dumps(text,
ensure_ascii=False)` value per line); ids/domains live in the parquet files.

## 3. Guarantees downstream code may rely on

- **Encoding:** all files are UTF-8. Text is NFC-normalized; Vietnamese
  diacritics are preserved (never stripped).
- **Document boundaries:** each JSONL line is exactly one document; `\n` inside
  a line is a line break inside that document. Parquet rows equal JSONL lines
  in order and content (`text` column == decoded line). Verified per run by
  tests and by the audit.
- **Deduplication:** no two rows share an `id`, and no two rows share a
  `text_hash`, within the scanned population. These guarantees hold **across
  splits**: a document (by id or normalized text) appears in at most one of
  train/validation/test. The guarantee is backend-agnostic: `memory` and
  `sqlite` deduplication produce identical accept/reject decisions (verified
  byte-identical outputs on smoke, 200k, and 500k scans), and the sqlite
  backend enforces uniqueness with PRIMARY KEY columns so the state survives
  restarts and scales to full-dataset scans.
- **Splits:** assignment is deterministic per `(id, seed)`. With the default
  ratios the splits are train 90% / validation 5% / test 5%. Split counts are
  recorded in `preprocessing_stats.json`.
- **Normalization already performed:** line endings to LF; repeated spaces
  collapsed; tabs to single space; at most one consecutive blank line; leading
  and trailing whitespace trimmed; control characters removed (LF and tab
  kept). No lowercase-ing, no punctuation or number removal, no stemming, no
  word segmentation, no tokenization.
- **Rejection rules:** by default, documents with >= 3 replacement characters
  (`encoding_corruption`) and documents dominated by unusual Unicode
  (`binary_or_invalid_content`) are removed and counted; every other quality
  signal is audit-only (see section 8). Long documents are never removed
  solely for length.
- **Reproducibility:** `manifest.json` records the dataset, resolved revision,
  seed, configuration, environment, output checksums, row counts, and dedup
  backend statistics. The output of a run can be reconstructed or compared
  exactly if the dataset revision is pinned (`--dataset-revision`).

## 4. Fields that may change between versions

- `character_count` and `text_hash` are derived and may change if the cleaning
  rules change.
- The parquet schema may gain advisory columns (for example quality-audit
  signals) in future runs; consuming code should select columns explicitly
  rather than assuming a fixed column set.
- `statistics/*` file names are stable, but their exact keys may grow.
- The `.txt` JSONL content format will not change.

## 5. Responsibilities left to the tokenizer / model members

- Tokenization, subword splitting, and `input_ids` conversion.
- Sequence chunking and context-window selection (the pipeline intentionally
  does not chunk long documents; the longest accepted documents exceed any
  realistic window).
- Vocabulary and tokenizer training, model training and evaluation.
- Deciding whether to filter by length thresholds for training efficiency;
  the parquet `character_count` column exists for that.

## 6. Loading examples

```python
# Recommended: parquet with pandas
import pandas as pd
train = pd.read_parquet("data/processed/processed/train.parquet")
docs = train["text"].tolist()

# Streaming-friendly: JSONL, one document per line
import json
with open("data/processed/processed/train.txt", encoding="utf-8") as fh:
    for line in fh:
        document = json.loads(line)   # full document, newlines included

# PyArrow for row-group iteration on very large splits
import pyarrow.parquet as pq
table = pq.read_table("data/processed/processed/train.parquet")
for batch in table.to_batches():
    for text in batch.column("text").to_pylist():
        ...
```

## 7. Near-duplicate caveat

Exact duplicates are removed. Near-duplicates (SimHash similarity) are
audited, not removed: the audit writes `data/audit/near_duplicate_pairs.csv`
when `--near-duplicates` is enabled. Near-duplicate leakage between splits is
possible in principle and is reported by the audit; if the tokenizer/model
side cares, re-run the audit after a fresh run and inspect the report.

## 8. Raw text vs corrupted text

The pipeline distinguishes three tiers:

1. **Raw text** - everything accepted and exported. Still contains web noise
   that cleaning conservatively preserves (scraped menus, URLs, mixed
   Vietnamese-English, occasional stray format characters).
2. **Flagged but accepted** - documents that trip audit-only indicators
   (very low Vietnamese-character ratio, VNI patterns, foreign-script
   dominance, page-dump lines). These appear in the manual review packet at
   `data/review/quality_review.csv` with a `proposed_decision` and empty
   `human_decision`/`reviewer_notes` columns. The team must review this
   packet before finalizing language-quality thresholds; no automatic
   language detection is trusted.
3. **Rejected** - documents removed by enabled rules and counted per reason
   in `statistics/removal_counts.csv`; up to 20 examples per reason in
   `samples/rejected_examples.csv`.

Rules that remove data by default: `encoding_corruption` (>= 3 U+FFFD) and
`binary_or_invalid_content` (unusual-Unicode soup). Rules that are audit-only
by default: `foreign_script_dominant` and `concatenated_dump`. VNI detection
is audit-only because letter+question-mark patterns are also produced by JS
ternaries, URL query strings, and no-space question marks in real data.
