# Vietnamese NLP Preprocessing Pipeline

Streaming preprocessing for raw Vietnamese text, part of the university NLP final
project (next-token generation from raw Vietnamese text). This repository covers
only the **data preprocessing** responsibility: converting the source dataset
into clean, model-ready raw text. Tokenization, model training, and evaluation
are handled by other team members.

Source dataset: [VTSNLP/vietnamese_curated_dataset](https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset)

## 1. What the pipeline does

- Loads a Hugging Face dataset with **streaming** by default (never loads the
  whole dataset into RAM).
- Applies **conservative Vietnamese text cleaning** (see section 7).
- Rejects unusable documents, recording a removal reason for every one.
- Detects duplicate IDs and exact duplicate normalized texts (SHA-256 hashes).
- Samples a deterministic subset with a seeded reservoir sampler (no naive
  first-N bias).
- Splits deterministically into train / validation / test (90 / 5 / 5 default).
- Exports clean raw text as parquet + JSONL files, plus statistics, domain
  distributions, removal counts, and before/after samples.
- Saves the effective configuration with the outputs for reproducibility.

## 2. What it intentionally does not do

- No tokenization, subword splitting, word segmentation, or `input_ids`.
- No model training, fine-tuning, evaluation, perplexity, or generation.
- No sequence chunking or context-window selection.
- No lowercase-ing, accent removal, punctuation removal, stemming, or
  lemmatization. Text stays raw: diacritics, capitalization, numbers, and
  paragraph structure are preserved because they are useful for language
  modelling.

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Tests: `pip install pytest` (included in
`requirements.txt`) then `python -m pytest`.

## 4. Smoke test

```bash
python -m src.data.preprocess --mode smoke --max-documents 2000 --seed 42
```

The `smoke` mode targets ~2,000 accepted documents and scans at most 50,000
source documents. Output goes to `data/processed/` by default. Add
`--overwrite` to replace existing outputs.

## 5. Larger preprocessing jobs

```bash
# Development run (~20,000 documents)
python -m src.data.preprocess --mode development --seed 42

# Final run (~50,000 documents, scans the whole stream)
python -m src.data.preprocess --mode final --seed 42 --overwrite

# Custom run: different dataset, no scan limit, tuned thresholds
python -m src.data.preprocess --dataset VTSNLP/vietnamese_curated_dataset \
  --mode final --max-documents 50000 --scan-limit 0 \
  --min-chars 30 --max-chars 10000 --train-ratio 0.9 --val-ratio 0.05 \
  --output-dir data/processed --seed 42 --overwrite
```

All options are visible with `python -m src.data.preprocess --help`.

### Exact deduplication backends

| Backend | State | Memory | Use when |
| --- | --- | --- | --- |
| `memory` (default) | two Python sets | ~126 MiB per 500k candidates, unbounded | bounded scans |
| `sqlite` | on-disk SQLite with PRIMARY KEY columns, WAL, batched commits | flat in RAM; db file on disk | full-dataset scans |

```bash
python -m src.data.preprocess --mode final --dedup-backend sqlite --dedup-dir data/dedup
```

Both backends make identical accept/reject decisions (verified: byte-identical
outputs on smoke and 200k/500k scans). The sqlite database is created fresh
per run, lives in `--dedup-dir`, and is deleted after a successful run unless
`--keep-dedup-db` is given. The backend choice and database statistics are
recorded in `manifest.json` (`dedup` key).

### Conservative rejection rules

Four optional rules remove only clearly unusable text. Thresholds are visible
in the configuration and every rejection is counted:

| Rule | Default | Trigger (defaults) |
| --- | --- | --- |
| `encoding_corruption` | reject | >= 3 replacement characters (U+FFFD) |
| `binary_or_invalid_content` | reject | >= 8 unusual Unicode chars AND >= 0.5% of chars (icon fonts, bidi marks, ZWSP soup) |
| `foreign_script_dominant` | audit-only | non-Latin script (Armenian/Arabic/Greek/Cyrillic/CJK/...) >= 30% of chars |
| `concatenated_dump` | audit-only | a single line > 20,000 characters |

Disable or enable per rule with `--reject-encoding-corruption` /
`--no-reject-encoding-corruption` (same pattern for the others).

Deliberate non-decisions, all measured against real data:

- **VNI mojibake is audit-only.** Letter+question-mark patterns also appear in
  JS ternaries, URL query strings, and no-space question marks, so automatic
  VNI detection is unreliable (the corpus contains all three).
- **Pure English is not rejected** by any rule (Latin script, no corruption
  markers); it is flagged `very_low_vietnamese` in the review packet.
- **Mixed Vietnamese-English is never rejected** for containing English.
- **Long documents are never rejected for being long**; only the page-dump
  signature triggers `concatenated_dump`.
- **No automatic VNI-to-Unicode conversion** is attempted.

Rejected documents are recorded (capped) in
`samples/rejected_examples.csv` with id, domain, reason, and a truncated
preview.

### Pinning the dataset revision

Pass `--dataset-revision <commit-sha-or-tag>` to pin the exact dataset version.
Every run also records the resolved revision in `manifest.json` so results can
be compared across runs even without an explicit pin.

### Quality audit (read-only)

After a run, audit the outputs without touching them:

```bash
python -m src.data.audit --output-dir data/processed --audit-dir data/audit \
  --seed 42 --near-duplicates
```

Produces per-document quality signals, extreme-metric review samples, domain
drift (inspected -> accepted -> retained -> splits), length-outlier previews,
and (with `--near-duplicates`) a SimHash near-duplicate report. The audit
flags documents; it never removes or modifies them.

### Manual review packet

The audit also exports every flagged document to `data/review/`:

- `quality_review.csv` - all metrics plus a truncated preview
- `quality_review.html` - readable HTML table
- `quality_review_decisions.template.csv` - fill-in template with
  `proposed_decision`, empty `human_decision` and `reviewer_notes` columns

802 of 20,000 development documents were flagged (4.0%); 50 proposed rejects.
The language-quality thresholds (`--min-non-latin-share`, VNI handling) must
be finalized from human review of this packet, not from automatic detection.

### Modes

| Mode | Accepted docs | Source scan limit |
| --- | --- | --- |
| `smoke` | 2,000 | 50,000 |
| `development` | 20,000 | 500,000 |
| `final` | 50,000 | unlimited (whole stream) |

Values are defaults, not hard-coded: `--max-documents` and `--scan-limit`
override them. `--scan-limit 0` means unlimited.

## 6. Output file formats

```
data/processed/
├── processed/
│   ├── train.parquet        # id, domain, text, text_hash, character_count
│   ├── validation.parquet
│   ├── test.parquet
│   ├── train.txt            # JSONL: one JSON-escaped document per line
│   ├── validation.txt
│   └── test.txt
├── statistics/
│   ├── preprocessing_stats.json   # effective config, counts, split counts
│   ├── domain_distribution.csv    # inspected vs accepted per domain
│   ├── removal_counts.csv         # every rejection reason, counted
│   └── length_statistics.json     # min/max/mean/median/p90/p95/p99
└── samples/
    └── before_after_examples.csv  # truncated display pairs (10 by default)
```

Plus `manifest.json` beside `processed/`: dataset name, requested and resolved
revision, seed, effective configuration, Python/dependency versions, git
commit (when the repo has one), per-file SHA-256 checksums, row counts, and
runtime. See `docs/data_contract.md` for the full handoff contract.

The `.txt` files are **JSONL** (one JSON-escaped document per line). This is
deliberate: raw text contains newlines and blank lines, so a plain-text format
could not separate documents reliably. Downstream readers load each line and
`json.loads` it; multiline documents round-trip exactly. The `.parquet` files
carry the same text plus metadata (`text_hash`, `character_count`).

## 7. Cleaning rules

Reject:
- missing or whitespace-only text (`empty_text`)
- text with no meaningful content, i.e. no letter or digit (`invalid_text`)
- text shorter than `--min-chars` (default 20) (`too_short`)
- text longer than `--max-chars`, when set (`too_long`)
- duplicate `id` values (`duplicate_id`)
- exact duplicate normalized texts (`duplicate_text`)

Normalize:
- Unicode to NFC (Vietnamese diacritics preserved)
- line endings to LF (`\r\n` / `\r` -> `\n`)
- repeated spaces -> single space, tabs -> space
- at most one consecutive blank line (paragraph boundaries preserved)
- trim leading/trailing whitespace, strip trailing spaces per line

Keep:
- punctuation, capitalization, numbers, paragraph structure, all valid
  Vietnamese/Unicode symbols. Control characters are removed; LF and tab stay.

Every rejected document increments a named counter, so nothing is discarded
silently.

## 8. Splitting method

Each accepted document is assigned a split with a **stable hash**:

```python
value = sha256("split|{seed}|{doc_id}") / 2^64
train        if value < train_ratio
validation   if value < train_ratio + val_ratio
test         otherwise
```

The same document always lands in the same split for a given seed, across runs
and machines. Duplicates are removed before splitting, so no ID or normalized
text appears in two splits. Ratios default to 90 / 5 / 5.

## 9. Known limitations

- **Sampling:** the reservoir sampler is uniform over everything it scans. With
  a scan limit (smoke/development) the sample is uniform over that prefix, which
  may not represent the whole dataset. `final` scans the whole stream. Domain
  stratification is not guaranteed; the inspected vs accepted domain
  distribution is recorded so bias is visible, not hidden.
- **Deduplication scope:** duplicate hashes are tracked for every accepted
  candidate within the scanned range. Whole-dataset runs keep hashes in memory;
  very large runs use more RAM (one 32-byte digest per candidate).
- **Determinism:** results are stable given the same dataset version and seed.
  Hugging Face streaming order is fixed per dataset version, but a dataset
  update can change sampling results.
- **Streaming fallback:** if the stream yields fewer usable documents than the
  target, the run succeeds with fewer accepted documents and prints a warning.

## 10. Consuming the output (tokenizer / model teammates)

- **Raw text:** read the `.txt` files line by line; `json.loads` each line to
  get one document. Feed this directly into your tokenizer.
- **Metadata:** the `.parquet` files carry `id`, `domain`, `text_hash`, and
  `character_count` alongside the text, for lineage and filtering.
- Splits are guaranteed disjoint by both `id` and normalized text. Use
  `train.parquet` / `train.txt`, `validation.*`, `test.*` as-is.
- Do not re-clean: text is already NFC-normalized and deduplicated.
