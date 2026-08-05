"""Tests for the quality-audit signals and sample selection."""

from data.audit import (
    compute_quality_signals,
    domain_drift,
    load_rows,
    longest_documents_preview,
    select_audit_samples,
)

TEXT = (
    "Hà Nội ngày 02/09/1945. Chủ tịch Hồ Chí Minh đọc bản Tuyên ngôn Độc lập.\n"
    "Đây là đoạn văn thứ hai với nội dung khác nhau.\n\n"
    "Đoạn văn cuối cùng để kiểm tra tỷ lệ ký tự."
)


def _doc(seed, extra=""):
    text = TEXT + extra
    return {
        "split": "train",
        "id": f"doc-{seed}",
        "domain": "news",
        "text_hash": "x" * 64,
        "text": text,
        "signals": compute_quality_signals(text),
    }


def test_signals_on_clean_vietnamese_text():
    signals = compute_quality_signals(TEXT)
    assert signals["character_count"] == len(TEXT)
    assert signals["line_count"] == 4  # trailing blank line from "\n\n" split
    assert signals["paragraph_count"] == 2
    assert signals["url_count"] == 0
    assert signals["url_char_ratio"] == 0.0
    assert signals["replacement_char_count"] == 0
    assert signals["html_tag_count"] == 0
    assert signals["alpha_ratio"] > 0.5
    assert signals["digit_ratio"] > 0.0  # "02/09/1945"
    assert signals["punctuation_ratio"] > 0.0
    assert signals["vietnamese_char_count"] > 0
    assert signals["vietnamese_diacritic_ratio"] > 0.0
    assert signals["unusual_unicode_count"] == 0


def test_signals_detect_urls_and_html():
    text = "Trang: <div class=\"menu\">menu</div> link https://example.com/a?b=1 www.demo.vn/x"
    signals = compute_quality_signals(text)
    assert signals["url_count"] == 2
    assert signals["url_char_ratio"] > 0.05
    assert signals["html_tag_count"] == 2


def test_signals_detect_repeated_lines_and_paragraphs():
    text = ("Một dòng lặp lại.\nMột dòng lặp lại.\nMột dòng lặp lại.\n"
            "Dòng duy nhất.")
    signals = compute_quality_signals(text)
    assert signals["repeated_line_ratio"] == 0.75


def test_signals_detect_replacement_and_control_chars():
    text = "abc\ufffddef\x00ghi\x1f"
    signals = compute_quality_signals(text)
    assert signals["replacement_char_count"] == 1
    assert signals["unusual_unicode_count"] == 2


def test_signals_detect_repeated_ngrams():
    text = ("học sinh học sinh học sinh học sinh học sinh "
            "và một câu hoàn toàn khác hẳn ở phía sau")
    signals = compute_quality_signals(text)
    assert signals["repeated_ngram_ratio"] > 0.5


def test_max_line_length():
    text = "ngắn\n" + "dài " * 100
    signals = compute_quality_signals(text)
    assert signals["max_line_length"] > 100


def test_sample_selection_is_deterministic_and_ranked():
    rows = []
    for i in range(200):
        text = "văn bản mẫu " * (i + 1)
        rows.append({
            "split": "train", "id": f"d-{i}", "domain": "d", "text_hash": "h",
            "text": text, "signals": compute_quality_signals(text),
        })
    first = select_audit_samples(rows, seed=42)
    second = select_audit_samples(rows, seed=42)
    assert [s["id"] for s in first] == [s["id"] for s in second]

    random_ids = {s["id"] for s in first if "random" in s["reasons"]}
    assert len(random_ids) == 100
    longest_ids = {s["id"] for s in first if "longest" in s["reasons"]}
    top50 = sorted(rows, key=lambda d: d["signals"]["character_count"], reverse=True)[:50]
    assert {d["id"] for d in top50} == longest_ids


def test_longest_preview_truncates_text_only():
    rows = [_doc(i, extra="thêm " * 300) for i in range(10)]
    preview = longest_documents_preview(rows, take=3)
    assert len(preview) == 3
    assert len(preview[0]["preview"]) < 500
    assert "…" in preview[0]["preview"]


def test_domain_drift_roundtrip(tmp_path):
    # Build a minimal processed dir by hand.
    import pyarrow as pa
    import pyarrow.parquet as pq

    processed = tmp_path / "processed"
    stats_dir = tmp_path / "statistics"
    processed.mkdir()
    stats_dir.mkdir()

    schema = pa.schema([pa.field("id", pa.string()), pa.field("domain", pa.string())])
    pq.write_table(pa.Table.from_pylist(
        [{"id": "a", "domain": "news"}, {"id": "b", "domain": "news"},
         {"id": "c", "domain": "wiki"}], schema=schema),
        processed / "train.parquet")
    pq.write_table(pa.Table.from_pylist([], schema=schema),
                   processed / "validation.parquet")
    pq.write_table(pa.Table.from_pylist([], schema=schema),
                   processed / "test.parquet")

    stats_dir.joinpath("preprocessing_stats.json").write_text(
        '{"source_documents_inspected": 100}', encoding="utf-8")
    stats_dir.joinpath("domain_distribution.csv").write_text(
        "domain,inspected,accepted\nnews,70,70\nwiki,30,30\n", encoding="utf-8")

    result = domain_drift(processed, stats_dir)
    assert result["summary"]["retained_total"] == 3
    news = next(r for r in result["rows"] if r["domain"] == "news")
    assert news["retained_n"] == 2
    assert news["retained_pct"] == 66.6667
    assert abs(news["drift_retained_vs_inspected_pp"]) > 0


def test_load_rows_reads_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    processed = tmp_path / "processed"
    processed.mkdir()
    schema = pa.schema([pa.field("id", pa.string()), pa.field("domain", pa.string()),
                        pa.field("text", pa.string()), pa.field("text_hash", pa.string()),
                        pa.field("character_count", pa.int64())])
    pq.write_table(pa.Table.from_pylist(
        [{"id": "1", "domain": "news", "text": "Văn bản một.",
          "text_hash": "h1", "character_count": 10}], schema=schema),
        processed / "train.parquet")
    rows = load_rows(processed)
    assert len(rows) == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["text"] == "Văn bản một."
