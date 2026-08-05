"""Non-destructive quality audit over processed outputs.

Computes quality signals per document, selects review samples by extreme
metrics, analyzes domain drift across inspected / accepted / retained /
split populations, and describes length outliers. Nothing is removed or
modified: the audit only reads outputs and writes reports.

Run as:

    python -m src.data.audit --output-dir data/processed --audit-dir data/audit
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq
from tqdm import tqdm

from .config import Settings
from .export import SPLIT_NAMES, truncate_for_display, write_csv, write_json
from .quality_rules import (
    DEFAULT_REJECT_POLICY,
    DEFAULT_THRESHOLDS,
    REASON_BINARY,
    REASON_CONCATENATED,
    REASON_ENCODING,
    REASON_FOREIGN,
    non_latin_script_share,
    rule_checks,
    vni_pattern_count,
)
from .sampling import stable_float

URL_RE = re.compile(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+")
HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")

VIETNAMESE_CHARS = (
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


def compute_quality_signals(text: str) -> dict[str, float | int]:
    """One-pass quality signals for a document (cleaned text)."""
    n = len(text)
    lines = text.split("\n")
    non_empty_lines = [ln for ln in lines if ln.strip()]
    paragraphs = [p for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
    urls = URL_RE.findall(text)
    html_tags = HTML_TAG_RE.findall(text)

    categories = Counter(unicodedata.category(ch) for ch in text)

    def prefix_sum(prefix: str) -> int:
        return sum(v for k, v in categories.items() if k.startswith(prefix))

    whitespace_chars = sum(1 for ch in text if ch in " \t\n" or unicodedata.category(ch) == "Zs")
    # Cc except normal line breaks and tabs is "unusual" for cleaned text.
    unusual = (
        prefix_sum("C") - text.count("\n") - text.count("\t") - text.count("\r")
    )
    return {
        "character_count": n,
        "line_count": len(lines),
        "paragraph_count": len(paragraphs),
        "url_count": len(urls),
        "url_char_ratio": _ratio(sum(len(u) for u in urls), n),
        "repeated_line_ratio": _repetition_ratio(non_empty_lines),
        "repeated_paragraph_ratio": _repetition_ratio(paragraphs),
        "alpha_ratio": _ratio(prefix_sum("L"), n),
        "digit_ratio": _ratio(prefix_sum("N"), n),
        "whitespace_ratio": _ratio(whitespace_chars, n),
        "punctuation_ratio": _ratio(prefix_sum("P"), n),
        "replacement_char_count": text.count("\ufffd"),
        "html_tag_count": len(html_tags),
        "max_line_length": max((len(ln) for ln in lines), default=0),
        "unusual_unicode_count": unusual,
        "vietnamese_char_count": sum(1 for ch in text if ch in VIETNAMESE_CHARS)
        + prefix_sum("M"),
        "vietnamese_diacritic_ratio": _ratio(
            sum(1 for ch in text if ch in VIETNAMESE_CHARS) + prefix_sum("M"), n
        ),
        "repeated_ngram_ratio": _repeated_ngram_ratio(text),
        "non_latin_script_share": non_latin_script_share(text),
        "vni_pattern_count": vni_pattern_count(text),
    }

def _ratio(part: int, total: int) -> float:
    return round(part / total, 6) if total else 0.0


def _repetition_ratio(units: list[str]) -> float:
    """Fraction of unit instances that appear more than once."""
    if not units:
        return 0.0
    counts = Counter(units)
    duplicated = sum(c for c in counts.values() if c > 1)
    return round(duplicated / len(units), 6)


def _repeated_ngram_ratio(text: str, n: int = 3, token_cap: int = 5000) -> float:
    """Duplicate ratio of word n-grams over the first `token_cap` tokens."""
    tokens = text.split()[:token_cap]
    if len(tokens) < n:
        return 0.0
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    duplicated = sum(c for c in grams.values() if c > 1)
    return round(duplicated / len(grams), 6)


def _preview(text: str, limit: int = 200) -> str:
    """Single-line, safely truncated preview for CSV cells."""
    shown = text.replace("\n", "\\n").replace("\r", "\\r")[:limit]
    if len(text) > limit or "\n" in text[:limit]:
        shown += "…"
    return shown


def load_rows(processed_dir: Path) -> list[dict]:
    """Read all accepted documents from the split parquet files."""
    rows: list[dict] = []
    for split in SPLIT_NAMES:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path)
        ids = table.column("id").to_pylist()
        domains = table.column("domain").to_pylist()
        texts = table.column("text").to_pylist()
        hashes = table.column("text_hash").to_pylist()
        rows.extend(
            {
                "split": split,
                "id": ids[i],
                "domain": domains[i],
                "text_hash": hashes[i],
                "text": texts[i],
            }
            for i in range(table.num_rows)
        )
    return rows


# --------------------------------------------------------------------------
# Sample selection
# --------------------------------------------------------------------------

def _ranked(rows: list[dict], key, reverse: bool, take: int) -> list[dict]:
    return sorted(rows, key=key, reverse=reverse)[:take]


def select_audit_samples(rows: list[dict], seed: int) -> list[dict]:
    """Deterministic review-sample selection by extreme metrics."""
    def rng(doc: dict) -> float:
        return stable_float("audit_random", seed, doc["id"])

    selections: list[tuple[str, list[dict]]] = [
        ("random", _ranked(rows, rng, False, 100)),
        ("shortest", _ranked(rows, lambda d: d["signals"]["character_count"], False, 20)),
        ("longest", _ranked(rows, lambda d: d["signals"]["character_count"], True, 50)),
        ("highest_url_ratio", _ranked(rows, lambda d: d["signals"]["url_char_ratio"], True, 20)),
        ("highest_repeated_line_ratio",
         _ranked(rows, lambda d: d["signals"]["repeated_line_ratio"], True, 20)),
        ("highest_punctuation_ratio",
         _ranked(rows, lambda d: d["signals"]["punctuation_ratio"], True, 20)),
        ("lowest_alpha_ratio",
         _ranked(rows, lambda d: d["signals"]["alpha_ratio"], False, 20)),
        ("most_unusual_unicode",
         _ranked(rows, lambda d: d["signals"]["unusual_unicode_count"], True, 20)),
        ("html_heavy",
         _ranked(rows, lambda d: d["signals"]["html_tag_count"], True, 20)),
    ]

    reasons: dict[str, set[str]] = {doc["id"]: set() for doc in rows}
    for name, selected in selections:
        for doc in selected:
            reasons[doc["id"]].add(name)

    samples = []
    for doc in rows:
        if reasons[doc["id"]]:
            signals = doc["signals"]
            samples.append({
                "split": doc["split"],
                "id": doc["id"],
                "domain": doc["domain"],
                "reasons": "|".join(sorted(reasons[doc["id"]])),
                "character_count": signals["character_count"],
                "line_count": signals["line_count"],
                "url_count": signals["url_count"],
                "url_char_ratio": signals["url_char_ratio"],
                "repeated_line_ratio": signals["repeated_line_ratio"],
                "punctuation_ratio": signals["punctuation_ratio"],
                "alpha_ratio": signals["alpha_ratio"],
                "unusual_unicode_count": signals["unusual_unicode_count"],
                "html_tag_count": signals["html_tag_count"],
                "vietnamese_diacritic_ratio": signals["vietnamese_diacritic_ratio"],
                "preview": _preview(doc["text"]),
            })
    samples.sort(key=lambda s: (s["reasons"], s["id"]))
    return samples


def longest_documents_preview(rows: list[dict], take: int = 15) -> list[dict]:
    top = _ranked(rows, lambda d: d["signals"]["character_count"], True, take)
    return [
        {
            "rank": i + 1,
            "id": doc["id"],
            "domain": doc["domain"],
            "split": doc["split"],
            "character_count": doc["signals"]["character_count"],
            "line_count": doc["signals"]["line_count"],
            "paragraph_count": doc["signals"]["paragraph_count"],
            "url_count": doc["signals"]["url_count"],
            "url_char_ratio": doc["signals"]["url_char_ratio"],
            "repeated_line_ratio": doc["signals"]["repeated_line_ratio"],
            "repeated_paragraph_ratio": doc["signals"]["repeated_paragraph_ratio"],
            "html_tag_count": doc["signals"]["html_tag_count"],
            "preview": _preview(doc["text"], limit=400),
        }
        for i, doc in enumerate(top)
    ]


# --------------------------------------------------------------------------
# Domain drift across populations
# --------------------------------------------------------------------------

def domain_drift(processed_dir: Path, stats_dir: Path) -> dict:
    """Compare domain distribution: inspected -> accepted -> retained -> splits."""
    stats = json.loads((stats_dir / "preprocessing_stats.json").read_text("utf-8"))
    distribution = {}
    csv_path = stats_dir / "domain_distribution.csv"
    if csv_path.exists():
        lines = csv_path.read_text("utf-8").strip().splitlines()[1:]
        for line in lines:
            parts = line.split(",")
            distribution[parts[0]] = {"inspected": int(parts[1]), "accepted": int(parts[2])}

    retained: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    per_split: dict[str, Counter[str]] = {s: Counter() for s in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path, columns=["domain"])
        domains = table.column("domain").to_pylist()
        retained.update(domains)
        per_split[split].update(domains)
        split_counts[split] = len(domains)

    inspected_total = sum(d["inspected"] for d in distribution.values())
    accepted_total = sum(d["accepted"] for d in distribution.values())
    retained_total = sum(retained.values())

    def pct(count: int, total: int) -> float:
        return round(100.0 * count / total, 4) if total else 0.0

    domains = sorted(set(distribution) | set(retained))
    rows = []
    for domain in domains:
        insp = distribution.get(domain, {}).get("inspected", 0)
        acc = distribution.get(domain, {}).get("accepted", 0)
        ret = retained.get(domain, 0)
        row = {
            "domain": domain,
            "inspected_n": insp,
            "inspected_pct": pct(insp, inspected_total),
            "accepted_n": acc,
            "accepted_pct": pct(acc, accepted_total),
            "retained_n": ret,
            "retained_pct": pct(ret, retained_total),
            "drift_retained_vs_inspected_pp": round(
                pct(ret, retained_total) - pct(insp, inspected_total), 4
            ),
        }
        max_split_drift = 0.0
        for split in SPLIT_NAMES:
            count = per_split[split].get(domain, 0)
            row[f"{split}_n"] = count
            row[f"{split}_pct"] = pct(count, split_counts[split])
            if split_counts[split]:
                max_split_drift = max(
                    max_split_drift,
                    abs(pct(count, split_counts[split]) - pct(ret, retained_total)),
                )
        row["max_split_drift_pp"] = round(max_split_drift, 4)
        rows.append(row)

    def worst(rows_in: list[dict], key: str, metric: str) -> dict:
        if not rows_in:
            return {"domain": None, metric: 0.0}
        return max(rows_in, key=lambda r: abs(r[metric]))

    summary = {
        "inspected_total": inspected_total,
        "accepted_total": accepted_total,
        "retained_total": retained_total,
        "split_counts": dict(split_counts),
        "max_abs_drift_retained_vs_inspected_pp": worst(rows, "drift_retained_vs_inspected_pp", "drift_retained_vs_inspected_pp")["drift_retained_vs_inspected_pp"],
        "worst_domain_drift": worst(rows, "drift_retained_vs_inspected_pp", "drift_retained_vs_inspected_pp")["domain"],
        "max_abs_split_drift_pp": worst(rows, "max_split_drift_pp", "max_split_drift_pp")["max_split_drift_pp"],
        "worst_domain_split_drift": worst(rows, "max_split_drift_pp", "max_split_drift_pp")["domain"],
    }
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------
# Manual review packet
# --------------------------------------------------------------------------

REVIEW_LOW_VIETNAMESE = 0.05
REVIEW_VERY_LOW_VIETNAMESE = 0.001
REVIEW_LOW_ALPHA = 0.5
REVIEW_HUGE_WITH_FEW_LINES_CHARS = 100_000
REVIEW_HUGE_WITH_FEW_LINES_COUNT = 100


def review_flags(doc: dict) -> tuple[list[str], str | None]:
    """Triggered signals for one audited document plus a proposed decision.

    Proposed decision: "reject" only when an enabled rule fires (the same
    policy the pipeline used, loaded from the run manifest); otherwise
    "review". No automatic language classification happens here.
    """
    signals = doc["signals"]
    text = doc["text"]
    checks = rule_checks(text, signals, DEFAULT_REJECT_POLICY, DEFAULT_THRESHOLDS)
    flags: list[str] = []
    if signals["vietnamese_diacritic_ratio"] < REVIEW_VERY_LOW_VIETNAMESE:
        flags.append("very_low_vietnamese")
    if signals["vietnamese_diacritic_ratio"] < REVIEW_LOW_VIETNAMESE:
        flags.append("low_vietnamese")
    if signals["alpha_ratio"] < REVIEW_LOW_ALPHA:
        flags.append("low_alpha")
    if signals["replacement_char_count"] >= 1:
        flags.append("replacement_chars")
    if signals["unusual_unicode_count"] >= 1:
        flags.append("unusual_unicode")
    if signals["vni_pattern_count"] >= 3:
        flags.append("vni_pattern")
    if checks[REASON_ENCODING]:
        flags.append(REASON_ENCODING)
    if checks[REASON_BINARY]:
        flags.append(REASON_BINARY)
    if checks[REASON_FOREIGN]:
        flags.append(REASON_FOREIGN)
    if checks[REASON_CONCATENATED]:
        flags.append(REASON_CONCATENATED)
    if signals["character_count"] > REVIEW_HUGE_WITH_FEW_LINES_CHARS and \
            signals["line_count"] < REVIEW_HUGE_WITH_FEW_LINES_COUNT:
        flags.append("extreme_length_few_lines")
    proposed = "reject" if any(checks[r] and DEFAULT_REJECT_POLICY[r]
                               for r in (REASON_ENCODING, REASON_BINARY,
                                         REASON_FOREIGN, REASON_CONCATENATED)) else "review"
    return flags, proposed


def build_review_packet(rows: list[dict], out_dir: Path) -> dict:
    """Export every document triggering any review condition, deduplicated.

    Writes quality_review.csv, quality_review.html, and the fill-in
    decisions template. One row per document even when several signals fire.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for doc in rows:
        flags, proposed = review_flags(doc)
        if not flags:
            continue
        signals = doc["signals"]
        entries.append({
            "id": doc["id"],
            "domain": doc["domain"],
            "split": doc["split"],
            "triggered_signals": "|".join(sorted(set(flags))),
            "character_count": signals["character_count"],
            "line_count": signals["line_count"],
            "max_line_length": signals["max_line_length"],
            "alpha_ratio": signals["alpha_ratio"],
            "vietnamese_diacritic_ratio": signals["vietnamese_diacritic_ratio"],
            "digit_ratio": signals["digit_ratio"],
            "punctuation_ratio": signals["punctuation_ratio"],
            "replacement_char_count": signals["replacement_char_count"],
            "unusual_unicode_count": signals["unusual_unicode_count"],
            "html_tag_count": signals["html_tag_count"],
            "url_count": signals["url_count"],
            "non_latin_script_share": signals["non_latin_script_share"],
            "vni_pattern_count": signals["vni_pattern_count"],
            "preview": _preview(doc["text"], limit=200),
            "proposed_decision": proposed,
            "human_decision": "",
            "reviewer_notes": "",
        })
    entries.sort(key=lambda e: (e["triggered_signals"], e["id"]))

    header = list(entries[0].keys()) if entries else ["id"]
    write_csv(out_dir / "quality_review.csv", header,
              [list(e.values()) for e in entries])
    write_csv(out_dir / "quality_review_decisions.template.csv",
              ["id", "domain", "split", "triggered_signals", "preview",
               "proposed_decision", "human_decision", "reviewer_notes"],
              [[e["id"], e["domain"], e["split"], e["triggered_signals"],
                e["preview"], e["proposed_decision"], "", ""] for e in entries])
    _write_review_html(out_dir / "quality_review.html", entries, header)

    from collections import Counter

    return {
        "review_documents": len(entries),
        "review_by_signal": dict(Counter(
            signal for e in entries for signal in e["triggered_signals"].split("|")
        )),
        "proposed_reject": sum(1 for e in entries if e["proposed_decision"] == "reject"),
        "proposed_review": sum(1 for e in entries if e["proposed_decision"] == "review"),
        "files": ["quality_review.csv", "quality_review.html",
                  "quality_review_decisions.template.csv"],
    }


def _write_review_html(path: Path, entries: list[dict], header: list[str]) -> None:
    def cell(value) -> str:
        text = str(value)
        return (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    parts = [
        "<!DOCTYPE html><html lang=\"vi\"><head><meta charset=\"utf-8\">",
        "<title>Quality review packet</title>",
        "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse;"
        "font-size:12px}th,td{border:1px solid #ccc;padding:4px 6px;text-align:left;"
        "vertical-align:top}th{background:#f0f0f0}tr:nth-child(even){background:#fafafa}"
        ".reject{background:#ffe9e9}</style></head><body>",
        f"<h1>Quality review packet</h1><p>{len(entries)} documents flagged. "
        "Human decisions go into quality_review_decisions.template.csv.</p>",
        "<table><tr>" + "".join(f"<th>{cell(h)}</th>" for h in header) + "</tr>",
    ]
    for entry in entries:
        cls = " class=\"reject\"" if entry["proposed_decision"] == "reject" else ""
        parts.append(
            "<tr" + cls + ">" +
            "".join(f"<td>{cell(entry[h])}</td>" for h in header) +
            "</tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_audit(settings: Settings) -> dict:
    output_dir = Path(settings.output_dir)
    audit_dir = Path(settings.audit_dir)
    processed_dir = output_dir / "processed"
    stats_dir = output_dir / "statistics"
    audit_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows = load_rows(processed_dir)
    print(f"auditing {len(rows)} accepted documents")

    for row in tqdm(rows, desc="signals", unit="docs"):
        row["signals"] = compute_quality_signals(row["text"])

    signal_names = list(next(iter(rows))["signals"].keys()) if rows else []
    header = ["split", "id", "domain", "text_hash"] + signal_names
    write_csv(
        audit_dir / "quality_signals.csv",
        header,
        [[row["split"], row["id"], row["domain"], row["text_hash"]]
         + [row["signals"][name] for name in signal_names]
         for row in rows],
    )

    samples = select_audit_samples(rows, settings.seed)
    write_csv(audit_dir / "audit_samples.csv", list(samples[0].keys()) if samples else ["id"],
              [list(s.values()) for s in samples])

    outliers = longest_documents_preview(rows)
    _write_outlier_preview(audit_dir / "longest_preview.md", outliers)

    drift = domain_drift(processed_dir, stats_dir)
    drift_rows = drift["rows"]
    write_csv(
        audit_dir / "domain_drift.csv",
        list(drift_rows[0].keys()) if drift_rows else ["domain"],
        [list(r.values()) for r in drift_rows],
    )
    write_json(audit_dir / "domain_drift_summary.json", drift["summary"])

    if settings.chart:
        from .charts import render_domain_chart, render_length_histogram

        chart_dir = Path(settings.chart).parent
        chart_dir.mkdir(parents=True, exist_ok=True)
        render_domain_chart(drift_rows, Path(settings.chart))
        render_length_histogram(
            [r["signals"]["character_count"] for r in rows],
            chart_dir / "length_histogram.svg",
        )

    near_summary = None
    if settings.near_duplicates:
        from .near_duplicates import find_near_duplicates

        near_summary = find_near_duplicates(
            [(r["id"], r["split"], r["text"]) for r in rows],
            seed=settings.seed,
            threshold=settings.simhash_threshold,
            shingle_size=settings.shingle_size,
            out_dir=audit_dir,
        )
        print(f"near-duplicate audit: {near_summary['pairs']} pairs, "
              f"{near_summary['cross_split_pairs']} cross-split")

    review_out = Path(settings.review_dir)
    review_summary = build_review_packet(rows, review_out)
    print(f"review packet: {review_summary['review_documents']} documents "
          f"flagged under {review_out}")

    summary = {
        "documents_audited": len(rows),
        "sample_counts": dict(Counter(s["reasons"] for s in samples)),
        "longest_document_chars": max((r["signals"]["character_count"] for r in rows), default=None),
        "median_character_count": median(
            r["signals"]["character_count"] for r in rows
        ) if rows else None,
        "near_duplicates": near_summary,
        "review": review_summary,
        "elapsed_seconds": round(time.time() - started, 1),
        "files_written": [p.name for p in audit_dir.iterdir()],
    }
    write_json(audit_dir / "audit_summary.json", summary)
    print(f"audit finished in {summary['elapsed_seconds']}s; "
          f"reports written under {audit_dir}")
    return summary


def _write_outlier_preview(path: Path, outliers: list[dict]) -> None:
    lines = [
        "# Longest-document preview",
        "",
        "Top documents by character count, for manual outlier classification.",
        "Preview text is truncated for display only; processed data is unchanged.",
        "",
    ]
    for row in outliers:
        lines.extend([
            f"## rank {row['rank']} — id {row['id']} ({row['domain']}, split={row['split']})",
            f"- characters: {row['character_count']}, lines: {row['line_count']}, "
            f"paragraphs: {row['paragraph_count']}",
            f"- url_count: {row['url_count']}, url_char_ratio: {row['url_char_ratio']}, "
            f"html_tags: {row['html_tag_count']}",
            f"- repeated_line_ratio: {row['repeated_line_ratio']}, "
            f"repeated_paragraph_ratio: {row['repeated_paragraph_ratio']}",
            "",
            "```",
            row["preview"],
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_audit_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.data.audit",
        description="Non-destructive quality audit of preprocessing outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/processed",
                        help="directory produced by the preprocessing pipeline")
    parser.add_argument("--audit-dir", default="data/audit",
                        help="directory for audit reports")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for deterministic random sample selection")
    parser.add_argument("--near-duplicates", action="store_true",
                        help="enable the optional SimHash near-duplicate audit")
    parser.add_argument("--simhash-threshold", type=int, default=3,
                        help="max Hamming distance to call two documents near-duplicates")
    parser.add_argument("--shingle-size", type=int, default=4,
                        help="character shingle size for SimHash")
    parser.add_argument("--chart", default="docs/assets/domain_distribution.svg",
                        help="path for the domain-distribution chart (SVG)")
    parser.add_argument("--review-dir", default="data/review",
                        help="directory for the manual review packet")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_audit_parser()
    args = parser.parse_args(argv)
    settings = Settings(
        output_dir=args.output_dir,
        seed=args.seed,
        audit_dir=args.audit_dir,
        near_duplicates=args.near_duplicates,
        simhash_threshold=args.simhash_threshold,
        shingle_size=args.shingle_size,
        chart=args.chart,
        review_dir=args.review_dir,
    )
    try:
        run_audit(settings)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
