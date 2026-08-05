"""Small dependency-free SVG charts for report assets."""

from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_domain_chart(drift_rows: list[dict], path: Path) -> None:
    """Grouped horizontal bar chart: inspected / accepted / retained % per domain."""
    rows = sorted(
        [r for r in drift_rows if r["retained_n"] > 0],
        key=lambda r: r["retained_n"],
        reverse=True,
    )[:12]

    series = [
        ("inspected_pct", "#8fa8bf"),
        ("accepted_pct", "#c9a26b"),
        ("retained_pct", "#5b8a72"),
    ]
    max_pct = max((r[key] for r in rows for key, _ in series), default=100.0)
    scale = (max_pct * 1.1) or 1.0
    bar_width = 560.0
    row_height = 52.0
    bar_h = 9.0
    gap = 5.0
    margin_left = 190.0
    margin_top = 70.0
    width = 950
    height = int(margin_top + len(rows) * row_height + 40)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="28" font-size="17" font-family="sans-serif" font-weight="bold">'
        f'Domain distribution: inspected vs accepted vs retained (%)</text>',
        f'<text x="20" y="48" font-size="12" font-family="sans-serif" fill="#555">'
        f'top 12 domains by retained count; percentages of each population</text>',
    ]
    legend_x = 660
    for i, (label, color) in enumerate(series):
        x = legend_x + i * 90
        parts.append(
            f'<rect x="{x}" y="20" width="16" height="16" fill="{color}"/>'
            f'<text x="{x + 20}" y="32" font-size="12" font-family="sans-serif">{label}</text>'
        )

    for i, row in enumerate(rows):
        y = margin_top + i * row_height
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + bar_h + 2}" '
            f'font-size="12" font-family="sans-serif" text-anchor="end">{_escape(row["domain"])}</text>'
        )
        for j, (key, color) in enumerate(series):
            value = row[key]
            bar_y = y + j * (bar_h + gap)
            bar_len = bar_width * (value / scale)
            parts.append(
                f'<rect x="{margin_left}" y="{bar_y}" width="{bar_len:.1f}" '
                f'height="{bar_h}" fill="{color}"/>'
            )
        drift = row["drift_retained_vs_inspected_pp"]
        parts.append(
            f'<text x="{margin_left + bar_width + 8}" y="{y + bar_h + 2}" '
            f'font-size="11" font-family="sans-serif" fill="#666">'
            f'drift {drift:+.2f}pp</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_length_histogram(lengths: list[int], path: Path) -> None:
    """Bar chart of document character counts in 1,000-char bins (top bin open)."""
    bins = [(i * 1000, (i + 1) * 1000) for i in range(10)]
    counts = [0] * len(bins)
    for length in lengths:
        index = min(length // 1000, len(bins) - 1)
        counts[index] += 1
    total = len(lengths)
    max_count = max(counts) or 1
    scale = 300.0 / max_count
    bar_w = 66.0
    margin_left = 70.0
    margin_top = 60.0
    width = margin_left + len(bins) * (bar_w + 12) + 30
    height = 420

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="28" font-size="17" font-family="sans-serif" font-weight="bold">'
        f'Document length histogram (characters, accepted documents, n={total})</text>',
    ]
    for i, (low, high) in enumerate(bins):
        x = margin_left + i * (bar_w + 12)
        bar_h = counts[i] * scale
        y = 340 - bar_h
        label = f"{low}-{high}" if i < len(bins) - 1 else f"{low}+"
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="#5b8a72"/>'
            f'<text x="{x + bar_w / 2}" y="360" font-size="11" font-family="sans-serif" '
            f'text-anchor="middle">{label}</text>'
            f'<text x="{x + bar_w / 2}" y="{y - 6:.1f}" font-size="11" '
            f'font-family="sans-serif" text-anchor="middle">{counts[i]}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
