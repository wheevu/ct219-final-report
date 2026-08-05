"""Conservative, individually configurable rejection rules.

These rules reject only clearly unusable text. Anything uncertain stays
audit-only: the audit signals and the review packet flag it, but no rule
removes it. Thresholds are visible in the configuration and default to
conservative values so valid text is never collateral damage.

Design constraints:

- Mixed Vietnamese-English content is never rejected solely for containing
  English (English is Latin script; no rule keys on Latin share).
- Long documents are never rejected solely for being long; only page-dump
  signatures (a single "line" of tens of thousands of characters) trigger
  `concatenated_dump`.
- No automatic VNI-to-Unicode conversion: VNI-pattern text is flagged or
  rejected, never rewritten.
- Language detection is deliberately absent: `foreign_script_dominant` keys
  on measurable script shares, not on a language model.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- script ranges

NON_LATIN_SCRIPTS: dict[str, str] = {
    "armenian": r"[\u0530-\u058F]",
    "arabic": r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]",
    "greek": r"[\u0370-\u03FF\u1F00-\u1FFF]",
    "cyrillic": r"[\u0400-\u04FF\u0500-\u052F]",
    "hebrew": r"[\u0590-\u05FF]",
    "thai": r"[\u0E00-\u0E7F]",
    "devanagari": r"[\u0900-\u097F]",
    "hangul": r"[\uAC00-\uD7AF]",
    "cjk": r"[\u3400-\u4DBF\u4E00-\u9FFF]",
    "hiragana": r"[\u3040-\u309F]",
    "katakana": r"[\u30A0-\u30FF]",
}

_SCRIPT_RE = {name: re.compile(pattern) for name, pattern in NON_LATIN_SCRIPTS.items()}
VNI_PATTERN_RE = re.compile(r"[A-Za-z]\?[a-zà-ỹ]")

# ---------------------------------------------------------------- rule checks

# Reason names (also recorded in removal counters and the manifest).
REASON_ENCODING = "encoding_corruption"
REASON_BINARY = "binary_or_invalid_content"
REASON_FOREIGN = "foreign_script_dominant"
REASON_CONCATENATED = "concatenated_dump"
RULE_REASONS = (REASON_ENCODING, REASON_BINARY, REASON_FOREIGN, REASON_CONCATENATED)


def non_latin_script_share(text: str) -> float:
    """Share of characters belonging to non-Latin scripts (max over scripts)."""
    n = len(text)
    if not n:
        return 0.0
    return max((len(re.findall(_SCRIPT_RE[name], text)) for name in _SCRIPT_RE), default=0) / n


def vni_pattern_count(text: str) -> int:
    """Distinct VNI-encoding signatures: letter + '?' + lowercase letter.

    Distinctness matters: a single corrupted token repeated many times
    (for example 'Bar?a' in a football article) is not a corrupted document,
    while several distinct signatures indicate pervasive VNI text.
    """
    return len(set(VNI_PATTERN_RE.findall(text)))


def encoding_corruption(text: str, signals: dict, *, max_replacement_chars: int) -> bool:
    """Unambiguous data loss: replacement characters (U+FFFD).

    VNI-style mojibake detection is deliberately NOT part of this rule:
    on real web text, letter+question-mark patterns also come from JS
    ternaries, URL query strings, and no-space question marks, so automatic
    VNI classification is unreliable. VNI patterns remain an audit-only
    signal for the review packet.
    """
    return signals["replacement_char_count"] >= max_replacement_chars


def binary_or_invalid_content(text: str, signals: dict, *, min_unusual_unicode_count: int,
                              min_unusual_unicode_ratio: float) -> bool:
    """Format/private-use character soup (icon fonts, bidi marks, ZWSP).

    A few stray format characters are ordinary web-scraping noise; only
    documents where they form a substantial share are clearly unusable.
    """
    count = signals["unusual_unicode_count"]
    if count < min_unusual_unicode_count:
        return False
    total = signals.get("character_count") or len(text)
    return count / total >= min_unusual_unicode_ratio


def foreign_script_dominant(text: str, signals: dict, *, min_non_latin_share: float) -> bool:
    """A non-Latin script (Armenian, Arabic, Greek, Cyrillic, CJK, ...) dominates."""
    return signals["non_latin_script_share"] > min_non_latin_share


def concatenated_dump(text: str, signals: dict, *, max_line_length: int) -> bool:
    """One enormous 'line' is a scraped page dump, not prose."""
    return signals["max_line_length"] > max_line_length


DEFAULT_THRESHOLDS = {
    "max_replacement_chars": 3,
    "min_unusual_unicode_count": 8,
    "min_unusual_unicode_ratio": 0.005,
    "min_non_latin_share": 0.3,
    "max_line_length": 20_000,
}

# Default rejection policy: conservative. Only unambiguous data loss rejects
# by default; script dominance and page dumps stay audit-only until the
# manual review justifies enabling them.
DEFAULT_REJECT_POLICY = {
    REASON_ENCODING: True,
    REASON_BINARY: True,
    REASON_FOREIGN: False,
    REASON_CONCATENATED: False,
}


def rule_signals(text: str) -> dict:
    """Cheap signal subset needed by the rules (no n-grams, no category pass).

    Computed only when at least one rule is enabled, so default-off runs pay
    nothing for rule evaluation.
    """
    unusual = sum(
        1 for ch in text
        if unicodedata.category(ch) in ("Cf", "Cs", "Co", "Cc")
        and ch not in "\n\t\r"
    )
    return {
        "replacement_char_count": text.count("\ufffd"),
        "unusual_unicode_count": unusual,
        "max_line_length": max((len(line) for line in text.split("\n")), default=0),
        "non_latin_script_share": non_latin_script_share(text),
        "vni_pattern_count": vni_pattern_count(text),
    }


def rule_checks(
    text: str,
    signals: dict,
    policy: dict[str, bool] | None = None,
    thresholds: dict | None = None,
) -> dict[str, bool]:
    """Evaluate every rule; returns {reason: triggered} regardless of policy."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    policy = {**DEFAULT_REJECT_POLICY, **(policy or {})}
    checks = {
        REASON_ENCODING: encoding_corruption(
            text, signals,
            max_replacement_chars=thresholds["max_replacement_chars"],
        ),
        REASON_BINARY: binary_or_invalid_content(
            text, signals,
            min_unusual_unicode_count=thresholds["min_unusual_unicode_count"],
            min_unusual_unicode_ratio=thresholds["min_unusual_unicode_ratio"],
        ),
        REASON_FOREIGN: foreign_script_dominant(
            text, signals, min_non_latin_share=thresholds["min_non_latin_share"]
        ),
        REASON_CONCATENATED: concatenated_dump(
            text, signals, max_line_length=thresholds["max_line_length"]
        ),
    }
    return checks


def rejection_decision(checks: dict[str, bool], policy: dict[str, bool]) -> str | None:
    """First enabled-and-triggered reason, or None when the doc is accepted."""
    for reason in RULE_REASONS:
        if checks.get(reason) and policy.get(reason):
            return reason
    return None
