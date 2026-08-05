"""Conservative Vietnamese text cleaning and document acceptance checks.

The cleaning is deliberately conservative: punctuation, capitalization, numbers,
paragraph structure, and Vietnamese diacritics are all preserved because they
are useful for language modelling.
"""

from __future__ import annotations

import unicodedata

# Rejection reasons recorded for every discarded document.
REASON_EMPTY = "empty_text"
REASON_SHORT = "too_short"
REASON_LONG = "too_long"
REASON_DUP_ID = "duplicate_id"
REASON_DUP_TEXT = "duplicate_text"
REASON_INVALID = "invalid_text"
REASON_ENCODING = "encoding_corruption"
REASON_BINARY = "binary_or_invalid_content"
REASON_FOREIGN = "foreign_script_dominant"
REASON_CONCATENATED = "concatenated_dump"
ALL_REASONS = (
    REASON_EMPTY,
    REASON_SHORT,
    REASON_LONG,
    REASON_DUP_ID,
    REASON_DUP_TEXT,
    REASON_INVALID,
    REASON_ENCODING,
    REASON_BINARY,
    REASON_FOREIGN,
    REASON_CONCATENATED,
)


def nfc_normalize(text: str) -> str:
    """Normalize Unicode to NFC; Vietnamese diacritics are preserved."""
    return unicodedata.normalize("NFC", text)


def normalize_line_endings(text: str) -> str:
    """Convert CRLF and CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_control_characters(text: str) -> str:
    """Drop invalid control characters, keeping LF and tab."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Cc" or ch in "\n\t"
    )


def _collapse_line(line: str) -> str:
    """Collapse repeated spaces and trailing whitespace inside one line."""
    parts = line.replace("\t", " ").split(" ")
    return " ".join(p for p in parts if p).rstrip()


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    """Keep at most one consecutive blank line; trim leading/trailing blanks."""
    collapsed: list[str] = []
    prev_blank = True  # drops leading blank lines
    for line in lines:
        if line == "":
            if not prev_blank:
                collapsed.append("")
            prev_blank = True
        else:
            collapsed.append(line)
            prev_blank = False
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return collapsed


def clean_text(text: str) -> str:
    """Apply the full conservative cleaning pipeline.

    Order: line endings -> NFC -> control characters -> per-line whitespace
    normalization -> blank-line reduction -> edge trimming.
    """
    text = normalize_line_endings(text)
    text = nfc_normalize(text)
    text = remove_control_characters(text)
    lines = [_collapse_line(line) for line in text.split("\n")]
    return "\n".join(_collapse_blank_lines(lines))


def has_meaningful_text(text: str) -> bool:
    """True when the text contains at least one letter or digit."""
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in text)


def inspect_document(
    raw_text: str | None,
    min_chars: int,
    max_chars: int | None = None,
) -> tuple[str, str | None]:
    """Clean a raw document and return (cleaned_text, rejection_reason).

    Returns the reason for any rejected document, or None when accepted.
    """
    if raw_text is None or not raw_text.strip():
        return "", REASON_EMPTY
    cleaned = clean_text(raw_text)
    if not cleaned:
        return "", REASON_EMPTY
    if not has_meaningful_text(cleaned):
        return cleaned, REASON_INVALID
    if len(cleaned) < min_chars:
        return cleaned, REASON_SHORT
    if max_chars is not None and len(cleaned) > max_chars:
        return cleaned, REASON_LONG
    return cleaned, None
