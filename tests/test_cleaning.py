"""Tests for the conservative Vietnamese cleaning pipeline."""

import unicodedata

from data.cleaning import (
    REASON_EMPTY,
    REASON_INVALID,
    REASON_LONG,
    REASON_SHORT,
    clean_text,
    has_meaningful_text,
    inspect_document,
    nfc_normalize,
    normalize_line_endings,
    remove_control_characters,
)

VI_TEXT = "Xin chào, tôi là sinh viên. Hôm nay trời nắng đẹp!"


def test_nfc_normalization_composes_decomposed_diacritics():
    decomposed = unicodedata.normalize("NFD", "Tôi đến Việt Nam")
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    normalized = nfc_normalize(decomposed)
    assert normalized == "Tôi đến Việt Nam"
    assert unicodedata.is_normalized("NFC", normalized)


def test_nfc_preserves_vietnamese_diacritics():
    original = "ă â đ ê ô ơ ư ế ệ ỹ ả à ạ"
    assert nfc_normalize(original) == original


def test_line_endings_normalized():
    assert normalize_line_endings("a\r\nb\rc") == "a\nb\nc"


def test_control_characters_removed_but_newline_and_tab_kept():
    text = "a\x00b\x1fc\n\tg"
    cleaned = remove_control_characters(text)
    assert cleaned == "abc\n\tg"


def test_repeated_spaces_collapsed_and_tabs_replaced():
    assert clean_text("Tôi   thích    học  NLP") == "Tôi thích học NLP"
    assert clean_text("a\tb") == "a b"


def test_excessive_blank_lines_reduced_to_one():
    text = "Đoạn một.\n\n\n\n\nĐoạn hai."
    assert clean_text(text) == "Đoạn một.\n\nĐoạn hai."


def test_leading_trailing_whitespace_trimmed():
    assert clean_text("  \n  Xin chào.  \n  ") == "Xin chào."
    assert clean_text("\n\n\n") == ""


def test_paragraph_boundaries_preserved():
    text = "Đoạn một.\nĐoạn hai.\nĐoạn ba."
    assert clean_text(text) == "Đoạn một.\nĐoạn hai.\nĐoạn ba."


def test_punctuation_numbers_capitalization_preserved():
    cleaned = clean_text("Điểm số: 9,5 (CAO NHẤT). Xin chào!")
    assert cleaned == "Điểm số: 9,5 (CAO NHẤT). Xin chào!"


def test_empty_text_rejected():
    assert inspect_document("", 20) == ("", REASON_EMPTY)
    assert inspect_document(None, 20) == ("", REASON_EMPTY)
    assert inspect_document("   \n\t  ", 20) == ("", REASON_EMPTY)


def test_invalid_text_without_letters_or_digits_rejected():
    cleaned, reason = inspect_document("!!!  ??? ###", 20)
    assert reason == REASON_INVALID
    assert not has_meaningful_text(cleaned)


def test_short_text_rejected_with_threshold():
    cleaned, reason = inspect_document("Xin chào", 20)
    assert reason == REASON_SHORT
    assert cleaned == "Xin chào"


def test_short_text_accepted_at_threshold():
    long_enough = "Đây là một câu dài đủ điều kiện chấp nhận."
    assert len(long_enough) >= 20
    _, reason = inspect_document(long_enough, 20)
    assert reason is None


def test_long_text_rejected_when_max_chars_set():
    text = "Đoạn " * 100
    _, reason = inspect_document(text, 20, max_chars=200)
    assert reason == REASON_LONG


def test_max_chars_optional_by_default():
    text = "Đoạn " * 100
    _, reason = inspect_document(text, 20)
    assert reason is None


def test_vietnamese_diacritics_survive_full_clean():
    raw = "Cộng hòa Xã hội chủ nghĩa Việt Nam  -  Hà Nội, ngày 02/09/1945."
    cleaned = clean_text(raw)
    assert "Cộng hòa" in cleaned
    assert "Việt Nam" in cleaned
    assert unicodedata.is_normalized("NFC", cleaned)
