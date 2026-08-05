"""Tests for the conservative quality rules and the review packet.

The section-4 validation list: valid Vietnamese, Vietnamese with English
quotations, pure English, Armenian/Arabic/Greek/Cyrillic, replacement
characters, VNI-looking text, binary garbage, valid long-form Vietnamese,
and concatenated repeated page fragments.
"""

import unicodedata

import pytest

from data.audit import build_review_packet, compute_quality_signals, review_flags
from data.quality_rules import (
    DEFAULT_REJECT_POLICY,
    DEFAULT_THRESHOLDS,
    REASON_BINARY,
    REASON_CONCATENATED,
    REASON_ENCODING,
    REASON_FOREIGN,
    rejection_decision,
    rule_checks,
)

VALID_VI = (
    "Hà Nội, ngày 02/09/1945. Chủ tịch Hồ Chí Minh đọc bản Tuyên ngôn Độc lập "
    "tại Quảng trường Ba Đình. Đây là một đoạn văn tiếng Việt hoàn chỉnh "
    "với dấu câu, số và chữ hoa được giữ nguyên vẹn.\n"
    "Đoạn thứ hai tiếp tục mạch văn với nội dung khác nhau hoàn toàn."
)
VI_WITH_ENGLISH = (
    "Trang web này giới thiệu sản phẩm 'Made in Vietnam' và công nghệ "
    "machine learning. Người dùng có thể tham khảo thêm tại mục FAQ."
)
PURE_ENGLISH = (
    "The quick brown fox jumps over the lazy dog. Natural language processing "
    "is a field of artificial intelligence concerned with the interactions "
    "between computers and human language."
)
ARMENIAN = "Ազատություն ռ/կ կայքի գլխավոր էջ - Website Analysis & Details ազատություն"
ARABIC = "مرحباً بكم في موقعنا الإلكتروني الجديد. نقدم لكم أفضل الخدمات"
GREEK = "Ψαλτολόγιον Psaltologion Καὶ ἐγὼ εἶπα ἐν τῇ καρδίᾳ μου περὶ τῶν υἱῶν τοῦ ἀνθρώπου online hymnals"
CYRILLIC = "Приоритезация трафика (shaping) для разных провайдеров, тарифов"
REPLACEMENT = "Văn bản bị hỏng \ufffd\ufffd\ufffd ký tự thay thế trong dòng này"
VNI = "D? nh?n ? H? N?i: ?n nh? h? ??i, l?m nh? tr?u m?ng D? nh?n t?i H? N?i"
BARCA_REPEATED = "Cổ điển giữa Madrid-Bar?a cổ điển giữa Madrid-Bar?a " * 6
BINARY_GARBAGE = "yk;@y`~ \ufdd0\ufdd1\ufdd2\ufdd3\ufdd4\ufdd5\ufdd6\ufdd7\ufdd8\ufdd9 junk @g~ bvw~ e~ yk; y#yQ kQy` a#w~@w~ kOm`r"
LONG_VI = ("Đây là một bài văn dài hợp lệ. " * 2000)
CONCATENATED = "Trang bị ghép nối không xuống dòng. " * 4000 + "Kết thúc trang."


def _checks(text: str) -> dict[str, bool]:
    return rule_checks(text, compute_quality_signals(text),
                       DEFAULT_REJECT_POLICY, DEFAULT_THRESHOLDS)


def _rejects(text: str) -> str | None:
    return rejection_decision(_checks(text), DEFAULT_REJECT_POLICY)


def test_valid_vietnamese_not_flagged():
    checks = _checks(VALID_VI)
    assert not any(checks.values())
    assert _rejects(VALID_VI) is None


def test_vietnamese_with_english_quotations_not_rejected():
    assert _rejects(VI_WITH_ENGLISH) is None


def test_pure_english_not_rejected_by_rules():
    # Latin script, no corruption markers: no rule fires (audit-only signal).
    assert _rejects(PURE_ENGLISH) is None


@pytest.mark.parametrize("text", [ARMENIAN, ARABIC, GREEK, CYRILLIC])
def test_non_latin_script_dominant(text):
    checks = _checks(text)
    assert checks[REASON_FOREIGN] is True
    # audit-only by default: not rejected
    assert _rejects(text) is None


def test_replacement_characters_flag_encoding_corruption():
    assert _checks(REPLACEMENT)[REASON_ENCODING] is True
    assert _rejects(REPLACEMENT) == REASON_ENCODING


def test_vni_text_is_audit_only_not_rejected():
    # VNI detection is unreliable on real web text (JS ternaries, URLs,
    # no-space question marks), so it stays an audit signal, never a rule.
    assert _checks(VNI)[REASON_ENCODING] is False
    assert _rejects(VNI) is None
    flags, proposed = review_flags(_doc(VNI, "vni-1"))
    assert "vni_pattern" in flags
    assert proposed == "review"


def test_single_repeated_mojibake_token_not_flagged():
    # 'Bar?a' appears repeatedly but is one token, not a corrupted document.
    assert _checks(BARCA_REPEATED)[REASON_ENCODING] is False
    assert _rejects(BARCA_REPEATED) is None


def test_binary_garbage_flags_invalid_content():
    checks = _checks(BINARY_GARBAGE)
    assert checks[REASON_BINARY] is True
    assert _rejects(BINARY_GARBAGE) == REASON_BINARY


def test_valid_long_form_vietnamese_not_rejected():
    assert _rejects(LONG_VI) is None


def test_concatenated_dump_flags_but_audit_only_by_default():
    checks = _checks(CONCATENATED)
    assert checks[REASON_CONCATENATED] is True
    assert _rejects(CONCATENATED) is None


def test_concatenated_enabled_rejects():
    policy = {**DEFAULT_REJECT_POLICY, REASON_CONCATENATED: True}
    decision = rejection_decision(_checks(CONCATENATED), policy)
    assert decision == REASON_CONCATENATED


def test_foreign_script_enabled_rejects():
    policy = {**DEFAULT_REJECT_POLICY, REASON_FOREIGN: True}
    assert rejection_decision(_checks(ARMENIAN), policy) == REASON_FOREIGN


def test_mixed_vietnamese_russian_dominant_cyrillic_flags():
    text = "Một chút tiếng Việt " + CYRILLIC + " vẫn còn đây."
    checks = _checks(text)
    assert checks[REASON_FOREIGN] is True


def test_normalized_unicode_equivalence_keeps_decisions_stable():
    assert _rejects(unicodedata.normalize("NFD", VALID_VI)) is None
    assert _checks(unicodedata.normalize("NFD", VNI))[REASON_ENCODING] is False
    assert _checks(unicodedata.normalize("NFD", REPLACEMENT))[REASON_ENCODING] is True


def _doc(text: str, doc_id: str = "d-1", domain: str = "news", split: str = "train"):
    return {
        "split": split,
        "id": doc_id,
        "domain": domain,
        "text_hash": "h",
        "text": text,
        "signals": compute_quality_signals(text),
    }


def test_review_packet_deduplicates_and_sets_decisions(tmp_path):
    rows = [
        _doc(VALID_VI, "ok-1"),
        _doc(PURE_ENGLISH, "en-1"),
        _doc(ARMENIAN, "am-1"),
        _doc(REPLACEMENT, "rep-1"),
        _doc(VNI, "vni-1"),
        _doc(BINARY_GARBAGE, "bin-1"),
        _doc(CONCATENATED, "cat-1", split="test"),
        _doc(VALID_VI, "ok-2", domain="wiki"),
    ]
    summary = build_review_packet(rows, tmp_path)
    # pure English is flagged (low vietnamese) but proposed "review"; valid VI not listed
    assert summary["review_documents"] == 6
    csv_lines = (tmp_path / "quality_review.csv").read_text("utf-8").splitlines()
    assert len(csv_lines) == 7  # header + 6
    ids = [line.split(",")[0] for line in csv_lines[1:]]
    assert len(ids) == len(set(ids))  # one row per document
    assert "ok-1" not in ids and "ok-2" not in ids

    decisions = (tmp_path / "quality_review_decisions.template.csv").read_text(
        "utf-8").splitlines()
    header = decisions[0]
    assert "human_decision" in header and "reviewer_notes" in header
    rep_row = next(l for l in decisions[1:] if l.startswith("rep-1"))
    assert "reject" in rep_row
    assert "review" in next(l for l in decisions[1:] if l.startswith("en-1"))

    html = (tmp_path / "quality_review.html").read_text("utf-8")
    assert "<table>" in html and "en-1" in html


def test_review_flags_for_english_is_review_not_reject():
    flags, proposed = review_flags(_doc(PURE_ENGLISH))
    assert "very_low_vietnamese" in flags
    assert proposed == "review"
