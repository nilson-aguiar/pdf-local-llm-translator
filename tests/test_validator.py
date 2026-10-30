"""Tests for the translate.validator module."""

from __future__ import annotations

import pytest

from pdf_translator.models import Segment, TranslationBatch
from pdf_translator.translate.validator import (
    check_list_length,
    check_table_parity,
    validate_batch,
    validate_segment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(text: str, seg_id: int = 1, restorations: dict | None = None) -> Segment:
    return Segment(id=seg_id, text=text, restorations=restorations or {})


def _batch(*segments: Segment, batch_id: int = 1) -> TranslationBatch:
    return TranslationBatch(id=batch_id, segments=list(segments))


# ---------------------------------------------------------------------------
# validate_segment — happy path
# ---------------------------------------------------------------------------


class TestValidateSegmentClean:
    def test_plain_text_ok(self):
        report = validate_segment(_seg("Hello world"), "Hola mundo")
        assert report.ok is True

    def test_sentinel_preserved_ok(self):
        report = validate_segment(
            _seg("Formula ⟦MATH0⟧ end"),
            "Formel ⟦MATH0⟧ Ende",
        )
        assert report.ok is True

    def test_multiple_sentinels_all_present_ok(self):
        report = validate_segment(
            _seg("⟦MATH0⟧ and ⟦CODE1⟧"),
            "⟦MATH0⟧ und ⟦CODE1⟧",
        )
        assert report.ok is True

    def test_even_fence_count_ok(self):
        report = validate_segment(
            _seg("Code block"),
            "Code:\n```python\nx = 1\n```",
        )
        assert report.ok is True

    def test_balanced_inline_backticks_ok(self):
        report = validate_segment(_seg("Use `x` and `y`"), "Use `x` and `y`")
        assert report.ok is True

    def test_balanced_dollar_signs_ok(self):
        report = validate_segment(_seg("Equation $a + b$"), "Gleichung $a + b$")
        assert report.ok is True

    def test_blank_source_blank_translation_ok(self):
        """Blank source + blank translation must NOT raise empty-translation."""
        report = validate_segment(_seg("   "), "")
        assert report.ok is True


# ---------------------------------------------------------------------------
# validate_segment — empty-translation
# ---------------------------------------------------------------------------


class TestEmptyTranslation:
    def test_whitespace_only_translation_flagged(self):
        report = validate_segment(_seg("Some text here"), "   ")
        assert not report.ok
        assert any(i.kind == "empty-translation" for i in report.issues)

    def test_truly_empty_string_flagged(self):
        report = validate_segment(_seg("Text"), "")
        assert not report.ok
        assert any(i.kind == "empty-translation" for i in report.issues)


# ---------------------------------------------------------------------------
# validate_segment — sentinel checks
# ---------------------------------------------------------------------------


class TestMissingSentinel:
    def test_dropped_sentinel(self):
        """Sentinel in source but absent in translation → missing-sentinel."""
        report = validate_segment(
            _seg("The formula ⟦MATH0⟧ is important"),
            "The formula is important",
        )
        assert not report.ok
        issues = [i for i in report.issues if i.kind == "missing-sentinel"]
        assert issues, "Expected at least one missing-sentinel issue"
        assert "MATH0" in issues[0].detail

    def test_partially_dropped_sentinel(self):
        """Sentinel appears once in source but zero times in translation."""
        report = validate_segment(_seg("⟦CODE0⟧"), "nothing")
        assert any(i.kind == "missing-sentinel" for i in report.issues)


class TestExtraSentinel:
    def test_invented_sentinel(self):
        """Sentinel in translation not present in source → extra-sentinel."""
        report = validate_segment(
            _seg("No sentinels here"),
            "No ⟦CODE5⟧ sentinels here",
        )
        assert not report.ok
        assert any(i.kind == "extra-sentinel" for i in report.issues)

    def test_duplicated_sentinel_is_extra(self):
        """Sentinel appearing more times in translation than source → extra-sentinel."""
        report = validate_segment(
            _seg("⟦MATH0⟧ only once"),
            "⟦MATH0⟧ twice ⟦MATH0⟧",
        )
        assert not report.ok
        assert any(i.kind == "extra-sentinel" for i in report.issues)


# ---------------------------------------------------------------------------
# validate_segment — fence-imbalance
# ---------------------------------------------------------------------------


class TestFenceImbalance:
    def test_odd_fence_count_flagged(self):
        """Three ``` markers (one unclosed) → fence-imbalance."""
        report = validate_segment(
            _seg("Code example"),
            "Before\n```python\ncode\n```\n```\nunclosed",
        )
        assert not report.ok
        assert any(i.kind == "fence-imbalance" for i in report.issues)

    def test_single_fence_marker_flagged(self):
        report = validate_segment(_seg("x"), "Start ```\nnever closed")
        assert any(i.kind == "fence-imbalance" for i in report.issues)


# ---------------------------------------------------------------------------
# validate_segment — inline-code-imbalance
# ---------------------------------------------------------------------------


class TestInlineCodeImbalance:
    def test_odd_backticks_flagged(self):
        """3 inline backticks (after fences removed) → inline-code-imbalance."""
        # "Use `x and `y`" → 3 backticks, all inline
        report = validate_segment(_seg("x"), "Use `x and `y`")
        assert not report.ok
        assert any(i.kind == "inline-code-imbalance" for i in report.issues)

    def test_backticks_inside_fence_ignored(self):
        """Backticks inside a fenced block should not count toward inline parity."""
        # Two fences (even → no fence-imbalance), backtick inside fence ignored,
        # zero inline backticks outside → ok
        report = validate_segment(_seg("x"), "```\n`ignored`\n```")
        assert report.ok is True


# ---------------------------------------------------------------------------
# validate_segment — math-delimiter-imbalance
# ---------------------------------------------------------------------------


class TestMathDelimiterImbalance:
    def test_odd_dollar_count_flagged(self):
        report = validate_segment(_seg("Equation $a + b$"), "Equation $a + b")
        assert not report.ok
        assert any(i.kind == "math-delimiter-imbalance" for i in report.issues)

    def test_three_dollars_flagged(self):
        report = validate_segment(_seg("x"), "$a$ and $b")
        assert any(i.kind == "math-delimiter-imbalance" for i in report.issues)


# ---------------------------------------------------------------------------
# validate_batch
# ---------------------------------------------------------------------------


class TestValidateBatch:
    def test_clean_batch_ok(self):
        s1 = _seg("Hello", seg_id=1)
        s2 = _seg("World", seg_id=2)
        report = validate_batch(_batch(s1, s2), {1: "Hola", 2: "Mundo"})
        assert report.ok is True

    def test_missing_translation_flagged(self):
        s1 = _seg("Hello", seg_id=1)
        s2 = _seg("World", seg_id=2)
        report = validate_batch(_batch(s1, s2), {1: "Hola"})  # id=2 missing
        assert not report.ok
        missing = [i for i in report.issues if i.kind == "missing-translation"]
        assert missing
        assert any(i.segment_id == 2 for i in missing)

    def test_unexpected_translation_flagged(self):
        s1 = _seg("Hello", seg_id=1)
        report = validate_batch(_batch(s1), {1: "Hola", 99: "Ghost"})
        assert not report.ok
        unexpected = [i for i in report.issues if i.kind == "unexpected-translation"]
        assert unexpected
        assert any(i.segment_id == 99 for i in unexpected)

    def test_both_missing_and_unexpected_flagged(self):
        s1 = _seg("Hello", seg_id=1)
        s2 = _seg("World", seg_id=2)
        report = validate_batch(_batch(s1, s2), {1: "Hola", 99: "Ghost"})
        kinds = {i.kind for i in report.issues}
        assert "missing-translation" in kinds
        assert "unexpected-translation" in kinds

    def test_segment_structural_issues_merged(self):
        """Dropped sentinel in a segment must surface through validate_batch."""
        s1 = _seg("Formula ⟦MATH0⟧", seg_id=1)
        report = validate_batch(_batch(s1), {1: "Formula"})
        assert not report.ok
        assert any(i.kind == "missing-sentinel" for i in report.issues)

    def test_empty_batch_empty_translations_ok(self):
        report = validate_batch(_batch(), {})
        assert report.ok is True


# ---------------------------------------------------------------------------
# check_list_length
# ---------------------------------------------------------------------------


class TestCheckListLength:
    def test_matching_lengths_ok(self):
        report = check_list_length(["a", "b", "c"], ["x", "y", "z"])
        assert report.ok is True

    def test_translated_shorter_flagged(self):
        report = check_list_length(["a", "b", "c"], ["x", "y"])
        assert not report.ok
        assert any(i.kind == "list-length-mismatch" for i in report.issues)

    def test_translated_longer_flagged(self):
        report = check_list_length(["a"], ["x", "y"])
        assert not report.ok
        assert any(i.kind == "list-length-mismatch" for i in report.issues)

    def test_empty_lists_ok(self):
        assert check_list_length([], []).ok is True


# ---------------------------------------------------------------------------
# check_table_parity
# ---------------------------------------------------------------------------


class TestCheckTableParity:
    def test_matching_table_ok(self):
        src = [["a", "b"], ["c", "d"]]
        tr = [["x", "y"], ["z", "w"]]
        assert check_table_parity(src, tr).ok is True

    def test_row_count_mismatch_flagged(self):
        src = [["a", "b"], ["c", "d"]]
        tr = [["x", "y"]]
        report = check_table_parity(src, tr)
        assert not report.ok
        assert any(i.kind == "table-parity-mismatch" for i in report.issues)

    def test_column_count_mismatch_flagged(self):
        src = [["a", "b", "c"], ["d", "e", "f"]]
        tr = [["x", "y", "z"], ["p", "q"]]  # row 1 has 2 cols instead of 3
        report = check_table_parity(src, tr)
        assert not report.ok
        assert any(i.kind == "table-parity-mismatch" for i in report.issues)

    def test_empty_tables_ok(self):
        assert check_table_parity([], []).ok is True

    def test_single_row_single_col_ok(self):
        assert check_table_parity([["a"]], [["x"]]).ok is True
