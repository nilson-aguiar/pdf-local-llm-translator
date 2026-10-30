"""Anti-hallucination guard for translated segments.

Detects structural drift between a source :class:`~pdf_translator.models.Segment`
and its raw (still-protected) translation so the pipeline can retry or flag
problematic outputs before restoration.
"""

from __future__ import annotations

import re
from collections import Counter

from pdf_translator.models import (
    Segment,
    TranslationBatch,
    ValidationIssue,
    ValidationReport,
    find_sentinels,
)

# Matches paired fenced code blocks so inline-backtick counting can ignore them.
_FENCE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _issue(segment_id: int, kind: str, detail: str = "") -> ValidationIssue:
    """Return a :class:`ValidationIssue` for the given segment and kind."""
    return ValidationIssue(segment_id=segment_id, kind=kind, detail=detail)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_segment(source: Segment, translated: str) -> ValidationReport:
    """Validate one translated segment against its source for structural drift.

    Checks performed (each failure appends a :class:`ValidationIssue`):

    * ``empty-translation`` — ``translated`` is blank/whitespace when
      ``source.text`` had non-whitespace content.
    * ``missing-sentinel`` — a sentinel present in ``source.text`` is absent
      (or under-represented) in ``translated``.
    * ``extra-sentinel`` — a sentinel appears in ``translated`` that was not in
      ``source.text`` (or appears more times).
    * ``fence-imbalance`` — the count of triple-backtick fence markers (````)
      in ``translated`` is odd (unclosed fence).
    * ``inline-code-imbalance`` — after stripping fenced blocks, the remaining
      single-backtick count in ``translated`` is odd.
    * ``math-delimiter-imbalance`` — the ``$`` count in ``translated`` is odd.

    Parameters
    ----------
    source:
        The original (protected) :class:`Segment` whose ``text`` still contains
        sentinel tokens.
    translated:
        Raw model output (sentinels still present, restorations not yet applied).

    Returns
    -------
    ValidationReport
        Report with zero or more issues; ``report.ok`` is ``True`` iff clean.
    """
    report = ValidationReport()
    seg_id = source.id

    # 1. Empty translation ------------------------------------------------
    if source.text.strip() and not translated.strip():
        report.issues.append(_issue(
            seg_id,
            "empty-translation",
            "Translation is empty or whitespace-only but source had content.",
        ))

    # 2. Sentinel multiset comparison -------------------------------------
    source_counts = Counter(find_sentinels(source.text))
    translated_counts = Counter(find_sentinels(translated))

    for token, count in source_counts.items():
        tr_count = translated_counts.get(token, 0)
        if tr_count < count:
            report.issues.append(_issue(
                seg_id,
                "missing-sentinel",
                f"Sentinel {token!r} appears {count}x in source but {tr_count}x in translation.",
            ))

    for token, count in translated_counts.items():
        src_count = source_counts.get(token, 0)
        if count > src_count:
            report.issues.append(_issue(
                seg_id,
                "extra-sentinel",
                f"Sentinel {token!r} appears {count}x in translation but {src_count}x in source.",
            ))

    # 3. Code fence balance -----------------------------------------------
    fence_count = translated.count("```")
    if fence_count % 2 != 0:
        report.issues.append(_issue(
            seg_id,
            "fence-imbalance",
            f"Odd number of triple-backtick fence markers: {fence_count}.",
        ))

    # 4. Inline backtick parity (after stripping fenced blocks) -----------
    without_fences = _FENCE_BLOCK_RE.sub("", translated)
    backtick_count = without_fences.count("`")
    if backtick_count % 2 != 0:
        report.issues.append(_issue(
            seg_id,
            "inline-code-imbalance",
            f"Odd number of inline backticks after removing fenced blocks: {backtick_count}.",
        ))

    # 5. Math delimiter parity --------------------------------------------
    dollar_count = translated.count("$")
    if dollar_count % 2 != 0:
        report.issues.append(_issue(
            seg_id,
            "math-delimiter-imbalance",
            f"Odd number of $ math delimiters: {dollar_count}.",
        ))

    return report


def validate_batch(
    batch: TranslationBatch,
    translations: dict[int, str],
) -> ValidationReport:
    """Validate all segments in a batch against a translations map.

    Checks:

    * ``missing-translation`` — a segment id present in ``batch`` has no entry
      in ``translations``.
    * ``unexpected-translation`` — an id in ``translations`` is not present in
      ``batch``.
    * Per-segment structural checks via :func:`validate_segment` for every
      segment that does have a corresponding translation.

    Parameters
    ----------
    batch:
        The :class:`TranslationBatch` that was sent to the model.
    translations:
        Map of ``segment_id → raw translated text`` returned by the model.

    Returns
    -------
    ValidationReport
        Merged report across all checks and all segments.
    """
    report = ValidationReport()
    batch_ids = {seg.id for seg in batch.segments}
    translation_ids = set(translations.keys())

    # Missing translations
    for seg in batch.segments:
        if seg.id not in translation_ids:
            report.issues.append(_issue(
                seg.id,
                "missing-translation",
                f"Segment {seg.id} has no corresponding entry in translations.",
            ))

    # Unexpected translations
    for t_id in translation_ids:
        if t_id not in batch_ids:
            report.issues.append(_issue(
                t_id,
                "unexpected-translation",
                f"Translation id {t_id} does not match any segment in the batch.",
            ))

    # Per-segment structural validation
    for seg in batch.segments:
        if seg.id in translations:
            report.extend(validate_segment(seg, translations[seg.id]))

    return report


def check_list_length(
    source_items: list[str],
    translated_items: list[str],
) -> ValidationReport:
    """Check that a translated list has the same number of items as the source.

    This helper is exposed for use by upstream reassembly code but is NOT called
    internally by :func:`validate_segment` (list-shape integrity is guaranteed
    structurally by the pipeline before this guard runs).

    Parameters
    ----------
    source_items:
        Original list items.
    translated_items:
        Translated list items.

    Returns
    -------
    ValidationReport
        Report with a ``list-length-mismatch`` issue if lengths differ,
        otherwise ``report.ok is True``.
    """
    report = ValidationReport()
    if len(source_items) != len(translated_items):
        report.issues.append(ValidationIssue(
            segment_id=-1,
            kind="list-length-mismatch",
            detail=(
                f"Source has {len(source_items)} items but translation has "
                f"{len(translated_items)}."
            ),
        ))
    return report


def check_table_parity(
    source_rows: list[list[str]],
    translated_rows: list[list[str]],
) -> ValidationReport:
    """Check that a translated table has the same shape as the source table.

    Verifies both the overall row count and the column count for every row.
    If row counts differ the column-level check is skipped (the positions no
    longer correspond).

    This helper is exposed for use by upstream reassembly code but is NOT called
    internally by :func:`validate_segment` (table-shape integrity is guaranteed
    structurally by the pipeline before this guard runs).

    Parameters
    ----------
    source_rows:
        Original table rows; each row is a list of cell strings.
    translated_rows:
        Translated table rows.

    Returns
    -------
    ValidationReport
        Report with ``table-parity-mismatch`` issues for any shape discrepancy,
        otherwise ``report.ok is True``.
    """
    report = ValidationReport()

    if len(source_rows) != len(translated_rows):
        report.issues.append(ValidationIssue(
            segment_id=-1,
            kind="table-parity-mismatch",
            detail=(
                f"Source has {len(source_rows)} rows but translation has "
                f"{len(translated_rows)}."
            ),
        ))
        return report  # Column checks are meaningless when row counts differ

    for i, (src_row, tr_row) in enumerate(zip(source_rows, translated_rows)):
        if len(src_row) != len(tr_row):
            report.issues.append(ValidationIssue(
                segment_id=-1,
                kind="table-parity-mismatch",
                detail=(
                    f"Row {i}: source has {len(src_row)} columns but translation "
                    f"has {len(tr_row)}."
                ),
            ))

    return report
