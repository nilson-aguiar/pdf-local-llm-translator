"""Tests for :func:`pdf_translator.translate.batcher.batch_segments`.

Coverage areas
--------------
(a) Atomicity + completeness: every segment id appears exactly once, overall
    order is preserved across batches.
(b) Budget respected: no batch exceeds ``token_budget`` estimated tokens unless
    it is a single oversized segment.
(c) Oversized single segment lands alone in its own batch.
(d) ``max_segments`` cap forces a split even when under the token budget.
(e) Empty input returns ``[]``.
(f) Batch ids are contiguous from 0 to n-1.
"""

from __future__ import annotations

import pytest

from pdf_translator.models import Segment
from pdf_translator.translate.batcher import batch_segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_segment(seg_id: int, text: str) -> Segment:
    """Construct a :class:`Segment` with minimal required fields."""
    return Segment(id=seg_id, text=text, block_index=seg_id, field_index=0)


def _approx_tokens(text: str) -> int:
    """Mirror of ``estimate_tokens`` to keep tests independent of the import."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mixed_segments() -> list[Segment]:
    """~12 segments of varied length for broad coverage."""
    texts = [
        "Hi",                              # 0 → ≈1 tok
        "Short sentence here.",            # 1 → ≈5 tok
        "A" * 40,                          # 2 → ≈10 tok
        "Medium length text that fills more space.",  # 3 → ≈10 tok
        "x",                              # 4 → ≈1 tok
        "Another short one.",             # 5 → ≈4 tok
        "B" * 80,                          # 6 → ≈20 tok
        "One more short segment.",        # 7 → ≈6 tok
        "C" * 20,                          # 8 → ≈5 tok
        "Slightly longer segment with more words in it.",  # 9 → ≈12 tok
        "End segment A.",                 # 10 → ≈4 tok
        "End segment B.",                 # 11 → ≈4 tok
    ]
    return [make_segment(i, t) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# (a) Atomicity + completeness
# ---------------------------------------------------------------------------


def test_every_segment_appears_exactly_once(mixed_segments: list[Segment]) -> None:
    """All segment ids are present exactly once across all batches."""
    batches = batch_segments(mixed_segments, token_budget=30)

    all_ids: list[int] = [s.id for b in batches for s in b.segments]
    expected_ids = [s.id for s in mixed_segments]

    assert sorted(all_ids) == sorted(expected_ids), "Some ids are missing or duplicated"
    assert len(all_ids) == len(expected_ids), "Segment count mismatch"


def test_overall_order_preserved(mixed_segments: list[Segment]) -> None:
    """Concatenating batches in order reproduces the input order."""
    batches = batch_segments(mixed_segments, token_budget=30)

    reconstructed = [s.id for b in batches for s in b.segments]
    expected = [s.id for s in mixed_segments]

    assert reconstructed == expected


# ---------------------------------------------------------------------------
# (b) Budget respected
# ---------------------------------------------------------------------------


def test_budget_respected(mixed_segments: list[Segment]) -> None:
    """No batch exceeds token_budget unless it is a lone oversized segment."""
    budget = 25
    batches = batch_segments(mixed_segments, token_budget=budget)

    for batch in batches:
        if len(batch.segments) == 1:
            # A single segment is always allowed regardless of size.
            continue
        assert batch.approx_tokens <= budget, (
            f"Batch {batch.id} has {batch.approx_tokens} tokens > budget {budget} "
            f"with {len(batch.segments)} segments"
        )


def test_budget_enforced_precisely() -> None:
    """A segment that exactly fits should share a batch; one over should not."""
    # Each segment is exactly 4 chars → 1 token each
    budget = 3  # room for 3 segments
    segs = [make_segment(i, "abcd") for i in range(6)]  # 1 tok each
    batches = batch_segments(segs, token_budget=budget)

    for batch in batches:
        if len(batch.segments) > 1:
            assert batch.approx_tokens <= budget


# ---------------------------------------------------------------------------
# (c) Oversized single segment lands alone
# ---------------------------------------------------------------------------


def test_oversized_segment_alone() -> None:
    """A segment larger than the budget gets its own batch."""
    budget = 5  # very small
    big_text = "X" * 200   # ≈50 tokens — well over budget
    segments = [
        make_segment(0, "tiny"),
        make_segment(1, big_text),
        make_segment(2, "also tiny"),
    ]
    batches = batch_segments(segments, token_budget=budget)

    # Find which batch holds segment 1
    for batch in batches:
        ids_in_batch = [s.id for s in batch.segments]
        if 1 in ids_in_batch:
            assert ids_in_batch == [1], (
                f"Oversized segment 1 should be alone, but batch contains {ids_in_batch}"
            )


def test_oversized_segment_does_not_drop_others() -> None:
    """Neighbors of an oversized segment still appear in their own batches."""
    budget = 5
    big_text = "Y" * 200
    segments = [
        make_segment(0, "aaa"),
        make_segment(1, big_text),
        make_segment(2, "bbb"),
    ]
    batches = batch_segments(segments, token_budget=budget)
    all_ids = {s.id for b in batches for s in b.segments}
    assert all_ids == {0, 1, 2}


# ---------------------------------------------------------------------------
# (d) max_segments cap
# ---------------------------------------------------------------------------


def test_max_segments_cap_forces_split() -> None:
    """Even under token budget, max_segments=2 must split a 5-segment input."""
    # 1-char segments → 1 token each, budget=100 would fit all 5 without splitting
    segments = [make_segment(i, "a") for i in range(5)]
    batches = batch_segments(segments, token_budget=100, max_segments=2)

    for batch in batches:
        assert len(batch.segments) <= 2, (
            f"Batch {batch.id} has {len(batch.segments)} segments, max is 2"
        )

    # Completeness still holds
    all_ids = [s.id for b in batches for s in b.segments]
    assert sorted(all_ids) == list(range(5))


def test_max_segments_default_is_40() -> None:
    """Default max_segments=40 keeps 40 tiny segments in one batch."""
    segments = [make_segment(i, "a") for i in range(40)]
    batches = batch_segments(segments, token_budget=10_000)
    assert len(batches) == 1
    assert len(batches[0].segments) == 40


def test_max_segments_41_splits_into_two() -> None:
    """41 tiny segments under budget but over default max_segments=40 → 2 batches."""
    segments = [make_segment(i, "a") for i in range(41)]
    batches = batch_segments(segments, token_budget=10_000)
    assert len(batches) == 2
    assert len(batches[0].segments) == 40
    assert len(batches[1].segments) == 1


# ---------------------------------------------------------------------------
# (e) Empty input
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list() -> None:
    assert batch_segments([], token_budget=100) == []


# ---------------------------------------------------------------------------
# (f) Batch ids are 0..n-1 contiguous
# ---------------------------------------------------------------------------


def test_batch_ids_contiguous(mixed_segments: list[Segment]) -> None:
    """Batch ids must be sequential integers starting at 0."""
    batches = batch_segments(mixed_segments, token_budget=20)
    ids = [b.id for b in batches]
    assert ids == list(range(len(batches))), f"Non-contiguous batch ids: {ids}"


def test_single_segment_batch_id_is_zero() -> None:
    segs = [make_segment(0, "hello")]
    batches = batch_segments(segs, token_budget=100)
    assert len(batches) == 1
    assert batches[0].id == 0
