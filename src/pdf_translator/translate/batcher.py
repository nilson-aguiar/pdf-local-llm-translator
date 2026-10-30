"""Greedy segment batcher for the translation stage.

Segments are atomic — they must never be split across batches. This module
packs segments greedily in input order into :class:`~pdf_translator.models.TranslationBatch`
objects, respecting a per-batch token budget and a maximum segment count. A
single segment that exceeds the token budget on its own is placed in a batch by
itself so that no segment is ever dropped.
"""

from __future__ import annotations

from pdf_translator.models import Segment, TranslationBatch


def batch_segments(
    segments: list[Segment],
    token_budget: int,
    max_segments: int = 40,
) -> list[TranslationBatch]:
    """Pack *segments* greedily into :class:`TranslationBatch` objects.

    Packing rules
    -------------
    * Segments are consumed in their given order and that order is preserved
      across batches — concatenating ``batch.segments`` for every batch in
      result order reproduces the original input exactly.
    * A segment is added to the current batch as long as doing so would **not**
      push the batch over *token_budget* estimated tokens **and** the batch has
      fewer than *max_segments* entries.
    * When a segment does not fit (either limit), the current batch is closed and
      a new one is started.
    * A segment whose :attr:`~pdf_translator.models.Segment.approx_tokens`
      already exceeds *token_budget* is placed alone in its own batch — it is
      atomic and must not be dropped.
    * Empty input produces an empty list.
    * Every input segment appears in **exactly one** output batch.

    Parameters
    ----------
    segments:
        Ordered list of segments to pack. Must not be modified by the caller
        while this function runs.
    token_budget:
        Maximum estimated tokens allowed per batch. A single oversized segment
        still forms its own batch.
    max_segments:
        Maximum number of segments per batch (default 40).

    Returns
    -------
    list[TranslationBatch]
        Batches assigned sequential ids starting from 0.
    """
    if not segments:
        return []

    batches: list[TranslationBatch] = []
    current: list[Segment] = []
    current_tokens: int = 0

    def _flush(batch_list: list[TranslationBatch], bucket: list[Segment]) -> None:
        if bucket:
            batch_list.append(
                TranslationBatch(id=len(batch_list), segments=list(bucket))
            )
            bucket.clear()

    for segment in segments:
        seg_tokens = segment.approx_tokens

        # Would adding this segment exceed either limit?
        would_exceed_budget = (current_tokens + seg_tokens) > token_budget
        would_exceed_count = len(current) >= max_segments

        if current and (would_exceed_budget or would_exceed_count):
            # Close the current batch before starting a new one.
            _flush(batches, current)
            current_tokens = 0

        current.append(segment)
        current_tokens += seg_tokens

        # If this single oversized segment fills a whole batch, flush immediately
        # so the next segment starts fresh.
        if current_tokens > token_budget and len(current) == 1:
            _flush(batches, current)
            current_tokens = 0

    _flush(batches, current)
    return batches
