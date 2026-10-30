"""Block ⇄ segment bridge for the translate stage.

Turns each block's translatable text fields into protected :class:`Segment`s
(markdown protection composed with glossary protection), and reassembles
translated text back into blocks by (block_index, field_index). Keeping this in
one place guarantees the extract→translate→reassemble round-trip preserves block
counts, list lengths, and table shapes.
"""

from __future__ import annotations

from typing import Optional

from ..extract import markdown
from ..models import SENTINEL_RE, Block, ProtectedText, Segment
from .glossary import Glossary


def is_translatable(text: str) -> bool:
    """True if ``text`` has natural-language content worth sending to the model.

    Empty, whitespace-only, or sentinel/number/punctuation-only text is passed
    through unchanged (no wasted request, and the validator won't flag it).
    """
    stripped = SENTINEL_RE.sub("", text).strip()
    return any(ch.isalpha() for ch in stripped)


def build_segments(blocks: list[Block], glossary: Optional[Glossary] = None) -> list[Segment]:
    """Create one protected segment per translatable text field, in order."""
    segments: list[Segment] = []
    seg_id = 0
    for block_index, block in enumerate(blocks):
        for field_index, text in enumerate(block.translatable_texts()):
            if is_translatable(text):
                protected = markdown.protect(text)
                if glossary is not None:
                    protected = glossary.protect(protected)
            else:
                protected = ProtectedText(text=text)
            segments.append(
                Segment(
                    id=seg_id,
                    text=protected.text,
                    restorations=protected.restorations,
                    block_index=block_index,
                    field_index=field_index,
                )
            )
            seg_id += 1
    return segments


def reassemble(
    blocks: list[Block], segments: list[Segment], final_by_id: dict[int, str]
) -> list[Block]:
    """Rebuild blocks with translated (restored) text, preserving shape."""
    by_block: dict[int, list[str]] = {}
    for seg in sorted(segments, key=lambda s: (s.block_index, s.field_index)):
        by_block.setdefault(seg.block_index, []).append(final_by_id[seg.id])
    out: list[Block] = []
    for block_index, block in enumerate(blocks):
        texts = by_block.get(block_index)
        out.append(block.with_translations(texts) if texts else block)
    return out
