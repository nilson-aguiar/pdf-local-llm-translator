"""Stage 2 — group raw spans into logical blocks with reading order.

Font-size clustering → heading levels; bullet/number glyphs → lists; monospace →
code; everything else → paragraphs. Tables (from pdfplumber) and figures (from
the detector) are interleaved into the same per-page reading order so the final
block list is a faithful linear reading of the document.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from ..logging import get_logger
from ..models import (
    Block,
    CaptionBlock,
    CodeBlock,
    HeadingBlock,
    ListBlock,
    PageLayout,
    ParagraphBlock,
    RawTextBlock,
    TableBlock,
)

logger = get_logger(__name__)

_BULLET_RE = re.compile(r"^\s*[•‣◦⁃∙•·▪◦‣\-\*–—]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*(\d{1,3}|[a-zA-Z])[.)]\s+(.*)$")
_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|figura|tabela|abbildung|tabelle|schéma|grafik)\s*\d+",
    re.IGNORECASE,
)
_DEFAULT_BODY_SIZE = 11.0


def build_blocks(
    pages: list[PageLayout],
    tables_by_page: Optional[dict[int, list[TableBlock]]] = None,
    figures_by_page: Optional[dict[int, list[Block]]] = None,
) -> list[Block]:
    """Turn detected pages into an ordered list of logical blocks."""
    tables_by_page = tables_by_page or {}
    figures_by_page = figures_by_page or {}

    body_size = _body_size(pages)
    heading_sizes = _heading_size_ranking(pages, body_size)

    all_blocks: list[Block] = []
    order = 0
    for page in pages:
        table_blocks = tables_by_page.get(page.number, [])
        table_bboxes = [tb.bbox for tb in table_blocks if tb.bbox]
        figure_blocks = figures_by_page.get(page.number, [])

        text_blocks: list[Block] = []
        for raw in page.text_blocks:
            if _inside_any(raw, table_bboxes):
                continue  # already represented as a table
            block = _classify(raw, body_size, heading_sizes, page.number)
            if block is not None:
                text_blocks.append(block)

        items: list[Block] = [*text_blocks, *table_blocks, *figure_blocks]
        for block in _reading_order(items, page.width):
            block.page = page.number
            block.order = order
            order += 1
            all_blocks.append(block)

    logger.debug("built %d logical blocks from %d pages", len(all_blocks), len(pages))
    return all_blocks


# --- font statistics -------------------------------------------------------


def _body_size(pages: list[PageLayout]) -> float:
    weights: Counter[int] = Counter()
    for page in pages:
        for tb in page.text_blocks:
            for line in tb.lines:
                for span in line.spans:
                    weights[round(span.size)] += max(1, len(span.text.strip()))
    if not weights:
        return _DEFAULT_BODY_SIZE
    return float(weights.most_common(1)[0][0])


def _heading_size_ranking(pages: list[PageLayout], body_size: float) -> dict[int, int]:
    """Map a rounded heading size → heading level (1 = largest), capped at 6."""
    sizes: set[int] = set()
    threshold = body_size * 1.15
    for page in pages:
        for tb in page.text_blocks:
            max_size = _block_max_size(tb)
            if max_size >= threshold and _is_short(tb):
                sizes.add(round(max_size))
    ranking = {}
    for level, size in enumerate(sorted(sizes, reverse=True), start=1):
        ranking[size] = min(level, 6)
    return ranking


def _block_max_size(raw: RawTextBlock) -> float:
    return max((s.size for line in raw.lines for s in line.spans), default=0.0)


def _is_short(raw: RawTextBlock) -> bool:
    words = len(raw.text.split())
    return len(raw.lines) <= 2 and 0 < words <= 14


# --- classification --------------------------------------------------------


def _classify(
    raw: RawTextBlock, body_size: float, heading_sizes: dict[int, int], page_number: int
) -> Optional[Block]:
    text = raw.text.strip()
    if not text:
        return None

    if _is_code(raw):
        return CodeBlock(content=raw.text.rstrip("\n"), language=None, bbox=raw.bbox, page=page_number)

    list_block = _as_list(raw, page_number)
    if list_block is not None:
        return list_block

    if _CAPTION_RE.match(text) and _is_short(raw):
        return CaptionBlock(content=_join_lines(raw), bbox=raw.bbox, page=page_number)

    max_size = _block_max_size(raw)
    if max_size >= body_size * 1.15 and _is_short(raw):
        level = heading_sizes.get(round(max_size), 2)
        return HeadingBlock(level=level, content=_join_lines(raw), bbox=raw.bbox, page=page_number)

    return ParagraphBlock(content=_join_lines(raw), bbox=raw.bbox, page=page_number)


def _is_code(raw: RawTextBlock) -> bool:
    mono = 0
    total = 0
    for line in raw.lines:
        for span in line.spans:
            n = max(1, len(span.text.strip()))
            total += n
            if span.is_monospace:
                mono += n
    return total > 0 and (mono / total) >= 0.8


def _as_list(raw: RawTextBlock, page_number: int) -> Optional[ListBlock]:
    items: list[str] = []
    ordered_votes = 0
    bullet_votes = 0
    for line in raw.lines:
        line_text = line.text.strip()
        if not line_text:
            continue
        m_bullet = _BULLET_RE.match(line_text)
        m_ordered = _ORDERED_RE.match(line_text)
        if m_bullet:
            items.append(m_bullet.group(1).strip())
            bullet_votes += 1
        elif m_ordered:
            items.append(m_ordered.group(2).strip())
            ordered_votes += 1
        elif items:
            # continuation of the previous item (wrapped line)
            items[-1] = f"{items[-1]} {line_text}".strip()
        else:
            return None  # first non-empty line isn't a marker -> not a list
    markers = ordered_votes + bullet_votes
    if markers < 1 or not items:
        return None
    return ListBlock(
        ordered=ordered_votes > bullet_votes,
        items=items,
        bbox=raw.bbox,
        page=page_number,
    )


def _join_lines(raw: RawTextBlock) -> str:
    """Join a block's lines into one string, repairing end-of-line hyphenation."""
    out = ""
    for line in raw.lines:
        t = line.text.strip()
        if not t:
            continue
        if not out:
            out = t
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha():
            out = out[:-1] + t
        else:
            out = f"{out} {t}"
    return out


# --- reading order ---------------------------------------------------------


def _inside_any(raw: RawTextBlock, table_bboxes) -> bool:
    for tbbox in table_bboxes:
        if raw.bbox.intersection_area(tbbox) >= 0.6 * max(raw.bbox.area, 1.0):
            return True
    return False


def _reading_order(items: list[Block], page_width: float) -> list[Block]:
    placed = [b for b in items if b.bbox is not None]
    unplaced = [b for b in items if b.bbox is None]
    split_x = _column_split(placed, page_width)
    if split_x is None:
        placed.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
    else:
        placed.sort(
            key=lambda b: (
                0 if (b.bbox.x0 + b.bbox.x1) / 2 < split_x else 1,
                b.bbox.y0,
                b.bbox.x0,
            )
        )
    return [*placed, *unplaced]


def _column_split(items: list[Block], page_width: float) -> Optional[float]:
    """Return an x split for a clear 2-column layout, else None."""
    if len(items) < 4 or page_width <= 0:
        return None
    centers = sorted((b.bbox.x0 + b.bbox.x1) / 2 for b in items)
    best_gap = 0.0
    best_mid = None
    for a, b in zip(centers, centers[1:]):
        gap = b - a
        if gap > best_gap:
            best_gap = gap
            best_mid = (a + b) / 2
    if best_mid is not None and best_gap > 0.2 * page_width:
        left = sum(1 for c in centers if c < best_mid)
        right = len(centers) - left
        if left >= 2 and right >= 2:
            return best_mid
    return None
