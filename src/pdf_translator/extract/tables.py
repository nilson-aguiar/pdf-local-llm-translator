"""Stage 2 — table extraction via pdfplumber.

pdfplumber and fitz both use PDF points with a top-left origin, so table
bounding boxes are directly comparable to fitz coordinates — the block grouper
uses them to avoid emitting the same text twice (once as a paragraph, once as a
table cell).
"""

from __future__ import annotations

import pdfplumber

from ..exceptions import ExtractError
from ..logging import get_logger
from ..models import BBox, TableBlock

logger = get_logger(__name__)


def _clean(cell: object) -> str:
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def extract_tables(pdf_path: str) -> dict[int, list[TableBlock]]:
    """Return GFM-ready table blocks keyed by 1-based page number."""
    tables_by_page: dict[int, list[TableBlock]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                found = page.find_tables()
                blocks: list[TableBlock] = []
                for table in found:
                    rows = table.extract()
                    rows = [[_clean(c) for c in row] for row in rows if row]
                    rows = [row for row in rows if any(cell for cell in row)]
                    if not rows:
                        continue
                    width = max(len(row) for row in rows)
                    # pad ragged rows so column parity holds downstream
                    rows = [row + [""] * (width - len(row)) for row in rows]
                    bbox = BBox.from_tuple(
                        (
                            float(table.bbox[0]),
                            float(table.bbox[1]),
                            float(table.bbox[2]),
                            float(table.bbox[3]),
                        )
                    )
                    blocks.append(
                        TableBlock(rows=rows, has_header=True, page=page_number, bbox=bbox)
                    )
                if blocks:
                    tables_by_page[page_number] = blocks
    except Exception as exc:
        raise ExtractError(f"failed to extract tables from {pdf_path!r}: {exc}") from exc
    return tables_by_page
