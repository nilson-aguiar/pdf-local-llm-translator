"""Stage 1 — ingest & visual detection.

Open the PDF with ``fitz`` (PyMuPDF), read native text spans and raster image
placements per page, classify regions, and flag hybrid (native + image) pages.
Also renders arbitrary page regions to PNG bytes (used to extract figures and,
in Phase B, to feed OCR/VLM backends).
"""

from __future__ import annotations

import pymupdf as fitz

from ..exceptions import IngestError
from ..logging import get_logger
from ..models import (
    BBox,
    ImageRegion,
    PageLayout,
    RawTextBlock,
    TextLine,
    TextSpan,
)

logger = get_logger(__name__)

# Ignore decorative/again tiny images when extracting figures.
_MIN_FIGURE_AREA = 2500.0  # pt^2
_MIN_FIGURE_SIDE = 24.0  # pt


def open_document(path: str) -> fitz.Document:
    """Open a PDF, raising :class:`IngestError` on failure."""
    try:
        return fitz.open(path)
    except Exception as exc:  # pragma: no cover - fitz raises many types
        raise IngestError(f"cannot open PDF {path!r}: {exc}") from exc


def _bbox(raw) -> BBox:
    return BBox.from_tuple((float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])))


def detect_page(page: "fitz.Page") -> PageLayout:
    """Classify a single page into native text blocks and image regions."""
    rect = page.rect
    data = page.get_text("dict")

    text_blocks: list[RawTextBlock] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:  # 0 = text, 1 = image
            continue
        lines: list[TextLine] = []
        for line in block.get("lines", []):
            spans: list[TextSpan] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                spans.append(
                    TextSpan(
                        text=text,
                        bbox=_bbox(span["bbox"]),
                        font=span.get("font", ""),
                        size=float(span.get("size", 0.0)),
                        flags=int(span.get("flags", 0)),
                        color=int(span.get("color", 0)),
                    )
                )
            if spans:
                lines.append(TextLine(bbox=_bbox(line["bbox"]), spans=spans))
        if lines:
            text_blocks.append(RawTextBlock(bbox=_bbox(block["bbox"]), lines=lines))

    image_regions: list[ImageRegion] = []
    for img in page.get_images(full=True):
        xref = int(img[0])
        name = img[7] if len(img) > 7 else None
        try:
            rects = page.get_image_rects(xref)
        except Exception:  # pragma: no cover - defensive
            rects = []
        for r in rects:
            image_regions.append(
                ImageRegion(
                    bbox=BBox.from_tuple((r.x0, r.y0, r.x1, r.y1)),
                    xref=xref,
                    name=str(name) if name else None,
                )
            )

    is_hybrid = bool(text_blocks) and _has_overlaid_text(text_blocks, image_regions)
    return PageLayout(
        number=page.number + 1,
        width=rect.width,
        height=rect.height,
        text_blocks=text_blocks,
        image_regions=image_regions,
        is_hybrid=is_hybrid,
    )


def detect_document(doc: "fitz.Document") -> list[PageLayout]:
    """Detect every page of an already-open document."""
    return [detect_page(page) for page in doc]


def _has_overlaid_text(
    text_blocks: list[RawTextBlock], image_regions: list[ImageRegion]
) -> bool:
    """True when native text overlaps an image region (hybrid scanned+native)."""
    for region in image_regions:
        for tb in text_blocks:
            if region.bbox.intersection_area(tb.bbox) > 0.3 * tb.bbox.area:
                return True
    return False


def is_figure(region: ImageRegion) -> bool:
    """Heuristic: a region large enough to be a meaningful figure, not an icon."""
    b = region.bbox
    return b.area >= _MIN_FIGURE_AREA and b.width >= _MIN_FIGURE_SIDE and b.height >= _MIN_FIGURE_SIDE


def render_region(doc: "fitz.Document", page_number: int, bbox: BBox, dpi: int = 200) -> bytes:
    """Render a page region (1-based ``page_number``) to PNG bytes."""
    page = doc[page_number - 1]
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    clip = fitz.Rect(*bbox.as_tuple())
    pix = page.get_pixmap(matrix=matrix, clip=clip)
    return pix.tobytes("png")
