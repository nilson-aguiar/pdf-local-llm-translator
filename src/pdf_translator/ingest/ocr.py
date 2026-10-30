"""Stage 1 — OCR / VLM fallback for image regions.

Phase B. The tesseract (bbox) and VLM-OCR (local Ollama ``/api/chat``) backends
are not yet wired. The functions below present the intended interface and fail
loudly with :class:`BackendNotAvailableError` so the full CLI flag surface is
accepted today without silently mistranslating scanned pages.

See ``docs/roadmap.md`` for the planned implementation.
"""

from __future__ import annotations

from ..exceptions import BackendNotAvailableError
from ..models import Block, ImageRegion

_PHASE_B = (
    "{backend!r} OCR backend is a Phase B feature and is not yet wired up. "
    "Use --ocr-backend auto (native text only) or --ocr-backend none for now. "
    "See docs/roadmap.md."
)


def ocr_region_tesseract(image_png: bytes, region: ImageRegion, lang: str = "eng") -> list[Block]:
    """Planned: pytesseract image_to_data → words+bbox+conf → reassembled blocks."""
    raise BackendNotAvailableError(_PHASE_B.format(backend="tesseract"))


def ocr_region_vlm(image_png: bytes, region: ImageRegion, model: str, base_url: str) -> list[Block]:
    """Planned: base64 region → local VLM /api/chat → structured markdown blocks."""
    raise BackendNotAvailableError(_PHASE_B.format(backend="vlm"))
