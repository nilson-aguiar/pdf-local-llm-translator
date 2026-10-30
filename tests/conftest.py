"""Shared pytest fixtures — notably a synthetic PDF for integration tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from PIL import Image


def _build_sample_pdf(path: Path) -> None:
    """Create a one-page PDF with a heading, paragraph, list, ruled table, image."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 points

    page.insert_text((72, 80), "Annual Report Summary", fontsize=20)
    page.insert_text(
        (72, 120),
        "This document summarizes the yearly results and key findings for the team.",
        fontsize=11,
    )
    page.insert_text((72, 150), "- First important point about revenue", fontsize=11)
    page.insert_text((72, 166), "- Second important point about growth", fontsize=11)

    # Ruled 2x3 table so pdfplumber's line strategy detects it.
    x0, y0, cw, rh, cols, rows = 72, 210, 160, 26, 2, 3
    for i in range(rows + 1):
        page.draw_line((x0, y0 + i * rh), (x0 + cols * cw, y0 + i * rh))
    for j in range(cols + 1):
        page.draw_line((x0 + j * cw, y0), (x0 + j * cw, y0 + rows * rh))
    cells = [["Metric", "Value"], ["Revenue", "100 million"], ["Growth", "15 percent"]]
    for r in range(rows):
        for c in range(cols):
            page.insert_text((x0 + c * cw + 6, y0 + r * rh + 17), cells[r][c], fontsize=10)

    # Embedded raster image (a plain colored block — figure, no text).
    img = Image.new("RGB", (120, 90), (200, 60, 60))
    buf = BytesIO()
    img.save(buf, "PNG")
    page.insert_image(pymupdf.Rect(72, 360, 260, 500), stream=buf.getvalue())

    doc.save(str(path))
    doc.close()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    _build_sample_pdf(path)
    return path
