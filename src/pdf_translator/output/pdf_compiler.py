"""Stage 4 — recompile translated Markdown to PDF.

A small pluggable engine interface so Phase B can drop in a typst backend
without touching callers. The **weasyprint** backend is implemented; it renders
Markdown → HTML (markdown-it-py, GFM preset) → PDF, with CSS that adds generous
line-height and ``table-layout:auto`` to absorb the 15–30% text expansion typical
of EN→PT/DE without clipping.

Known limitation: weasyprint runs no JavaScript, so inline math renders as its
source text. The typst backend (Phase B) renders math natively.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Protocol

from markdown_it import MarkdownIt

from ..config import Settings
from ..exceptions import BackendNotAvailableError, OutputError
from ..logging import get_logger

logger = get_logger(__name__)

# Homebrew lib dirs where weasyprint's native deps (glib/pango/cairo) live on
# macOS; ctypes won't find them unless they're on the dyld fallback path.
_BREW_LIB_DIRS = ("/opt/homebrew/lib", "/usr/local/lib")


def _ensure_native_libs() -> None:
    """On macOS, make Homebrew's glib/pango/… discoverable before importing
    weasyprint (its native libs are loaded via ctypes at import time)."""
    if platform.system() != "Darwin":
        return
    key = "DYLD_FALLBACK_LIBRARY_PATH"
    existing = os.environ.get(key, "")
    parts = [p for p in existing.split(os.pathsep) if p]
    changed = False
    for d in _BREW_LIB_DIRS:
        if os.path.isdir(d) and d not in parts:
            parts.append(d)
            changed = True
    if changed:
        os.environ[key] = os.pathsep.join(parts)


def _css(settings: Settings) -> str:
    # Feed the expansion factor into line spacing so longer target text breathes.
    line_height = round(min(1.3 + (settings.expansion_factor - 1.0), 2.0), 2)
    return f"""
    @page {{ margin: 2cm; size: A4; }}
    body {{
        font-family: "Helvetica Neue", Arial, "Noto Sans", sans-serif;
        line-height: {line_height};
        font-size: 11pt;
        color: #1a1a1a;
    }}
    h1, h2, h3, h4, h5, h6 {{ line-height: 1.25; margin: 1.1em 0 0.5em; }}
    p {{ margin: 0 0 0.7em; }}
    code, pre {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    pre {{ background: #f5f5f5; padding: 0.8em; border-radius: 4px; white-space: pre-wrap;
           word-wrap: break-word; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: auto; margin: 0.8em 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35em 0.6em; text-align: left;
              vertical-align: top; word-wrap: break-word; }}
    th {{ background: #f0f0f0; }}
    img {{ max-width: 100%; height: auto; }}
    blockquote {{ border-left: 3px solid #ddd; margin: 0.8em 0; padding-left: 1em; color: #555; }}
    """


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{css}</style></head>
<body>{body}</body></html>"""


class PdfEngine(Protocol):
    name: str

    def compile(self, markdown_text: str, out_path: Path, base_dir: Path, settings: Settings) -> None:
        ...


class WeasyPrintEngine:
    name = "weasyprint"

    def compile(self, markdown_text: str, out_path: Path, base_dir: Path, settings: Settings) -> None:
        _ensure_native_libs()
        try:
            from weasyprint import HTML  # imported lazily (native libs)
        except OSError as exc:  # pragma: no cover - environment dependent
            raise OutputError(
                "weasyprint could not load its native libraries (pango/glib/cairo). "
                "On macOS: `brew install pango`. Details: " + str(exc)
            ) from exc
        md = MarkdownIt("gfm-like")
        body = md.render(markdown_text)
        html_doc = _HTML_TEMPLATE.format(css=_css(settings), body=body)
        try:
            # base_url lets relative asset links (assets/...) resolve.
            HTML(string=html_doc, base_url=str(base_dir)).write_pdf(str(out_path))
        except Exception as exc:
            raise OutputError(f"weasyprint failed to write PDF: {exc}") from exc


class TypstEngine:
    name = "typst"

    def compile(self, markdown_text: str, out_path: Path, base_dir: Path, settings: Settings) -> None:
        raise BackendNotAvailableError(
            "typst PDF engine is a Phase B feature. Use --pdf-engine weasyprint "
            "(default) or --no-pdf for now. See docs/roadmap.md."
        )


def get_engine(name: str) -> PdfEngine:
    if name == "weasyprint":
        return WeasyPrintEngine()
    if name == "typst":
        return TypstEngine()
    raise BackendNotAvailableError(f"unknown PDF engine {name!r}")


def compile_pdf(markdown_text: str, out_path: Path, base_dir: Path, settings: Settings) -> None:
    """Compile ``markdown_text`` to ``out_path`` using the configured engine."""
    engine = get_engine(settings.pdf_engine)
    logger.debug("compiling PDF with %s engine", engine.name)
    engine.compile(markdown_text, out_path, base_dir, settings)
