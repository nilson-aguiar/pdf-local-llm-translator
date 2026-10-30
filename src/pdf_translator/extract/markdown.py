"""Markdown protection and block-to-GFM rendering for the translate pipeline.

This module owns two responsibilities:

1. **Protection** – swapping non-translatable spans (math, code, URLs) with
   opaque sentinel tokens before the text is handed to the translation model,
   and restoring those tokens afterwards.

2. **Rendering** – converting the logical :data:`~pdf_translator.models.Block`
   union into GitHub-Flavored Markdown so the output stage can write ``.md``
   files or compile them to PDF.
"""

from __future__ import annotations

import re
from itertools import count
from typing import TYPE_CHECKING

from pdf_translator.models import (
    apply_restorations,
    make_sentinel,
    Block,
    CaptionBlock,
    CodeBlock,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    MathBlock,
    ParagraphBlock,
    ProtectedText,
    TableBlock,
)

# ---------------------------------------------------------------------------
# Sentinel counter (module-level, incremented per replacement across all calls
# so every token is unique within a session).
# ---------------------------------------------------------------------------

_counter: count[int] = count()


def _next_idx() -> int:
    """Return the next global sentinel index."""
    return next(_counter)


# ---------------------------------------------------------------------------
# Compiled patterns (order of protection matters — see :func:`protect`)
# ---------------------------------------------------------------------------

# Display math must be tried before inline math so ``$$...$$`` is not
# consumed as two ``$`` delimiters by the shorter pattern.
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$(.+?)\$", re.DOTALL)

# Fenced code (``` ``` ```) before inline (`` ` ``), same reason.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`(.+?)`")

# Bare URLs starting with http://, https://, or www.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def protect(text: str) -> ProtectedText:
    """Replace non-translatable spans in *text* with sentinel tokens.

    Spans are replaced in a fixed order so inner delimiters are never
    double-matched:

    1. Display math ``$$...$$`` (tag ``MATH``)
    2. Inline math ``$...$`` (tag ``MATH``)
    3. Fenced code blocks ```` ```...``` ```` (tag ``CODE``)
    4. Inline code `` `...` `` (tag ``CODE``)
    5. Bare URLs ``http://``, ``https://``, ``www.`` (tag ``URL``)

    The returned :class:`~pdf_translator.models.ProtectedText` records every
    sentinel → original mapping, guaranteeing round-trip:
    ``restore(pt.text, pt.restorations) == original_text``.
    """
    restorations: dict[str, str] = {}

    def _sub(pattern: re.Pattern[str], tag: str, src: str) -> str:
        def _replacer(m: re.Match[str]) -> str:
            sentinel = make_sentinel(tag, _next_idx())
            restorations[sentinel] = m.group(0)
            return sentinel

        return pattern.sub(_replacer, src)

    text = _sub(_DISPLAY_MATH_RE, "MATH", text)
    text = _sub(_INLINE_MATH_RE, "MATH", text)
    text = _sub(_FENCED_CODE_RE, "CODE", text)
    text = _sub(_INLINE_CODE_RE, "CODE", text)
    text = _sub(_URL_RE, "URL", text)

    return ProtectedText(text=text, restorations=restorations)


def restore(text: str, restorations: dict[str, str]) -> str:
    """Restore sentinel tokens in *text* to their original values.

    Thin wrapper over :func:`~pdf_translator.models.apply_restorations`
    re-exported so callers can do ``from pdf_translator.extract.markdown import
    restore`` without knowing the underlying utility.
    """
    return apply_restorations(text, restorations)


# ---------------------------------------------------------------------------
# Block → GFM rendering
# ---------------------------------------------------------------------------


def _escape_pipe(cell: str) -> str:
    """Escape literal ``|`` in a table cell so GFM does not split the column."""
    return cell.replace("|", "\\|")


def _render_row(row: list[str]) -> str:
    """Render one table row as a GFM pipe-table line."""
    cells = " | ".join(_escape_pipe(c) for c in row)
    return f"| {cells} |"


def block_to_markdown(block: Block) -> str:  # type: ignore[type-arg]
    """Render a single logical *block* to GitHub-Flavored Markdown.

    Each block kind is rendered according to the GFM spec:

    - **heading** → ``#``×level + space + content
    - **paragraph** → content verbatim
    - **list** → ``- item`` lines (``1.``/``2.``… when ordered)
    - **table** → pipe table; first row is the header row when
      ``has_header`` is ``True``, followed by a ``---`` separator row
    - **code** → fenced block with optional language tag
    - **math** → ``$$\\n{content}\\n$$`` (display) or ``${content}$`` (inline)
    - **figure** → ``![{caption}]({asset_path})``
    - **caption** → ``*{content}*``
    """
    if isinstance(block, HeadingBlock):
        prefix = "#" * block.level
        return f"{prefix} {block.content}"

    if isinstance(block, ParagraphBlock):
        return block.content

    if isinstance(block, ListBlock):
        lines: list[str] = []
        for i, item in enumerate(block.items):
            marker = f"{i + 1}." if block.ordered else "-"
            lines.append(f"{marker} {item}")
        return "\n".join(lines)

    if isinstance(block, TableBlock):
        if not block.rows:
            return ""
        lines = []
        if block.has_header:
            lines.append(_render_row(block.rows[0]))
            n_cols = len(block.rows[0])
            lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
            for row in block.rows[1:]:
                lines.append(_render_row(row))
        else:
            for row in block.rows:
                lines.append(_render_row(row))
        return "\n".join(lines)

    if isinstance(block, CodeBlock):
        lang = block.language or ""
        return f"```{lang}\n{block.content}\n```"

    if isinstance(block, MathBlock):
        if block.display:
            return f"$$\n{block.content}\n$$"
        return f"${block.content}$"

    if isinstance(block, FigureBlock):
        alt = block.caption or ""
        img = f"![{alt}]({block.asset_path})"
        if block.caption:
            return f"{img}\n\n*{block.caption}*"
        return img

    if isinstance(block, CaptionBlock):
        return f"*{block.content}*"

    # Fallback: unknown block kind (should not happen with the frozen union)
    return ""


def render_blocks(blocks: list[Block]) -> str:  # type: ignore[type-arg]
    """Render *blocks* to a single GFM document string.

    Blocks are separated by a blank line and the result ends with a trailing
    newline.  The caller is responsible for ordering blocks (e.g. by
    ``(page, order)``) before passing them here.
    """
    return "\n\n".join(block_to_markdown(b) for b in blocks) + "\n"
