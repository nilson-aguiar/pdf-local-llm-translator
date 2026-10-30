"""Tests for pdf_translator.extract.markdown — protection and GFM rendering."""

from __future__ import annotations

import pytest

from pdf_translator.extract.markdown import block_to_markdown, protect, render_blocks, restore
from pdf_translator.models import (
    CaptionBlock,
    CodeBlock,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    MathBlock,
    ParagraphBlock,
    TableBlock,
    apply_restorations,
)


# ---------------------------------------------------------------------------
# protect / restore round-trip
# ---------------------------------------------------------------------------


def test_protect_restore_round_trip_mixed():
    """Protecting a rich string and restoring it yields the original."""
    original = (
        "Inline math: $x^2 + y^2$ and display: $$\\int_0^1 f(t)\\, dt$$ "
        "plus code: `foo(bar)` plus fenced:\n"
        "```python\nprint('hello')\n```\n"
        "and a URL: https://example.com/path?q=1"
    )
    pt = protect(original)
    assert restore(pt.text, pt.restorations) == original


def test_protect_removes_raw_delimiters():
    """The protected text must contain no $, backtick, or raw http."""
    original = (
        "Formula $x^2$ and display $$E=mc^2$$ "
        "and code `snippet` "
        "and fenced ```lang\ncode\n``` "
        "and url https://example.com"
    )
    pt = protect(original)
    assert "$" not in pt.text, "dollar signs should be replaced"
    assert "`" not in pt.text, "backticks should be replaced"
    assert "http" not in pt.text, "bare URLs should be replaced"


def test_protect_display_math_before_inline():
    """$$...$$ must be captured as one unit, not split as two $..$ pairs."""
    original = "Display: $$a + b$$"
    pt = protect(original)
    # Exactly one sentinel should cover the whole $$...$$ span.
    assert pt.restorations.values().__len__() == 1
    restored_val = next(iter(pt.restorations.values()))
    assert restored_val == "$$a + b$$"


def test_protect_fenced_before_inline_code():
    """Fenced ``` blocks must be captured before inline ` spans."""
    original = "Block:\n```python\nx = `y`\n```\nDone."
    pt = protect(original)
    # The fenced block sentinel should contain the whole block (including inner backticks).
    assert any("```python" in v for v in pt.restorations.values()), (
        "fenced block should be in restorations"
    )
    # No remaining backticks in protected text.
    assert "`" not in pt.text


def test_protect_url():
    """Bare URLs are replaced with URL sentinels."""
    original = "See https://example.org for more."
    pt = protect(original)
    assert "https" not in pt.text
    assert restore(pt.text, pt.restorations) == original


def test_restore_is_thin_wrapper():
    """restore() delegates to apply_restorations correctly."""
    restorations = {"⟦MATH0⟧": "$x$"}
    assert restore("value: ⟦MATH0⟧", restorations) == "value: $x$"


# ---------------------------------------------------------------------------
# block_to_markdown — individual block kinds
# ---------------------------------------------------------------------------


def test_heading_level_1():
    block = HeadingBlock(level=1, content="Introduction", page=1, order=0)
    assert block_to_markdown(block) == "# Introduction"


def test_heading_level_3():
    block = HeadingBlock(level=3, content="Subsection", page=1, order=0)
    assert block_to_markdown(block) == "### Subsection"


def test_paragraph():
    block = ParagraphBlock(content="Hello world.", page=1, order=0)
    assert block_to_markdown(block) == "Hello world."


def test_unordered_list():
    block = ListBlock(ordered=False, items=["alpha", "beta", "gamma"], page=1, order=0)
    assert block_to_markdown(block) == "- alpha\n- beta\n- gamma"


def test_ordered_list():
    block = ListBlock(ordered=True, items=["first", "second"], page=1, order=0)
    assert block_to_markdown(block) == "1. first\n2. second"


def test_table_with_header():
    block = TableBlock(
        rows=[["Name", "Score"], ["Alice", "95"], ["Bob", "87"]],
        has_header=True,
        page=1,
        order=0,
    )
    md = block_to_markdown(block)
    lines = md.splitlines()
    # Header row
    assert lines[0] == "| Name | Score |"
    # Separator row
    assert lines[1] == "| --- | --- |"
    # Data rows
    assert lines[2] == "| Alice | 95 |"
    assert lines[3] == "| Bob | 87 |"


def test_table_without_header():
    block = TableBlock(
        rows=[["A", "B"], ["C", "D"]],
        has_header=False,
        page=1,
        order=0,
    )
    md = block_to_markdown(block)
    lines = md.splitlines()
    assert len(lines) == 2
    assert "---" not in md


def test_table_pipe_escaping():
    block = TableBlock(
        rows=[["a|b", "c"], ["d", "e|f"]],
        has_header=False,
        page=1,
        order=0,
    )
    md = block_to_markdown(block)
    assert r"\|" in md


def test_code_block_with_language():
    block = CodeBlock(content="print('hi')", language="python", page=1, order=0)
    assert block_to_markdown(block) == "```python\nprint('hi')\n```"


def test_code_block_no_language():
    block = CodeBlock(content="x = 1", language=None, page=1, order=0)
    assert block_to_markdown(block) == "```\nx = 1\n```"


def test_math_display():
    block = MathBlock(content=r"\int_0^1 f\, dt", display=True, page=1, order=0)
    md = block_to_markdown(block)
    assert md == "$$\n\\int_0^1 f\\, dt\n$$"


def test_math_inline():
    block = MathBlock(content="x^2", display=False, page=1, order=0)
    assert block_to_markdown(block) == "$x^2$"


def test_figure_with_caption():
    block = FigureBlock(asset_path="assets/fig1.png", caption="A nice figure", page=1, order=0)
    md = block_to_markdown(block)
    assert "![A nice figure](assets/fig1.png)" in md
    assert "*A nice figure*" in md


def test_figure_no_caption():
    block = FigureBlock(asset_path="assets/fig2.png", caption=None, page=1, order=0)
    md = block_to_markdown(block)
    assert md == "![](assets/fig2.png)"


def test_caption_block():
    block = CaptionBlock(content="Figure 1: Results.", page=1, order=0)
    assert block_to_markdown(block) == "*Figure 1: Results.*"


# ---------------------------------------------------------------------------
# render_blocks — integration
# ---------------------------------------------------------------------------


def test_render_blocks_heading_paragraph_list():
    blocks = [
        HeadingBlock(level=2, content="Overview", page=1, order=0),
        ParagraphBlock(content="Some text here.", page=1, order=1),
        ListBlock(ordered=False, items=["item one", "item two"], page=1, order=2),
    ]
    md = render_blocks(blocks)
    assert md.endswith("\n"), "render_blocks must end with a newline"
    parts = md.strip().split("\n\n")
    assert parts[0] == "## Overview"
    assert parts[1] == "Some text here."
    assert parts[2] == "- item one\n- item two"


def test_render_blocks_table_gfm_valid():
    """The rendered table must have a header separator and consistent column counts."""
    blocks = [
        HeadingBlock(level=1, content="Results", page=1, order=0),
        ParagraphBlock(content="See the table below.", page=1, order=1),
        TableBlock(
            rows=[
                ["Model", "Accuracy", "F1"],
                ["GPT-4", "0.92", "0.91"],
                ["Llama", "0.88", "0.87"],
            ],
            has_header=True,
            page=1,
            order=2,
        ),
    ]
    md = render_blocks(blocks)

    # Table section is the third block (index 2 after splitting on blank lines)
    sections = md.strip().split("\n\n")
    table_section = sections[2]
    table_lines = table_section.splitlines()

    # Separator must exist
    assert any("---" in line for line in table_lines), "GFM separator row missing"

    # Every row must have the same number of pipe characters (column count parity)
    pipe_counts = [line.count("|") for line in table_lines]
    assert len(set(pipe_counts)) == 1, (
        f"Inconsistent column counts across rows: {pipe_counts}"
    )


def test_render_blocks_trailing_newline():
    blocks = [ParagraphBlock(content="Only one.", page=1, order=0)]
    md = render_blocks(blocks)
    assert md == "Only one.\n"
