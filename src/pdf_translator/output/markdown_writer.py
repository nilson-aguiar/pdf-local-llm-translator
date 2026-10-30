"""Stage 4 — write the translated document as self-contained Markdown.

Reassembles blocks in reading order (page, order), renders GFM via
``extract.markdown.render_blocks``, copies referenced figure assets next to the
``.md`` so the output folder is self-contained, and returns the markdown path
and text (the latter is reused by the PDF compiler).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..exceptions import OutputError
from ..extract import markdown
from ..logging import get_logger
from ..models import Asset, Block

logger = get_logger(__name__)


def render_document(blocks: list[Block]) -> str:
    """Sort blocks into reading order and render the full GFM document."""
    ordered = sorted(blocks, key=lambda b: (b.page, b.order))
    return markdown.render_blocks(ordered)


def write_markdown(
    name: str,
    blocks: list[Block],
    assets: list[Asset],
    out_dir: Path,
) -> tuple[Path, str]:
    """Write ``<name>.md`` + copied assets into ``out_dir``. Returns (path, text)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render_document(blocks)
    md_path = out_dir / f"{name}.md"
    try:
        md_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise OutputError(f"failed to write {md_path}: {exc}") from exc

    for asset in assets:
        dest = out_dir / asset.rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(asset.source_path, dest)
        except OSError as exc:
            logger.warning("could not copy asset %s: %s", asset.rel_path, exc)
    return md_path, text
