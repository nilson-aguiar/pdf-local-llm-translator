"""Pipeline orchestration — wires stages 1–4 and progress reporting.

``run(settings)`` is synchronous; the translate stage runs its own asyncio loop
internally. Each stage communicates only through the pydantic models in
:mod:`pdf_translator.models`, so this module is just glue + progress + guards.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Iterator, Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .config import Settings
from .exceptions import BackendNotAvailableError
from .extract.blocks import build_blocks
from .extract import tables as tables_mod
from .ingest import detector
from .logging import console, get_logger
from .models import (
    Asset,
    Block,
    FigureBlock,
    PageLayout,
    apply_restorations,
)
from .output.markdown_writer import write_markdown
from .output.pdf_compiler import compile_pdf
from .translate.batcher import batch_segments
from .translate.client import TranslationClient
from .translate.glossary import Glossary, load_glossary
from .translate.segments import build_segments, is_translatable, reassemble
from .translate.validator import validate_batch

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    markdown_path: Path
    pdf_path: Optional[Path] = None
    page_count: int = 0
    block_count: int = 0
    segment_count: int = 0
    translated_segments: int = 0
    residual_issues: int = 0
    tokens_per_second: float = 0.0


# ---------------------------------------------------------------------------
# Guards & warnings for Phase A scope
# ---------------------------------------------------------------------------


def _guard_backends(settings: Settings) -> None:
    if settings.ocr_backend in ("tesseract", "vlm"):
        raise BackendNotAvailableError(
            f"--ocr-backend {settings.ocr_backend} is a Phase B feature and is not yet "
            "wired. Use --ocr-backend auto (native text only) or none. See docs/roadmap.md."
        )
    if not settings.no_pdf and settings.pdf_engine == "typst":
        raise BackendNotAvailableError(
            "--pdf-engine typst is a Phase B feature. Use weasyprint (default) or "
            "--no-pdf. See docs/roadmap.md."
        )


def _warn_unhandled_images(pages: list[PageLayout]) -> None:
    scanned = [p.number for p in pages if not p.has_native_text and p.image_regions]
    hybrid = [p.number for p in pages if p.is_hybrid]
    if scanned:
        logger.warning(
            "pages %s appear scanned (no native text); their embedded text is NOT "
            "translated in Phase A — OCR/VLM is Phase B.",
            scanned,
        )
    if hybrid:
        logger.warning(
            "pages %s are hybrid (native text over images); only the native text is "
            "translated in Phase A.",
            hybrid,
        )


# ---------------------------------------------------------------------------
# Stage 1 helper — extract figures to a temp dir
# ---------------------------------------------------------------------------


def _extract_figures(
    doc, pages: list[PageLayout], settings: Settings, tmp_dir: Path
) -> tuple[list[Asset], dict[int, list[Block]]]:
    assets: list[Asset] = []
    figures_by_page: dict[int, list[Block]] = {}
    for page in pages:
        for i, region in enumerate(page.image_regions):
            if not detector.is_figure(region):
                continue
            rel_path = f"assets/fig_p{page.number}_{i}.png"
            src = tmp_dir / f"fig_p{page.number}_{i}.png"
            try:
                src.write_bytes(detector.render_region(doc, page.number, region.bbox, settings.dpi))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not render figure on page %s: %s", page.number, exc)
                continue
            assets.append(
                Asset(
                    id=rel_path,
                    source_path=str(src),
                    rel_path=rel_path,
                    page=page.number,
                    bbox=region.bbox,
                )
            )
            figures_by_page.setdefault(page.number, []).append(
                FigureBlock(asset_path=rel_path, caption=None, bbox=region.bbox, page=page.number)
            )
    return assets, figures_by_page


# ---------------------------------------------------------------------------
# Stage 3 — async translation with live metrics and drift retry
# ---------------------------------------------------------------------------


@contextmanager
def _progress(total: int) -> Iterator[tuple[Progress, int, dict]]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Translating"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[rate]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task = progress.add_task("translate", total=total, rate="")
        yield progress, task, {"chars": 0, "start": time.monotonic()}


async def _resilient_batch(client, batch, settings, source, target, on_delta):
    """Translate a batch; on structural drift retry stricter + smaller."""
    result = await client.translate_batch(batch, source, target, strict=False, on_delta=on_delta)
    report = validate_batch(batch, result.translations)
    if report.ok:
        return result.translations, [], result

    logger.info("batch %s drifted (%d issue(s)); retrying stricter/smaller", batch.id, len(report.issues))
    merged = dict(result.translations)
    smaller = batch_segments(batch.segments, max(64, settings.batch_token_budget // 2), max_segments=5)
    for sub in smaller:
        r = await client.translate_batch(sub, source, target, strict=True, on_delta=on_delta)
        merged.update(r.translations)

    final_report = validate_batch(batch, merged)
    # Last resort: fill anything still missing with the (protected) source so the
    # document still completes; these are counted as residual issues.
    for seg in batch.segments:
        if seg.id not in merged or not merged[seg.id].strip():
            merged[seg.id] = seg.text
    return merged, final_report.issues, result


async def _translate_blocks(
    blocks: list[Block], settings: Settings, glossary: Glossary
) -> tuple[list[Block], int, int, float]:
    segments = build_segments(blocks, glossary)
    translatable = [s for s in segments if is_translatable(s.text)]
    raw_by_id: dict[int, str] = {s.id: s.text for s in segments if not is_translatable(s.text)}
    batches = batch_segments(translatable, settings.batch_token_budget)

    residual_issues: list = []
    total_eval = 0
    total_ns = 0

    if batches:
        with _progress(len(batches)) as (progress, task, state):
            def on_delta(piece: str) -> None:
                state["chars"] += len(piece)
                elapsed = max(1e-6, time.monotonic() - state["start"])
                rate = (state["chars"] / 4) / elapsed
                progress.update(task, rate=f"{rate:.0f} tok/s")

            async with TranslationClient(settings) as client:
                async def worker(batch) -> None:
                    nonlocal total_eval, total_ns
                    merged, issues, result = await _resilient_batch(
                        client, batch, settings, settings.source_lang, settings.target_lang, on_delta
                    )
                    raw_by_id.update(merged)
                    residual_issues.extend(issues)
                    total_eval += result.eval_count
                    total_ns += result.eval_duration_ns
                    progress.advance(task)

                await asyncio.gather(*(worker(b) for b in batches))

    if residual_issues:
        logger.warning(
            "translation validation flagged %d residual issue(s) after retries", len(residual_issues)
        )

    final_by_id: dict[int, str] = {}
    for seg in segments:
        raw = raw_by_id.get(seg.id, seg.text)
        final_by_id[seg.id] = apply_restorations(raw, seg.restorations)

    _verify_glossary(glossary, final_by_id)

    translated = reassemble(blocks, segments, final_by_id)
    tok_s = (total_eval / (total_ns / 1e9)) if total_ns > 0 else 0.0
    return translated, len(translatable), len(residual_issues), tok_s


def _verify_glossary(glossary: Glossary, final_by_id: dict[int, str]) -> None:
    if not glossary.terms:
        return
    missing: set[str] = set()
    for text in final_by_id.values():
        missing.update(glossary.verify(text))
    if missing:
        logger.warning("glossary: target term(s) not found in output: %s", sorted(missing))


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def run(settings: Settings) -> PipelineResult:
    """Run the full pipeline and return a summary of what was produced."""
    _guard_backends(settings)
    if not settings.input_path.exists():
        from .exceptions import IngestError

        raise IngestError(f"input PDF not found: {settings.input_path}")

    glossary = (
        load_glossary(settings.glossary_path, settings.target_lang)
        if settings.glossary_path
        else Glossary.empty()
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="pdf_translate_"))
    doc = detector.open_document(str(settings.input_path))
    try:
        logger.info("ingesting %s", settings.input_path)
        pages = detector.detect_document(doc)
        _warn_unhandled_images(pages)

        assets, figures_by_page = _extract_figures(doc, pages, settings, tmp_dir)
        tables_by_page = tables_mod.extract_tables(str(settings.input_path))
        blocks = build_blocks(pages, tables_by_page, figures_by_page)
        logger.info(
            "extracted %d blocks (%d tables, %d figures) from %d pages",
            len(blocks),
            sum(len(v) for v in tables_by_page.values()),
            len(assets),
            len(pages),
        )

        translated, translated_count, residual, tok_s = asyncio.run(
            _translate_blocks(blocks, settings, glossary)
        )

        name = settings.input_path.stem
        md_path, md_text = write_markdown(name, translated, assets, settings.output_dir)
        logger.info("wrote %s", md_path)

        pdf_path: Optional[Path] = None
        if settings.effective_pdf_engine:
            pdf_path = settings.output_dir / f"{name}.pdf"
            compile_pdf(md_text, pdf_path, settings.output_dir, settings)
            logger.info("wrote %s", pdf_path)

        return PipelineResult(
            markdown_path=md_path,
            pdf_path=pdf_path,
            page_count=len(pages),
            block_count=len(blocks),
            segment_count=sum(len(b.translatable_texts()) for b in blocks),
            translated_segments=translated_count,
            residual_issues=residual,
            tokens_per_second=tok_s,
        )
    finally:
        doc.close()
        rmtree(tmp_dir, ignore_errors=True)
