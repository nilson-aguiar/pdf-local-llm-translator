"""Typer CLI — the ``pdf-translate`` entry point.

Exposes the full flag surface (Phase B flags are accepted but guarded). Builds a
validated :class:`Settings`, runs the pipeline, and prints a summary.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_VLM_MODEL, Settings
from .exceptions import PDFTranslatorError
from .logging import console, get_logger, setup_logging
from .pipeline import PipelineResult, run

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Offline, privacy-first PDF translation via a local LLM (Ollama/vLLM).",
)
logger = get_logger(__name__)


class OcrBackend(str, Enum):
    auto = "auto"
    tesseract = "tesseract"
    vlm = "vlm"
    none = "none"


class PdfEngine(str, Enum):
    weasyprint = "weasyprint"
    typst = "typst"
    none = "none"


class ApiMode(str, Enum):
    ollama = "ollama"
    vllm = "vllm"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"pdf-translate {__version__}")
        raise typer.Exit()


@app.command()
def main(
    input: Path = typer.Option(..., "--input", "-i", help="Source PDF file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory."),
    source_lang: str = typer.Option("auto", "--source-lang", help="Source language (or 'auto')."),
    target_lang: str = typer.Option("en", "--target-lang", help="Target language."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Chat model (must be pulled locally)."),
    vlm_model: str = typer.Option(DEFAULT_VLM_MODEL, "--vlm-model", help="VLM model (Phase B)."),
    concurrency: int = typer.Option(4, "--concurrency", help="Max concurrent model requests."),
    ocr_backend: OcrBackend = typer.Option(OcrBackend.auto, "--ocr-backend", help="OCR fallback (tesseract/vlm are Phase B)."),
    pdf_engine: PdfEngine = typer.Option(PdfEngine.weasyprint, "--pdf-engine", help="PDF engine (typst is Phase B)."),
    api: ApiMode = typer.Option(ApiMode.ollama, "--api", help="Transport: ollama or vllm (/v1)."),
    glossary: Optional[Path] = typer.Option(None, "--glossary", help="JSON/YAML term map."),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url", help="Local model server URL."),
    dpi: int = typer.Option(200, "--dpi", help="Render DPI for image regions."),
    expansion_factor: float = typer.Option(1.3, "--expansion-factor", help="Expected text expansion for PDF layout."),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Skip PDF compilation."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    _version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
) -> None:
    """Translate a PDF end-to-end and write Markdown (+ optional PDF)."""
    setup_logging(verbose)
    try:
        settings = Settings(
            input_path=input,
            output_dir=output,
            source_lang=source_lang,
            target_lang=target_lang,
            model=model,
            vlm_model=vlm_model,
            concurrency=concurrency,
            ocr_backend=ocr_backend.value,
            pdf_engine=pdf_engine.value,
            api_mode=api.value,
            glossary_path=glossary,
            base_url=base_url,
            dpi=dpi,
            expansion_factor=expansion_factor,
            no_pdf=no_pdf,
            verbose=verbose,
        )
    except ValidationError as exc:
        console.print(f"[red]Invalid configuration:[/red]\n{exc}")
        raise typer.Exit(2)

    try:
        result = run(settings)
    except PDFTranslatorError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:  # pragma: no cover - last-resort surface
        logger.exception("unexpected failure")
        console.print(f"[red]Unexpected error:[/red] {exc}")
        raise typer.Exit(1)

    _print_summary(settings, result)


def _print_summary(settings: Settings, result: PipelineResult) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_row("Markdown", str(result.markdown_path))
    table.add_row("PDF", str(result.pdf_path) if result.pdf_path else "[dim]skipped[/dim]")
    table.add_row("Pages", str(result.page_count))
    table.add_row("Blocks", str(result.block_count))
    table.add_row(
        "Segments", f"{result.translated_segments} translated / {result.segment_count} total"
    )
    if result.tokens_per_second:
        table.add_row("Throughput", f"{result.tokens_per_second:.0f} tok/s")
    if result.residual_issues:
        table.add_row("[yellow]Warnings[/yellow]", f"{result.residual_issues} unresolved validation issue(s)")
    console.print(
        Panel(table, title=f"[green]Translated → {settings.target_lang}[/green]", expand=False)
    )


if __name__ == "__main__":  # pragma: no cover
    app()
