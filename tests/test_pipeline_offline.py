"""Full-pipeline integration test with translation stubbed to identity.

Exercises every stage (detect → blocks → tables → figures → segment → batch →
validate → restore → reassemble → markdown → PDF) deterministically and offline,
so it verifies wiring without needing a live model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_translator.config import Settings
from pdf_translator.exceptions import OutputError
from pdf_translator.models import TranslationResult
from pdf_translator.pipeline import run
from pdf_translator.translate.client import TranslationClient


@pytest.fixture(autouse=True)
def stub_translation(monkeypatch):
    """Echo each segment's (protected) text back as its 'translation'."""

    async def fake_translate_batch(self, batch, source_lang, target_lang, strict=False, on_delta=None):
        if on_delta:
            on_delta("x")
        return TranslationResult(
            batch_id=batch.id,
            translations={s.id: s.text for s in batch.segments},
            eval_count=len(batch.segments),
            eval_duration_ns=1_000_000,
        )

    monkeypatch.setattr(TranslationClient, "translate_batch", fake_translate_batch)


def _settings(sample_pdf: Path, out: Path, **kw) -> Settings:
    base = dict(
        input_path=sample_pdf,
        output_dir=out,
        target_lang="pt",
        model="stub",
        concurrency=2,
    )
    base.update(kw)
    return Settings(**base)


def test_pipeline_produces_markdown_and_assets(sample_pdf, tmp_path):
    out = tmp_path / "out"
    result = run(_settings(sample_pdf, out, no_pdf=True))

    assert result.markdown_path.exists()
    md = result.markdown_path.read_text(encoding="utf-8")

    # heading + paragraph text survive
    assert "Annual Report Summary" in md
    assert "yearly results" in md

    # table rendered as GFM (pipes + header separator)
    assert "|" in md
    assert "Metric" in md and "Revenue" in md
    assert "---" in md

    # figure extracted, linked, and copied next to the markdown
    assert "![" in md and "assets/" in md
    assets = list((out / "assets").glob("*.png"))
    assert assets, "expected at least one extracted figure asset"

    # structure preserved: at least heading, paragraph, table, figure
    assert result.block_count >= 4
    assert result.residual_issues == 0


def test_pipeline_compiles_pdf(sample_pdf, tmp_path):
    out = tmp_path / "out_pdf"
    try:
        result = run(_settings(sample_pdf, out, no_pdf=False))
    except OutputError as exc:  # weasyprint native libs unavailable in this env
        pytest.skip(f"PDF backend unavailable: {exc}")
    assert result.pdf_path is not None
    assert result.pdf_path.exists()
    assert result.pdf_path.stat().st_size > 0


def test_phase_b_backends_are_guarded(sample_pdf, tmp_path):
    from pdf_translator.exceptions import BackendNotAvailableError

    with pytest.raises(BackendNotAvailableError):
        run(_settings(sample_pdf, tmp_path / "o1", ocr_backend="tesseract", no_pdf=True))
    with pytest.raises(BackendNotAvailableError):
        run(_settings(sample_pdf, tmp_path / "o2", pdf_engine="typst"))
