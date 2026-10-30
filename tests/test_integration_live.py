"""Live end-to-end test against a local Ollama model.

Skipped unless a local Ollama server is reachable AND the model below is pulled,
so the default `uv run pytest` stays green offline. Run it explicitly with a
pulled model available to exercise the real translation path.
"""

from __future__ import annotations

import httpx
import pytest

from pdf_translator.config import Settings
from pdf_translator.pipeline import run

LIVE_MODEL = "gemma4:e4b"
BASE_URL = "http://localhost:11434"


def _model_available() -> bool:
    try:
        resp = httpx.get(f"{BASE_URL}/api/tags", timeout=2.0)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        return LIVE_MODEL in names
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _model_available(), reason=f"local Ollama model {LIVE_MODEL} not available"
)


def test_live_translation_end_to_end(sample_pdf, tmp_path):
    out = tmp_path / "out"
    settings = Settings(
        input_path=sample_pdf,
        output_dir=out,
        source_lang="en",
        target_lang="pt",
        model=LIVE_MODEL,
        base_url=BASE_URL,
        concurrency=2,
        no_pdf=True,
    )
    result = run(settings)

    assert result.markdown_path.exists()
    md = result.markdown_path.read_text(encoding="utf-8")
    assert md.strip(), "translated markdown is empty"
    # structure preserved
    assert result.block_count >= 4
    assert "|" in md  # table survived
    assert (out / "assets").exists() and list((out / "assets").glob("*.png"))
