"""Runtime configuration.

:class:`Settings` mirrors the full CLI flag surface so the whole pipeline reads
from one validated object. Environment variables (prefix ``PDF_TRANSLATE_``) can
override any field, which is handy for the local server URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Documented spec defaults. NOTE: these models must be pulled locally first
# (`ollama pull qwen2.5:14b`, `ollama pull qwen2.5-vl`). On this machine only
# `gemma4:e4b` / `llama3.1` are available — pass `--model gemma4:e4b` to run now.
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_VLM_MODEL = "qwen2.5-vl"
DEFAULT_BASE_URL = "http://localhost:11434"

OcrBackend = Literal["auto", "tesseract", "vlm", "none"]
PdfEngine = Literal["weasyprint", "typst", "none"]
ApiMode = Literal["ollama", "vllm"]


class Settings(BaseSettings):
    """Validated configuration for a single translation run."""

    model_config = SettingsConfigDict(
        env_prefix="PDF_TRANSLATE_",
        extra="ignore",
        protected_namespaces=(),  # we use a field literally named `model`
    )

    # --- I/O ---
    input_path: Path
    output_dir: Path

    # --- languages ---
    source_lang: str = "auto"
    target_lang: str = "en"

    # --- models / transport ---
    model: str = DEFAULT_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_mode: ApiMode = "ollama"
    concurrency: int = 4
    request_timeout: float = 600.0
    max_retries: int = 3

    # --- backends ---
    ocr_backend: OcrBackend = "auto"
    pdf_engine: PdfEngine = "weasyprint"
    no_pdf: bool = False

    # --- extraction / rendering knobs ---
    glossary_path: Path | None = None
    dpi: int = 200
    expansion_factor: float = 1.3
    batch_token_budget: int = 1200

    # --- misc ---
    verbose: bool = False

    @property
    def chat_endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if self.api_mode == "vllm":
            # vLLM exposes the OpenAI-compatible surface under /v1.
            if base.endswith("/v1"):
                return f"{base}/chat/completions"
            return f"{base}/v1/chat/completions"
        return f"{base}/api/chat"

    @property
    def effective_pdf_engine(self) -> PdfEngine | None:
        """Resolve the PDF engine, honoring ``--no-pdf``."""
        if self.no_pdf:
            return None
        return self.pdf_engine
