# Phase A — MVP (native-text path, end-to-end)

The MVP ships a **running end-to-end tool**: a digital (native-text) PDF in,
translated Markdown (+ optional PDF) out, entirely offline.

## What ships

- **Scaffold:** `uv`-managed package, `pyproject.toml` (hatchling,
  `pdf-translate` entry point), structured logging, exception hierarchy, pydantic
  data contracts, and the CLI with the **full flag surface**. Phase B flags
  (`--ocr-backend tesseract|vlm`, `--pdf-engine typst`) are accepted but fail
  with a clear "not-yet" guard (`BackendNotAvailableError`) rather than silently
  misbehaving.
- **Stage 1 — Ingest/Detect:** `fitz` opens the PDF; per page we read text spans
  (`get_text("dict")`) and image rectangles, classify native-text vs image
  regions, and flag hybrid pages.
- **Stage 2 — Extract/Normalize:** font-size clustering → heading levels;
  x-gap/reading-order grouping → paragraphs & lists; monospace → code. Tables via
  `pdfplumber` reconciled to fitz coordinates → GFM. Inline math/code/URLs are
  protected with sentinels before translation.
- **Stage 3 — Translate (async):** `httpx.AsyncClient` streams the local model's
  `/api/chat` (Ollama NDJSON), capped by an `asyncio.Semaphore` for GPU/VRAM
  safety, with live token metrics. A semantic batcher packs whole segments to a
  token budget without ever splitting one. A basic glossary enforces terms via
  placeholders; the validator checks structural parity and retries on drift.
- **Stage 4 — Output:** a self-contained translated `.md` with copied `assets/`,
  plus a recompiled PDF through a **pluggable** engine interface — the
  **weasyprint** backend is implemented (pango is already present, so no extra
  install).

## Native-text data flow

```
fitz spans ─▶ RawTextBlock/PageLayout ─▶ Block[] (+ TableBlock via pdfplumber)
   ─▶ Segment[] (protect math/code/URL + glossary)  ─▶ TranslationBatch[]
   ─▶ local LLM (async, streamed) ─▶ restore sentinels ─▶ translated Block[]
   ─▶ render GFM ─▶ out/<name>.md + assets/  (+ out/<name>.pdf)
```

## CLI surface

```
pdf-translate
  -i/--input PATH            source PDF (required)
  -o/--output PATH           output directory (required)
  --source-lang TEXT         source language or "auto" (default: auto)
  --target-lang TEXT         target language (default: en)
  --model TEXT               chat model (default: qwen2.5:14b; use gemma4:e4b locally)
  --vlm-model TEXT           VLM for OCR (Phase B)
  --concurrency INT          max concurrent model requests (default: 4)
  --ocr-backend [auto|tesseract|vlm|none]   (default: auto; tesseract/vlm are Phase B)
  --pdf-engine [weasyprint|typst|none]      (default: weasyprint; typst is Phase B)
  --glossary PATH            JSON/YAML term map
  --base-url TEXT            model server (default: http://localhost:11434)
  --api [ollama|vllm]        transport (default: ollama)
  --dpi INT                  render DPI for image regions (default: 200)
  --expansion-factor FLOAT   expected text expansion for PDF layout (default: 1.3)
  --no-pdf                   skip PDF compilation
  --verbose/-v               debug logging
```

## Acceptance criteria

- `uv run pdf-translate -i doc.pdf -o out/ --target-lang pt --model gemma4:e4b`
  produces `out/doc.md` and (unless `--no-pdf`) `out/doc.pdf`.
- Block count is preserved end-to-end (no dropped/merged content); tables keep
  their column count; lists keep their length.
- Math, inline code, and URLs survive translation unchanged (sentinel
  round-trip).
- Figures are extracted to `out/assets/` and linked from the Markdown.
- `--ocr-backend tesseract|vlm` and `--pdf-engine typst` fail fast with a clear
  message pointing at Phase B.

## Run & verify

```bash
# unit tests (batcher atomicity, protect/restore round-trip, validator drift)
uv run pytest

# end-to-end against a local model (pull one first, e.g. `ollama pull gemma4:e4b`)
uv run pdf-translate -i sample.pdf -o out/ --target-lang pt --model gemma4:e4b
#   → inspect out/sample.md and out/sample.pdf

uv run pdf-translate --help
```

## Known limitation

weasyprint runs no JavaScript, so inline math renders as its source text (e.g.
`$E=mc^2$`). The typst backend (Phase B) renders math natively.
