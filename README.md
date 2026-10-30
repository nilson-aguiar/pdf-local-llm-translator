# pdf-local-llm-translator

A modular, **fully offline, privacy-first** CLI that translates PDFs end-to-end
using a **local** LLM (Ollama or vLLM). No page content ever leaves your machine —
the only network call is to the model server you point `--base-url` at.

```bash
uv run pdf-translate -i doc.pdf -o out/ --source-lang en --target-lang pt --model gemma4:e4b
```

Output is a clean, self-contained Markdown file (with copied figures) plus an
optional recompiled PDF.

## Why

Cloud translation means uploading documents that may be confidential. This tool
keeps everything local: extraction, translation, and rendering all run on your
machine against a model you host. No telemetry, no uploads.

## Features

- **Native vector-text translation** end-to-end (Phase A, shipping now).
- **Structure-preserving:** headings, paragraphs, lists, and tables survive as
  GitHub-Flavored Markdown; column counts and list lengths are enforced.
- **Protected spans:** math (`$…$`), inline/fenced code, and URLs are swapped for
  sentinels so the model can't alter them, then restored verbatim.
- **Anti-hallucination validator:** detects dropped/merged segments and retries
  with a stricter, smaller batch before flagging.
- **Async local inference:** streamed Ollama `/api/chat` (or vLLM `/v1`), with a
  concurrency cap for GPU/VRAM safety and live tok/s metrics.
- **Glossary enforcement:** JSON/YAML term maps pinned through translation.
- **Pluggable PDF engine:** weasyprint implemented; typst planned.
- **Figures** are extracted, linked, and copied so the output folder is
  self-contained.

Coming in Phase B: OCR/VLM fallback for scanned pages and diagrams
(`--ocr-backend tesseract|vlm`), a typst PDF engine, and hardening. See
[`docs/roadmap.md`](docs/roadmap.md).

## Requirements

- [`uv`](https://docs.astral.sh/uv/) (manages Python, deps, and the lockfile).
- A reachable local model server (Ollama or vLLM) with a model pulled.
- macOS system libraries via Homebrew (below). Python deps are handled by `uv`.

### System dependencies (macOS / Homebrew)

```bash
# Tooling + local model server
brew install uv ollama

# Phase A (required now): weasyprint's native rendering libs
brew install pango
# `uv` auto-adds Homebrew's lib dir to the dyld path, so no manual env needed.

# Phase B (optional, for OCR/VLM fallback and the typst PDF engine)
brew install tesseract tesseract-lang poppler typst

# Pull a model (only gemma4:e4b needed to run Phase A locally)
ollama pull gemma4:e4b
# Spec defaults (pull if you want them): qwen2.5:14b (chat), qwen2.5-vl (VLM)
```

On Linux, install the equivalents with your package manager (e.g.
`apt install libpango-1.0-0 libpangocairo-1.0-0 tesseract-ocr poppler-utils`) and
pull models the same way.

## Install & run

```bash
git clone https://github.com/nilson-aguiar/pdf-local-llm-translator.git
cd pdf-local-llm-translator
uv sync                       # create the venv and install deps
uv run pdf-translate --help   # see all flags
uv run pytest                 # run the test suite
```

Or run without cloning-time install via `uvx`:

```bash
uvx --from . pdf-translate -i doc.pdf -o out/ --target-lang pt --model gemma4:e4b
```

## Usage examples

```bash
# Standard digital PDF, English → Portuguese
uv run pdf-translate -i doc.pdf -o out/ --source-lang en --target-lang pt --model gemma4:e4b

# Glossary-enforced translation
uv run pdf-translate -i doc.pdf -o out/ --target-lang pt --glossary config/glossary.yaml

# Skip the recompiled PDF (Markdown only)
uv run pdf-translate -i doc.pdf -o out/ --target-lang de --no-pdf

# Point at a vLLM OpenAI-compatible server
uv run pdf-translate -i doc.pdf -o out/ --target-lang pt --api vllm --base-url http://localhost:8000
```

Output in `out/`:

- `doc.md` — translated, self-contained GFM.
- `assets/` — copied figures, linked from the Markdown.
- `doc.pdf` — recompiled PDF (unless `--no-pdf`).

## How it works

Four decoupled stages connected by validated pydantic models:

```
PDF ─▶ (1) Ingest/Detect ─▶ (2) Extract/Normalize ─▶ (3) Translate ─▶ (4) Output
        fitz page tree        spans→logical blocks      async local LLM   Markdown (+PDF)
```

Full details in [`docs/architecture.md`](docs/architecture.md).

## Project layout

```
src/pdf_translator/
  cli.py config.py logging.py exceptions.py models.py pipeline.py
  ingest/   detector.py  ocr.py
  extract/  blocks.py    tables.py     markdown.py
  translate/client.py    batcher.py    glossary.py  prompts.py  validator.py  segments.py
  output/   markdown_writer.py  pdf_compiler.py
docs/  README.md  architecture.md  mvp.md  roadmap.md
tests/ unit + integration (offline, plus a live Ollama test)
```

## Documentation

- [`docs/README.md`](docs/README.md) — index + privacy guarantee.
- [`docs/architecture.md`](docs/architecture.md) — pipeline, modules, data contracts.
- [`docs/mvp.md`](docs/mvp.md) — Phase A scope, CLI surface, acceptance criteria.
- [`docs/roadmap.md`](docs/roadmap.md) — Phase B and the backlog.

## License

Not yet specified.
