# pdf-translate — documentation

A modular, **fully offline** CLI that translates PDFs end-to-end and emits clean
Markdown (plus an optional recompiled PDF). Translation runs against a **local**
Ollama/vLLM instance — **no data ever leaves the machine**.

## Privacy / offline guarantee

- The only network calls are to the model server you point `--base-url` at
  (default `http://localhost:11434`, i.e. a local Ollama).
- No telemetry, no cloud APIs, no uploads. PDFs, extracted images, and
  translations stay on local disk.
- Figures are extracted to a local temp dir and copied into your output folder;
  nothing is sent anywhere for OCR in the native-text path.

## Documents

| Doc | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | The 4-stage pipeline, module layout, pydantic data contracts, where the seams are. |
| [mvp.md](mvp.md) | **Phase A** — the native-text path that ships now: scope, CLI, acceptance criteria, verify steps. |
| [roadmap.md](roadmap.md) | **Phase B+** — OCR/VLM fallbacks, typst engine, hardening, and the backlog. |

## Quickstart

```bash
uv sync
uv run pdf-translate --help

# Translate a digital PDF to Portuguese with a locally-pulled model:
uv run pdf-translate -i doc.pdf -o out/ --source-lang en --target-lang pt --model gemma4:e4b
```

Output in `out/`:
- `doc.md` — translated, self-contained GitHub-Flavored Markdown.
- `assets/` — copied figures, linked from the Markdown.
- `doc.pdf` — recompiled PDF (unless `--no-pdf`).
