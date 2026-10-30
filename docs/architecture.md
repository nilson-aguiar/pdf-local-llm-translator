# Architecture

Four decoupled stages, connected by validated pydantic models. Each stage is
independently unit-testable; swapping an implementation (e.g. adding an OCR
backend, or a second PDF engine) touches one stage only.

```
PDF ─▶ (1) Ingest/Detect ─▶ (2) Extract/Normalize ─▶ (3) Translate ─▶ (4) Output
        fitz page tree        spans→logical blocks      async local LLM   Markdown (+PDF)
        text vs image         GFM + protected spans      semantic batches  linked assets
        OCR/VLM fallback*     tables via pdfplumber       glossary + valid. recompiled PDF
```

`*` OCR/VLM fallback is Phase B; the native-text path is Phase A (shipping now).

## Module layout

```
src/pdf_translator/
  cli.py            typer app + all flags, rich progress/live metrics
  config.py         pydantic Settings, model targets, endpoint resolution
  logging.py        structured logging via a shared rich console
  exceptions.py     custom exception hierarchy (Ingest/Extract/Translate/Output)
  models.py         data contracts + the shared sentinel/token protocol
  pipeline.py       orchestrates stages 1–4, wires progress
  ingest/
    detector.py     fitz: get_text("dict") spans + image rects; classify regions
    ocr.py          Phase B: tesseract (bbox) and VLM-OCR backends (guarded stub now)
  extract/
    blocks.py       group spans → logical blocks (heading/para/list/code); reading order
    tables.py       pdfplumber tables → GFM rows (bbox-reconciled to fitz coords)
    markdown.py     blocks → GFM; protect math/code/URL via sentinels
  translate/
    client.py       httpx.AsyncClient; Ollama /api/chat (NDJSON) + vLLM /v1; semaphore; retry
    batcher.py      semantic batching — never split a segment
    glossary.py     JSON/YAML term map; enforce via placeholders; post-verify
    prompts.py      system/user templates (preserve structure, keep sentinels verbatim)
    validator.py    anti-hallucination: sentinel parity, fence/delimiter balance, counts
    segments.py     block ⇄ segment extraction/reassembly (protect + glossary compose here)
  output/
    markdown_writer.py  reassemble reading order → self-contained .md + assets/ copies
    pdf_compiler.py     markdown→HTML→PDF (weasyprint); typst backend is Phase B
```

## Data contracts (`models.py`)

Everything flows as pydantic models, so stages share one source of truth.

- **Geometry / raw:** `BBox`, `TextSpan`, `TextLine`, `RawTextBlock`,
  `ImageRegion`, `PageLayout` (Stage 1 output).
- **Logical blocks:** `Block` is a **discriminated union** on `kind`:
  `heading | paragraph | list | table | code | math | figure | caption`. Each
  carries `page`, `order`, `bbox`, kind-specific content, and a uniform
  translation interface: `translatable_texts() -> list[str]` and
  `with_translations(texts) -> Block`. Native-text and (future) OCR/VLM blocks
  both normalize to this union, so everything downstream treats them identically.
- **Document:** `Document` = source metadata + ordered `list[Block]` + `Asset`s.
- **Translation:** `Segment` (one atomic, already-protected text unit),
  `TranslationBatch`, `TranslationResult` (with tok/s metrics).
- **Validation:** `ValidationIssue`, `ValidationReport`.
- **Wire:** `ChatMessage`, `OllamaChatRequest`/`OllamaChatChunk`,
  `OpenAIChatRequest`.

### The sentinel protocol

A **sentinel** is an opaque token (`⟦MATH0⟧`, `⟦CODE1⟧`, `⟦URL0⟧`, `⟦GLOSS0⟧`)
that replaces a span the model must not change (math, code, URLs) or a glossary
term that must map to a fixed target word. `models.py` owns the protocol
(`make_sentinel`, `SENTINEL_RE`, `find_sentinels`, `apply_restorations`,
`ProtectedText`) so the markdown-protection, glossary, prompt, and validator
layers stay decoupled while speaking the same language:

1. **Protect** (`extract/markdown.py` + `translate/glossary.py`): swap spans/terms
   for sentinels, recording `sentinel → literal` restorations.
2. **Translate** (`translate/*`): the prompt instructs the model to copy
   sentinels verbatim; the batcher packs whole segments; the validator checks the
   sentinels came back intact.
3. **Restore** (`apply_restorations`): put the literals back after translation.

## How the stages decouple (the seams)

- Stage 1 emits `PageLayout`s and knows nothing about Markdown or translation.
- Stage 2 turns raw spans into `Block`s; it is the only stage that understands
  PDF geometry and GFM structure.
- Stage 3 only ever sees `Segment`s (protected text) — it has no PDF or Markdown
  knowledge, which is why it can be swapped between Ollama and vLLM with a flag.
- Stage 4 only sees translated `Block`s; the PDF engine sits behind a small
  pluggable interface so a second backend (typst) drops in without touching the
  rest.
