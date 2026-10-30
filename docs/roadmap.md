# Roadmap — Phase B and beyond

Phase A ships the native-text path and the interfaces Phase B fills in — no
rework. Both phases share one architecture (see [architecture.md](architecture.md)).

## Phase B — full fallbacks & second engine

### OCR / VLM ingest (`ingest/ocr.py`)
- **tesseract backend:** `pytesseract.image_to_data(..., output_type=DICT)` →
  words + bounding boxes + confidence, reassembled into `Block`s by line/para.
- **VLM backend:** render the page/region to PNG, base64-encode, send to the
  local VLM via Ollama `/api/chat` (`--vlm-model`); the prompt returns structured
  Markdown that is pydantic-validated back into `Block`s.
- **`--ocr-backend auto` routing:** native text where spans exist; fall back to
  OCR/VLM only for image regions.
- **Hybrid pages:** segment a page into native + image regions and merge the
  resulting blocks by bbox reading order.

Requires: `brew install tesseract tesseract-lang poppler` and a pulled VLM
(`ollama pull qwen2.5-vl`).

### Second PDF engine (`output/pdf_compiler.py`)
- **typst backend** behind the existing pluggable `--pdf-engine` interface, for
  native math rendering. weasyprint remains the default. Requires
  `brew install typst`.

### Hardening
- **Glossary enforcement:** stricter placeholder injection + a retry loop when
  post-verify finds a missing target term.
- **Validator retry loop:** on structural drift, re-translate the offending batch
  with a smaller batch size and a stricter prompt before flagging.
- **Text-expansion tuning (15–30% for PT/DE):** flexible table CSS
  (`table-layout:auto`) + generous line-height; validator warns on
  overflow-risk rows; `--expansion-factor` feeds the layout.

## Backlog (nice-to-haves)

- **Caching:** content-hash cache of segment translations to skip re-work.
- **Resume:** checkpoint per stage so an interrupted run restarts mid-document.
- **Batch directories:** translate every PDF in a folder.
- **Language auto-detect:** detect source language instead of relying on `--source-lang`.
- **Layout-faithful PDF:** optional reflow that better mirrors the original
  geometry.
- **Per-page progress persistence** and a machine-readable run report (JSON).

## Keeping docs in sync

Update [mvp.md](mvp.md) and this file as phases land: move a bullet from
"Roadmap" to "What ships" when it's implemented, and note any new known
limitations.
