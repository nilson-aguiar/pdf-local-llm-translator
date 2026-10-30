"""Custom exception hierarchy.

Every stage raises a subclass of :class:`PDFTranslatorError`, so the CLI can
present a single clean error surface and callers can catch per-stage failures.
"""

from __future__ import annotations


class PDFTranslatorError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(PDFTranslatorError):
    """Invalid configuration, flags, or glossary."""


class IngestError(PDFTranslatorError):
    """Stage 1 — failed to open/read/detect the PDF."""


class ExtractError(PDFTranslatorError):
    """Stage 2 — failed to normalize spans into logical blocks."""


class TranslateError(PDFTranslatorError):
    """Stage 3 — failed to translate (transport, protocol, or exhausted retries)."""


class ValidationDriftError(TranslateError):
    """Stage 3 — translation structure drifted from the source and could not be
    recovered after retries (block/segment count, table/list parity, sentinels)."""


class OutputError(PDFTranslatorError):
    """Stage 4 — failed to write markdown or compile the PDF."""


class BackendNotAvailableError(PDFTranslatorError):
    """A requested backend is not wired up in this phase (clear 'not-yet' guard).

    Used for Phase B features (tesseract/VLM OCR, typst engine) so the full CLI
    flag surface is accepted while unsupported paths fail loudly and helpfully.
    """
