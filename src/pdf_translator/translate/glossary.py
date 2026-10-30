"""Glossary loading, term protection, and translation verification.

A :class:`Glossary` maps source-language terms to fixed target-language
equivalents.  Before text is sent to the translation model the glossary's
:meth:`~Glossary.protect` method replaces each source term with a
``⟦GLOSS{n}⟧`` sentinel pointing to the desired target term, composing
cleanly on top of the markdown protection layer (existing sentinels are never
disturbed).  After translation :meth:`~Glossary.verify` reports any target
terms that vanished from the output.
"""

from __future__ import annotations

import json
import re
from itertools import count
from pathlib import Path
from typing import Callable, Optional

from pdf_translator.exceptions import ConfigError
from pdf_translator.models import (
    SENTINEL_RE,
    ProtectedText,
    make_sentinel,
)

# ---------------------------------------------------------------------------
# Module-level counter for GLOSS sentinel indices (session-unique).
# ---------------------------------------------------------------------------

_gloss_counter: count[int] = count()


def _next_gloss_idx() -> int:
    """Return the next GLOSS sentinel index."""
    return next(_gloss_counter)


# ---------------------------------------------------------------------------
# Glossary class
# ---------------------------------------------------------------------------


class Glossary:
    """Source-to-target term map with sentinel-based protection.

    Attributes
    ----------
    terms:
        Mapping from source-language term to the required target-language term.

    Notes
    -----
    ``_protected_targets`` accumulates the target terms actually substituted
    during :meth:`protect` calls, giving :meth:`verify` an accurate set to
    check against instead of the full glossary.
    """

    def __init__(self, terms: dict[str, str]) -> None:
        self.terms: dict[str, str] = terms
        # Tracks target terms that were actually replaced (for verify).
        self._protected_targets: set[str] = set()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "Glossary":
        """Return a Glossary with no terms."""
        return cls({})

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def protect(self, pt: ProtectedText) -> ProtectedText:
        """Replace source terms in *pt* with ``GLOSS`` sentinels.

        Operates on the non-sentinel portions of ``pt.text`` only, so
        existing ``⟦...⟧`` tokens introduced by earlier protection passes
        (e.g. MATH, CODE, URL from :mod:`~pdf_translator.extract.markdown`)
        are never disturbed.

        Source terms are matched case-insensitively on word boundaries.
        Each matched occurrence is replaced with a fresh sentinel whose
        restoration value is the corresponding *target* term.

        Returns a new :class:`~pdf_translator.models.ProtectedText` that
        extends the incoming one (existing restorations are preserved).
        """
        # Start from a copy of the incoming restorations so we never mutate.
        restorations: dict[str, str] = dict(pt.restorations)
        text = pt.text

        # Split on existing sentinels so we never touch text inside them.
        # SENTINEL_RE.split returns alternating [plain, sentinel, plain, …]
        # SENTINEL_RE.findall returns the list of sentinel tokens.
        plain_parts = SENTINEL_RE.split(text)
        sentinel_parts = SENTINEL_RE.findall(text)

        new_plain: list[str] = []
        for part in plain_parts:
            for source, target in self.terms.items():
                pattern = re.compile(
                    r"\b" + re.escape(source) + r"\b",
                    re.IGNORECASE,
                )

                def _make_replacer(tgt: str) -> Callable[[re.Match[str]], str]:
                    def _replacer(_m: re.Match[str]) -> str:
                        sentinel = make_sentinel("GLOSS", _next_gloss_idx())
                        restorations[sentinel] = tgt
                        self._protected_targets.add(tgt)
                        return sentinel

                    return _replacer

                part = pattern.sub(_make_replacer(target), part)
            new_plain.append(part)

        # Reassemble: interleave plain parts with the original sentinels.
        result_parts: list[str] = []
        for i, plain in enumerate(new_plain):
            result_parts.append(plain)
            if i < len(sentinel_parts):
                result_parts.append(sentinel_parts[i])

        return ProtectedText(text="".join(result_parts), restorations=restorations)

    def verify(self, translated: str) -> list[str]:
        """Return target terms that were expected but are absent from *translated*.

        "Expected" means the source term was seen during a previous
        :meth:`protect` call (tracked in ``_protected_targets``).  This is a
        soft warning signal — the pipeline logs it but does not abort.
        """
        return [t for t in self._protected_targets if t not in translated]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_glossary(path: Path | str, target_lang: Optional[str] = None) -> Glossary:
    """Load a glossary from a JSON or YAML file.

    Accepted formats
    ----------------
    1. **Flat mapping** ``{"source term": "target term", …}``
    2. **Wrapped mapping** ``{"terms": {"source term": "target term", …}}``
    3. **Per-language mapping** ``{"<lang>": {"source term": "target term", …}}``
       — resolved via *target_lang* (required when this format is detected).

    Parameters
    ----------
    path:
        Path to a ``.json`` or ``.yaml``/``.yml`` file.
    target_lang:
        ISO language code used to select the right sub-dict in a per-language
        glossary.

    Raises
    ------
    pdf_translator.exceptions.ConfigError
        If the file cannot be read, is not valid JSON/YAML, or its structure
        does not match any recognised format.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read glossary file {path}: {exc}") from exc

    try:
        if suffix == ".json":
            data = json.loads(raw_text)
        elif suffix in {".yaml", ".yml"}:
            import yaml  # lazy import — optional dep at module level

            data = yaml.safe_load(raw_text)
        else:
            raise ConfigError(
                f"Unsupported glossary format '{suffix}' for {path}. "
                "Use .json, .yaml, or .yml."
            )
    except (json.JSONDecodeError, Exception) as exc:
        raise ConfigError(f"Failed to parse glossary {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Glossary {path} must be a mapping at the top level, got {type(data).__name__}."
        )

    # --- Detect format ---

    # Wrapped: {"terms": {...}}
    if "terms" in data and isinstance(data["terms"], dict):
        terms = data["terms"]

    # Per-language: all values are dicts (and no "terms" key with a dict value)
    elif all(isinstance(v, dict) for v in data.values()):
        if target_lang is None:
            raise ConfigError(
                f"Glossary {path} appears to be a per-language mapping "
                "(all values are dicts) but no target_lang was supplied."
            )
        if target_lang not in data:
            raise ConfigError(
                f"Glossary {path} has no entry for target_lang={target_lang!r}. "
                f"Available languages: {sorted(data)}"
            )
        terms = data[target_lang]

    # Flat mapping: {"source": "target", ...}
    elif all(isinstance(v, str) for v in data.values()):
        terms = data

    else:
        raise ConfigError(
            f"Glossary {path} has an unrecognised structure. "
            "Expected a flat mapping, a {{\"terms\": {{...}}}} wrapper, "
            "or a per-language dict."
        )

    if not all(isinstance(k, str) and isinstance(v, str) for k, v in terms.items()):
        raise ConfigError(f"Glossary {path}: all keys and values must be strings.")

    return Glossary(terms=terms)
