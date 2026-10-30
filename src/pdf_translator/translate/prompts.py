"""Prompt templates and the batch wire-format.

Segments are sent to the model wrapped in ``⟦SEG:n⟧`` markers (note the colon —
these deliberately do NOT match the content ``SENTINEL_RE``, so they never clash
with protected spans). The model is told to echo each marker and translate only
the text between markers, which lets us parse the reply back into a
``segment-id → translation`` map and detect dropped/merged segments.
"""

from __future__ import annotations

import re

from ..models import TranslationBatch

SEG_MARKER_RE = re.compile(r"⟦SEG:(\d+)⟧")


def seg_marker(index: int) -> str:
    return f"⟦SEG:{index}⟧"


def build_system_prompt(source_lang: str, target_lang: str, strict: bool = False) -> str:
    src = "the source language (auto-detect it)" if source_lang == "auto" else source_lang
    rules = [
        f"You are a professional translator. Translate from {src} to {target_lang}.",
        "Rules:",
        "1. Translate ONLY the natural-language text.",
        "2. Keep every ⟦...⟧ token EXACTLY as-is — never translate, reorder, drop, or alter them.",
        "3. Preserve all Markdown and inline formatting (**bold**, *italic*, `code`, lists, links).",
        "4. Do NOT add notes, explanations, or extra content.",
        "5. Return EACH segment prefixed by its own ⟦SEG:n⟧ marker, in the same order, "
        "with the same number of segments you received.",
        "6. Put the translation on the line(s) after its marker.",
    ]
    if strict:
        rules.append(
            "7. CRITICAL: output the markers verbatim and exactly as many segments as the input. "
            "Do not merge or split segments. Do not output anything before the first marker."
        )
    return "\n".join(rules)


def build_user_prompt(batch: TranslationBatch) -> str:
    parts = [
        "Translate each segment below. Echo its ⟦SEG:n⟧ marker, then its translation.",
        "",
    ]
    for seg in batch.segments:
        parts.append(seg_marker(seg.id))
        parts.append(seg.text)
        parts.append("")
    return "\n".join(parts)


def parse_response(text: str, expected_ids: list[int]) -> dict[int, str]:
    """Parse a model reply into ``{segment_id: translation}``.

    Splits on ``⟦SEG:n⟧`` markers; text before the first marker is ignored. Only
    ids that were expected are kept. Missing ids simply won't appear (the
    validator flags them so the pipeline can retry).
    """
    matches = list(SEG_MARKER_RE.finditer(text))
    if not matches:
        return {}
    wanted = set(expected_ids)
    result: dict[int, str] = {}
    for i, match in enumerate(matches):
        seg_id = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if seg_id in wanted:
            result[seg_id] = text[start:end].strip("\n").strip()
    return result
