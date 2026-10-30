"""Data contracts shared across all pipeline stages.

Every stage communicates through the pydantic models defined here, so the
ingest, extract, translate, and output stages stay decoupled and independently
testable. OCR/VLM blocks (Phase B) and native-text blocks both normalize to the
same :data:`Block` union, so everything downstream treats them uniformly.

This module also owns the **sentinel contract** (``make_sentinel`` /
``SENTINEL_RE`` / ``apply_restorations``) that the markdown-protection,
glossary, validator, and prompt layers all agree on — keeping those layers
decoupled while guaranteeing they speak the same placeholder language.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sentinel / placeholder contract (shared across markdown, glossary, validator,
# prompts). A sentinel is an opaque token that replaces a span the model must
# NOT translate (math, code, URLs) or a glossary term that must map to a fixed
# target word. The tokens use "white square brackets" (U+27E6/U+27E7) which are
# virtually never emitted by translation models, plus an uppercase tag and an
# integer index so they are both unique and trivially regex-detectable.
# ---------------------------------------------------------------------------

SENTINEL_OPEN = "⟦"  # ⟦
SENTINEL_CLOSE = "⟧"  # ⟧
SENTINEL_RE = re.compile(rf"{SENTINEL_OPEN}[A-Z]{{1,5}}\d+{SENTINEL_CLOSE}")


def make_sentinel(tag: str, index: int) -> str:
    """Build a sentinel token such as ``⟦MATH0⟧`` for the given tag/index."""
    return f"{SENTINEL_OPEN}{tag.upper()}{index}{SENTINEL_CLOSE}"


def find_sentinels(text: str) -> list[str]:
    """Return every sentinel token present in ``text`` (in order)."""
    return SENTINEL_RE.findall(text)


def apply_restorations(text: str, restorations: dict[str, str]) -> str:
    """Replace each sentinel token in ``text`` with its original/target value."""
    for token, original in restorations.items():
        text = text.replace(token, original)
    return text


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used only for batch packing."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class ProtectedText(BaseModel):
    """Text with non-translatable spans swapped for sentinel tokens.

    ``restorations`` maps each sentinel token back to the literal string that
    must appear in the final output (the original span, or — for glossary
    terms — the fixed *target-language* term).
    """

    text: str
    restorations: dict[str, str] = Field(default_factory=dict)

    def restore(self, translated: str) -> str:
        """Restore sentinels in a translated string back to literals."""
        return apply_restorations(translated, self.restorations)


# ---------------------------------------------------------------------------
# Geometry & raw text primitives (Stage 1 — ingest/detect)
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned bounding box in PDF point coordinates (y grows downward)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> "BBox":
        return cls(x0=t[0], y0=t[1], x1=t[2], y1=t[3])

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x1 <= other.x0
            or other.x1 <= self.x0
            or self.y1 <= other.y0
            or other.y1 <= self.y0
        )

    def intersection_area(self, other: "BBox") -> float:
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy


# fitz span flag bits (PyMuPDF): these are stable across the 1.x/2.x line.
_FLAG_ITALIC = 1 << 1
_FLAG_MONOSPACE = 1 << 3
_FLAG_BOLD = 1 << 4


class TextSpan(BaseModel):
    """A single run of same-style text, as produced by ``get_text('dict')``."""

    text: str
    bbox: BBox
    font: str = ""
    size: float = 0.0
    flags: int = 0
    color: int = 0

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & _FLAG_BOLD) or "bold" in self.font.lower()

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & _FLAG_ITALIC) or any(
            k in self.font.lower() for k in ("italic", "oblique")
        )

    @property
    def is_monospace(self) -> bool:
        return bool(self.flags & _FLAG_MONOSPACE) or any(
            k in self.font.lower() for k in ("mono", "courier", "consol")
        )


class TextLine(BaseModel):
    """A line of spans sharing a baseline."""

    bbox: BBox
    spans: list[TextSpan] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


class RawTextBlock(BaseModel):
    """A fitz text block (group of lines) before logical classification."""

    bbox: BBox
    lines: list[TextLine] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class ImageRegion(BaseModel):
    """A raster image placed on a page (candidate for OCR/VLM in Phase B)."""

    bbox: BBox
    xref: int = 0
    name: Optional[str] = None


class PageLayout(BaseModel):
    """Everything the detector found on one page."""

    number: int  # 1-based page number
    width: float
    height: float
    text_blocks: list[RawTextBlock] = Field(default_factory=list)
    image_regions: list[ImageRegion] = Field(default_factory=list)
    is_hybrid: bool = False  # native spans AND an image region overlap

    @property
    def has_native_text(self) -> bool:
        return any(b.lines for b in self.text_blocks)


# ---------------------------------------------------------------------------
# Logical blocks (Stage 2 — extract/normalize). Discriminated union on `kind`.
# ---------------------------------------------------------------------------

BlockKind = Literal[
    "heading",
    "paragraph",
    "list",
    "table",
    "code",
    "math",
    "figure",
    "caption",
]


class BlockBase(BaseModel):
    """Common fields + translation interface for every logical block."""

    page: int = 1
    order: int = 0
    bbox: Optional[BBox] = None

    def translatable_texts(self) -> list[str]:
        """Return the ordered list of text fragments to translate."""
        return []

    def with_translations(self, texts: list[str]) -> "BlockBase":
        """Return a copy with ``texts`` substituted (same count/shape)."""
        return self


class HeadingBlock(BlockBase):
    kind: Literal["heading"] = "heading"
    level: int = 1
    content: str = ""

    def translatable_texts(self) -> list[str]:
        return [self.content]

    def with_translations(self, texts: list[str]) -> "HeadingBlock":
        return self.model_copy(update={"content": texts[0]})


class ParagraphBlock(BlockBase):
    kind: Literal["paragraph"] = "paragraph"
    content: str = ""

    def translatable_texts(self) -> list[str]:
        return [self.content]

    def with_translations(self, texts: list[str]) -> "ParagraphBlock":
        return self.model_copy(update={"content": texts[0]})


class ListBlock(BlockBase):
    kind: Literal["list"] = "list"
    ordered: bool = False
    items: list[str] = Field(default_factory=list)

    def translatable_texts(self) -> list[str]:
        return list(self.items)

    def with_translations(self, texts: list[str]) -> "ListBlock":
        return self.model_copy(update={"items": list(texts)})


class TableBlock(BlockBase):
    kind: Literal["table"] = "table"
    rows: list[list[str]] = Field(default_factory=list)
    has_header: bool = True

    def translatable_texts(self) -> list[str]:
        return [cell for row in self.rows for cell in row]

    def with_translations(self, texts: list[str]) -> "TableBlock":
        it = iter(texts)
        rebuilt = [[next(it) for _ in row] for row in self.rows]
        return self.model_copy(update={"rows": rebuilt})


class CodeBlock(BlockBase):
    kind: Literal["code"] = "code"
    content: str = ""
    language: Optional[str] = None
    # code is never translated -> inherits empty translatable_texts()


class MathBlock(BlockBase):
    kind: Literal["math"] = "math"
    content: str = ""
    display: bool = True  # True for $$...$$ block math, False for inline
    # math is never translated -> inherits empty translatable_texts()


class FigureBlock(BlockBase):
    kind: Literal["figure"] = "figure"
    asset_path: str = ""  # output-relative, e.g. "assets/fig_p1_0.png"
    caption: Optional[str] = None

    def translatable_texts(self) -> list[str]:
        return [self.caption] if self.caption else []

    def with_translations(self, texts: list[str]) -> "FigureBlock":
        if not texts:
            return self
        return self.model_copy(update={"caption": texts[0]})


class CaptionBlock(BlockBase):
    kind: Literal["caption"] = "caption"
    content: str = ""

    def translatable_texts(self) -> list[str]:
        return [self.content]

    def with_translations(self, texts: list[str]) -> "CaptionBlock":
        return self.model_copy(update={"content": texts[0]})


Block = Annotated[
    Union[
        HeadingBlock,
        ParagraphBlock,
        ListBlock,
        TableBlock,
        CodeBlock,
        MathBlock,
        FigureBlock,
        CaptionBlock,
    ],
    Field(discriminator="kind"),
]


class Asset(BaseModel):
    """A binary asset (figure) extracted from the PDF to be copied to output."""

    id: str
    source_path: str  # where it currently lives (temp dir)
    rel_path: str  # output-relative destination, matches FigureBlock.asset_path
    page: int = 1
    bbox: Optional[BBox] = None


class Document(BaseModel):
    """The whole PDF normalized to an ordered list of logical blocks."""

    source_path: str
    page_count: int
    source_lang: str = "auto"
    target_lang: str = ""
    blocks: list[Block] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Translation contracts (Stage 3 — translate)
# ---------------------------------------------------------------------------


class Segment(BaseModel):
    """One atomic unit of translatable text (never split further).

    ``text`` is already protected (sentinels substituted). ``restorations``
    carries the sentinel→literal map so the raw model output can be restored.
    ``origin`` records which (block index, field index) the segment came from so
    translated text can be re-assembled into the right block.
    """

    id: int
    text: str
    restorations: dict[str, str] = Field(default_factory=dict)
    block_index: int = -1
    field_index: int = 0

    @property
    def approx_tokens(self) -> int:
        return estimate_tokens(self.text)


class TranslationBatch(BaseModel):
    """A group of segments sent to the model in one request."""

    id: int
    segments: list[Segment] = Field(default_factory=list)

    @property
    def approx_tokens(self) -> int:
        return sum(s.approx_tokens for s in self.segments)


class TranslationResult(BaseModel):
    """Raw (still-protected) translations keyed by segment id, plus metrics."""

    batch_id: int
    translations: dict[int, str] = Field(default_factory=dict)
    eval_count: int = 0
    eval_duration_ns: int = 0

    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration_ns <= 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1e9)


# ---------------------------------------------------------------------------
# Validation (anti-hallucination) contracts
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    segment_id: int
    kind: str  # e.g. "missing-sentinel", "fence-imbalance", "empty-translation"
    detail: str = ""


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def extend(self, more: "ValidationReport") -> None:
        self.issues.extend(more.issues)


# ---------------------------------------------------------------------------
# Wire schemas: Ollama /api/chat (NDJSON streaming) and OpenAI-compatible /v1
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    options: dict = Field(default_factory=dict)


class OllamaChatChunk(BaseModel):
    """One NDJSON line from Ollama's /api/chat stream."""

    model: str = ""
    message: Optional[ChatMessage] = None
    done: bool = False
    eval_count: int = 0
    eval_duration: int = 0  # nanoseconds


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible request for vLLM's /v1/chat/completions."""

    model: str
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float = 0.0
