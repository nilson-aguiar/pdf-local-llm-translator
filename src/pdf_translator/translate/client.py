"""Stage 3 — async translation transport.

Streams the local model's chat endpoint (Ollama ``/api/chat`` NDJSON, or vLLM's
OpenAI-compatible ``/v1/chat/completions`` SSE), caps concurrency with an
``asyncio.Semaphore`` for GPU/VRAM safety, retries with backoff, and reports
live token deltas. The client only ever deals with :class:`TranslationBatch` /
:class:`TranslationResult` — it has no PDF or Markdown knowledge.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Optional

import httpx

from ..config import Settings
from ..exceptions import TranslateError
from ..logging import get_logger
from ..models import ChatMessage, TranslationBatch, TranslationResult
from . import prompts

logger = get_logger(__name__)

OnDelta = Callable[[str], None]


class TranslationClient:
    """Async client for a local chat model, shared across batches."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._semaphore = asyncio.Semaphore(max(1, settings.concurrency))
        self._client = httpx.AsyncClient(timeout=settings.request_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "TranslationClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _payload(self, messages: list[ChatMessage]) -> dict:
        msgs = [m.model_dump() for m in messages]
        if self.settings.api_mode == "vllm":
            return {
                "model": self.settings.model,
                "messages": msgs,
                "stream": True,
                "temperature": 0.0,
            }
        return {
            "model": self.settings.model,
            "messages": msgs,
            "stream": True,
            "options": {"temperature": 0.0},
        }

    async def translate_batch(
        self,
        batch: TranslationBatch,
        source_lang: str,
        target_lang: str,
        strict: bool = False,
        on_delta: Optional[OnDelta] = None,
    ) -> TranslationResult:
        """Translate one batch, returning raw (still-protected) translations."""
        messages = [
            ChatMessage(
                role="system",
                content=prompts.build_system_prompt(source_lang, target_lang, strict=strict),
            ),
            ChatMessage(role="user", content=prompts.build_user_prompt(batch)),
        ]
        payload = self._payload(messages)
        expected_ids = [s.id for s in batch.segments]

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                async with self._semaphore:
                    content, eval_count, eval_duration = await self._stream(payload, on_delta)
                translations = prompts.parse_response(content, expected_ids)
                if not translations:
                    raise TranslateError("model returned no parseable segments")
                return TranslationResult(
                    batch_id=batch.id,
                    translations=translations,
                    eval_count=eval_count,
                    eval_duration_ns=eval_duration,
                )
            except (httpx.HTTPError, TranslateError, json.JSONDecodeError) as exc:
                last_exc = exc
                logger.warning(
                    "batch %s attempt %s/%s failed: %s",
                    batch.id,
                    attempt,
                    self.settings.max_retries,
                    exc,
                )
                if attempt < self.settings.max_retries:
                    await asyncio.sleep(min(2.0 ** attempt, 10.0))
        raise TranslateError(
            f"batch {batch.id} failed after {self.settings.max_retries} attempts: {last_exc}"
        ) from last_exc

    async def _stream(
        self, payload: dict, on_delta: Optional[OnDelta]
    ) -> tuple[str, int, int]:
        """POST and consume the stream; return (content, eval_count, eval_ns)."""
        url = self.settings.chat_endpoint
        chunks: list[str] = []
        eval_count = 0
        eval_duration = 0
        vllm = self.settings.api_mode == "vllm"

        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                if vllm:
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        piece = delta.get("content") or ""
                        if piece:
                            chunks.append(piece)
                            if on_delta:
                                on_delta(piece)
                    usage = obj.get("usage") or {}
                    if usage:
                        eval_count = int(usage.get("completion_tokens", eval_count))
                else:
                    obj = json.loads(line)
                    msg = obj.get("message") or {}
                    piece = msg.get("content") or ""
                    if piece:
                        chunks.append(piece)
                        if on_delta:
                            on_delta(piece)
                    if obj.get("done"):
                        eval_count = int(obj.get("eval_count", eval_count))
                        eval_duration = int(obj.get("eval_duration", eval_duration))

        return "".join(chunks), eval_count, eval_duration
