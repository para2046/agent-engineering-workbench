"""A small, strict client for model-based judging and analysis.

Everything that asks a model for a *structured opinion* goes through here, so
there is exactly one place that handles the awkward parts: getting JSON out of
a text model, refusing to invent a verdict when parsing fails, and recording
which prompt version produced the answer.

The hard rule: **a judge that cannot be parsed returns UNKNOWN.** It never
falls back to a default score. A silently-defaulted judge is worse than no
judge, because it looks like a measurement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..providers.base import ModelProvider, ProviderError
from ..types import Message, Usage


class JudgeError(RuntimeError):
    """The judge could not produce a usable structured answer."""


@dataclass
class JudgeReply:
    data: dict[str, Any]
    raw_text: str = ""
    usage: Usage = field(default_factory=Usage)
    prompt_id: str = ""
    model: str = ""


class JudgeClient:
    """Wraps any ModelProvider for structured, tool-free calls."""

    def __init__(self, provider: ModelProvider, max_evidence_chars: int = 6000):
        self.provider = provider
        self.max_evidence_chars = max_evidence_chars

    @property
    def model(self) -> str:
        return getattr(self.provider, "model", "")

    def ask_json(self, system: str, user: str, prompt_id: str = "") -> JudgeReply:
        """Call the model and require a JSON object back.

        Raises JudgeError rather than guessing. Callers convert that into an
        UNKNOWN verdict with the error attached as evidence.
        """
        try:
            resp = self.provider.generate(system, [Message(role="user", content=user)], [])
        except ProviderError as exc:
            raise JudgeError(f"judge provider failed: {exc}") from exc

        data = _extract_json(resp.text)
        if data is None:
            preview = (resp.text or "").strip()[:300]
            raise JudgeError(f"judge did not return parseable JSON; got: {preview!r}")
        return JudgeReply(data=data, raw_text=resp.text, usage=resp.usage,
                          prompt_id=prompt_id, model=self.model)

    def truncate_evidence(self, text: str) -> str:
        if len(text) <= self.max_evidence_chars:
            return text
        head = self.max_evidence_chars * 2 // 3
        tail = self.max_evidence_chars - head
        return f"{text[:head]}\n... [evidence elided] ...\n{text[-tail:]}"


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pull one JSON object out of a model reply.

    Tries the whole string, then a ```json fence, then the outermost braces.
    Deliberately does not repair malformed JSON -- a judge that needs its
    output repaired is a judge whose output should not be trusted.
    """
    if not text:
        return None
    candidates = [text.strip()]

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        candidates.append(fence.group(1))

    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])

    for c in candidates:
        try:
            parsed = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
