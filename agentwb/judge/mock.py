"""Deterministic judge fixtures.

Model graders are the easiest part of an eval system to fool yourself with, so
the tests need a judge whose replies are exactly known -- including the ugly
ones: prose instead of JSON, a score with no evidence, a flat refusal.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from ..types import Message, ModelResponse, ToolSchema, Usage


class FakeJudgeProvider:
    """Replays canned replies. Each item may be a dict (encoded as JSON) or a
    raw string (used verbatim, so malformed replies can be tested)."""

    name = "fake-judge"

    def __init__(self, replies: Optional[Iterable[Any]] = None, model: str = "fake-judge-v1"):
        self.model = model
        self._replies = list(replies or [])
        self._calls = 0
        self.prompts: list[str] = []

    def reset(self) -> None:
        self._calls = 0
        self.prompts = []

    def generate(self, system: str, messages: list[Message], tools: list[ToolSchema]) -> ModelResponse:
        self.prompts.append("\n".join(m.content for m in messages if m.content))
        if self._calls < len(self._replies):
            item = self._replies[self._calls]
        else:
            item = self._replies[-1] if self._replies else "{}"
        self._calls += 1
        text = item if isinstance(item, str) else json.dumps(item)
        return ModelResponse(text=text, usage=Usage(input_tokens=10, output_tokens=10),
                             stop_reason="end_turn")


def verdict(score: float, verdict_str: str = "PASS", quote: str = "observed evidence",
            uncertainty: float = 0.1, reasoning: str = "because of the quoted evidence") -> dict:
    """Build a well-formed judge reply."""
    return {
        "score": score,
        "verdict": verdict_str,
        "uncertainty": uncertainty,
        "evidence": [{"quote": quote, "why": "supports the dimension being judged"}],
        "reasoning": reasoning,
    }
