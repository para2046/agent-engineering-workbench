"""Deterministic judge fixtures.

Model graders are the easiest part of an eval system to fool yourself with, so
the tests need a judge whose replies are exactly known -- including the ugly
ones: prose instead of JSON, a score with no evidence, a flat refusal.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from ..providers.base import register_provider
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


class RoleFixtureProvider:
    """Emits protocol artifacts so the multi-agent loop is demonstrable offline.

    The single-agent mock speaks tool calls; the protocol wants typed JSON
    messages, so without this `multi-agent --provider mock` produces nothing but
    violations and the rung cannot be seen working without an API key.

    It is a fixture, not a model. It follows a fixed script per seat, does no
    reasoning, and never inspects the task. Do not read its runs as evidence
    about agent behaviour -- they are evidence about the plumbing.
    """

    name = "mock-role"

    #: seat -> the artifacts it produces, in order
    SCRIPTS: dict[str, list[dict]] = {
        "researcher": [
            {
                "type": "HYPOTHESIS",
                "claim": "the regression is a resource exhaustion, not a slow query",
                "evidence": [{"source": "metrics.csv",
                              "observation": "pool_in_use 12 -> 100 while db_cpu_pct stays ~43"}],
                "confidence": 0.6,
                "unknowns": ["why an individual lookup is slow"],
                "recommended_action": "inspect what the deploy added per line item",
                "verification": {"metric": "pool_in_use",
                                 "expected_if_correct": "saturated at the ceiling",
                                 "expected_if_wrong": "well below the ceiling"},
            },
            {
                "type": "CRITIQUE",
                "claim": "the write-up must say what the data does not show",
                "evidence": [{"source": "incident_notes.md",
                              "observation": "no query timings are recorded anywhere"}],
                "confidence": 0.5,
                "unknowns": [],
                "recommended_action": "state the unknowns explicitly in findings.md",
            },
        ],
        "engineer": [
            {
                "type": "IMPLEMENTATION_PLAN",
                "claim": "write findings.md naming deploy c4f1a9 and the pool saturation",
                "evidence": [{"source": "incident_notes.md",
                              "observation": "c4f1a9 shipped at 14:08; latency rose at 14:10"}],
                "confidence": 0.6,
                "unknowns": ["whether the lookup is cached"],
                "recommended_action": "write the file, then re-read it",
            },
            {
                "type": "FINAL_REPORT",
                "claim": "findings.md records the cause, the evidence, and the unknowns",
                "evidence": [{"source": "shell", "observation": "findings.md written"}],
                "confidence": 0.7,
                "unknowns": ["why the lookup itself is slow"],
                "recommended_action": "review before closing the incident",
            },
        ],
    }

    def __init__(self, role: str = "researcher", model: str = "mock-role-v1", **_):
        self.model = model
        self.role = role
        self._calls = 0

    def reset(self) -> None:
        self._calls = 0

    def generate(self, system: str, messages: list[Message], tools: list[ToolSchema]) -> ModelResponse:
        # The seat is named in the system prompt; fall back to the configured role.
        role = self.role
        for name in self.SCRIPTS:
            if f"the {name} agent" in (system or ""):
                role = name
                break

        script = self.SCRIPTS.get(role, self.SCRIPTS["researcher"])
        if self._calls < len(script):
            payload = script[self._calls]
        else:
            # Nothing further to add. Saying so is the protocol-correct move;
            # restating an earlier position would be recorded as duplicate work.
            payload = {"type": "BLOCKER",
                       "claim": f"the {role} has nothing further to add",
                       "confidence": 0.3, "evidence": [], "unknowns": []}
        self._calls += 1
        return ModelResponse(text=json.dumps(payload),
                             usage=Usage(input_tokens=20, output_tokens=20),
                             stop_reason="end_turn")


register_provider("mock-role", RoleFixtureProvider)
