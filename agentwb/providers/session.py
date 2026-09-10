"""A provider backed by a human-or-assistant answering turn by turn.

Every other provider here is either a network client or a fixture. This one is
neither: it lets whoever is at the terminal -- a person, or an assistant like
Claude Code driving the CLI -- act as the model for a run.

Why it exists: fixtures flatter the parts of an ACI that matter most. A
`ScriptedProvider` never misreads a tool description, never fumbles an argument
schema, never has to decide what to do with a truncated observation. Those are
precisely the failures this workbench is supposed to surface, and a suite that
is green against fixtures has not tested them once.

**How it works: replay-and-extend.** Answers accumulate in a JSON file. Each
pass replays the answers already given, then stops at the first unanswered
call, writes the exact prompt to `pending.json`, and raises `NeedsAnswer`. You
read the prompt, append your reply, and run again. The run advances one turn
per pass.

The workspace is rebuilt from the task definition on every pass, so replay is
faithful: the same answers against the same starting environment produce the
same state. That is the same property the eval harness relies on everywhere
else, used here to make a stateless loop resumable.

**What this is not.** A run driven by the same assistant that wrote the code is
not an independent evaluation, and nothing here pretends otherwise. Its value
is that the *graders* remain deterministic: they re-execute the test suite
against the environment afterwards, and no amount of confident narration in the
transcript changes what they find. The agent's answers are mine; the verdict is
not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..types import Message, ModelResponse, ToolCall, ToolSchema, Usage
from .base import register_provider


class NeedsAnswer(Exception):
    """Raised when the run reaches a turn nobody has answered yet.

    Carries the path of the written prompt so the driver can point at it.
    """

    def __init__(self, index: int, pending_path: Path):
        super().__init__(
            f"turn {index} needs an answer -- see {pending_path}"
        )
        self.index = index
        self.pending_path = pending_path


@dataclass
class SessionPaths:
    root: Path

    @property
    def answers(self) -> Path:
        return self.root / "answers.json"

    @property
    def pending(self) -> Path:
        return self.root / "pending.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


class SessionProvider:
    """Answers come from a file; unanswered turns stop the run."""

    name = "session"

    def __init__(self, root: str = "data/session", model: str = "claude-code-session", **_):
        self.model = model
        self.paths = SessionPaths(Path(root))
        self.paths.ensure()
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    # -- answers ---------------------------------------------------------
    def load_answers(self) -> list[dict[str, Any]]:
        if not self.paths.answers.is_file():
            return []
        try:
            data = json.loads(self.paths.answers.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.paths.answers} is not valid JSON: {exc}") from exc
        return data if isinstance(data, list) else []

    def generate(self, system: str, messages: list[Message], tools: list[ToolSchema]) -> ModelResponse:
        answers = self.load_answers()
        index, self._index = self._index, self._index + 1

        if index < len(answers):
            return _to_response(answers[index])

        # Nothing answered this turn yet. Write down exactly what the model
        # would have seen -- system prompt, full history, tool schemas -- so the
        # decision is made on the real input rather than a summary of it.
        self._write_pending(index, system, messages, tools)
        raise NeedsAnswer(index, self.paths.pending)

    def _write_pending(self, index: int, system: str, messages: list[Message],
                       tools: list[ToolSchema]) -> None:
        payload = {
            "turn": index,
            "instructions": (
                "Decide this turn, then append your reply to answers.json and re-run. "
                "Reply with either {\"tool\": \"<name>\", \"arguments\": {...}} to act, "
                "or {\"text\": \"...\"} to finish."
            ),
            "system": system,
            "tools": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in tools
            ],
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool": m.name,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments} for c in m.tool_calls
                    ] or None,
                }
                for m in messages
            ],
        }
        self.paths.ensure()
        self.paths.pending.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _to_response(answer: dict[str, Any]) -> ModelResponse:
    """Turn one recorded answer into a provider response.

    Two shapes: a tool call, or final text. An answer that is neither is a
    mistake worth failing on rather than silently treating as "done" -- a
    dropped turn would look like the agent choosing to stop.
    """
    if not isinstance(answer, dict):
        raise ValueError(f"each answer must be an object, got {type(answer).__name__}")

    if answer.get("tool"):
        return ModelResponse(
            text=str(answer.get("text") or ""),
            tool_calls=[ToolCall(
                id=f"call_{answer.get('id') or answer['tool']}",
                name=str(answer["tool"]),
                arguments=dict(answer.get("arguments") or {}),
            )],
            usage=Usage(**(answer.get("usage") or {})),
            stop_reason="tool_use",
        )

    if "text" in answer:
        return ModelResponse(
            text=str(answer["text"]),
            usage=Usage(**(answer.get("usage") or {})),
            stop_reason="end_turn",
        )

    raise ValueError(
        f"answer must contain either 'tool' or 'text'; got keys {sorted(answer)}"
    )


register_provider("session", SessionProvider)
