"""Deterministic providers used for tests and for running the workbench with
no API key.

Two flavours:

``ScriptedProvider``
    Replays a fixed list of responses. Use it in unit tests when you want to
    assert on exactly how the runtime handles a given model behaviour
    (a bad tool argument, an early finish, a repeated action).

``RuleProvider``
    A tiny rule-driven "agent" that can actually solve the bundled example
    task by reading a file, editing it and running the tests. It exists so
    ``agentwb run`` produces a real trajectory -- real tool calls against a
    real workspace, real graders -- before anyone configures a model. It is a
    test fixture, not a model: it does no reasoning.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Iterable, Optional

from ..types import Message, ModelResponse, ToolCall, ToolSchema, Usage
from .base import register_provider


class ScriptedProvider:
    name = "mock"

    def __init__(self, script: Optional[Iterable[ModelResponse]] = None, model: str = "scripted-v1", **_):
        self.model = model
        self._script = list(script or [])
        self._calls = 0
        self.seen: list[list[Message]] = []

    def reset(self) -> None:
        """Rewind to the start of the script for a fresh trajectory."""
        self._calls = 0
        self.seen = []

    def generate(self, system: str, messages: list[Message], tools: list[ToolSchema]) -> ModelResponse:
        self.seen.append(list(messages))
        if self._calls < len(self._script):
            resp = self._script[self._calls]
        else:
            resp = ModelResponse(text="no further scripted responses", stop_reason="end_turn")
        self._calls += 1
        return resp


class RuleProvider:
    """Solves the bundled `fix_divide_bug` style task deterministically.

    Behaviour, in order:
      1. list the workspace
      2. run the tests to observe the failure
      3. read the offending file
      4. apply the fix it can derive from the task's own instructions
      5. re-run the tests
      6. finish, reporting what it observed

    Anything it does not recognise it finishes on immediately rather than
    flailing -- a mock that pretends to be clever makes for misleading evals.
    """

    name = "mock"

    def __init__(self, model: str = "rule-v1", **_):
        self.model = model
        self._counter = itertools.count(1)
        self._stage = 0

    def reset(self) -> None:
        """Rewind the stage machine so each run starts from step one."""
        self._stage = 0

    def _call(self, name: str, **args: Any) -> ModelResponse:
        return ModelResponse(
            text="",
            tool_calls=[ToolCall(id=f"call_{next(self._counter)}", name=name, arguments=args)],
            usage=Usage(input_tokens=0, output_tokens=0),
            stop_reason="tool_use",
        )

    def generate(self, system: str, messages: list[Message], tools: list[ToolSchema]) -> ModelResponse:
        names = {t.name for t in tools}
        transcript = "\n".join(m.content for m in messages if m.content)
        stage, self._stage = self._stage, self._stage + 1

        if stage == 0 and "list_files" in names:
            return self._call("list_files", path=".")
        if stage == 1 and "run_tests" in names:
            return self._call("run_tests")
        if stage == 2 and "read_file_region" in names:
            target = self._guess_target(transcript)
            return self._call("read_file_region", path=target, start=1, end=60)
        if stage == 3 and "edit_file" in names:
            target = self._guess_target(transcript)
            old, new = self._guess_edit(transcript)
            if old is None:
                return ModelResponse(
                    text="I could not determine a safe edit from the task description.",
                    stop_reason="end_turn",
                )
            return self._call("edit_file", path=target, old=old, new=new)
        if stage == 4 and "run_tests" in names:
            return self._call("run_tests")
        return ModelResponse(
            text="Ran the test suite, located the failing branch, applied the fix "
                 "described by the task, and re-ran the tests to verify.",
            stop_reason="end_turn",
        )

    # -- crude extraction helpers; deliberately literal -------------------
    @staticmethod
    def _guess_target(transcript: str) -> str:
        m = re.search(r"([A-Za-z0-9_./-]+\.py)", transcript)
        return m.group(1) if m else "calculator.py"

    @staticmethod
    def _guess_edit(transcript: str) -> tuple[Optional[str], Optional[str]]:
        """Pull a `replace X with Y` instruction out of the task prompt."""
        m = re.search(r"replace\s+`([^`]+)`\s+with\s+`([^`]+)`", transcript, re.I)
        if m:
            return m.group(1), m.group(2)
        return None, None


register_provider("mock", RuleProvider)
register_provider("scripted", ScriptedProvider)
