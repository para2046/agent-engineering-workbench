"""A real model through the Claude Code CLI -- no API key required.

Anyone with Claude Code installed is already authenticated to a real model.
This provider shells out to ``claude -p`` (headless print mode), which turns
that session into a workbench provider: real reasoning driving the agent seat,
the judge seat, and the optimizer, with real token usage reported back.

Two calling modes, decided by whether tools were offered:

* **Tool mode** (the agent seat): the CLI is text-in/text-out, so tool use is
  carried in-band -- the prompt shows the tool schemas and asks for exactly one
  JSON action, ``{"tool": ..., "arguments": {...}}`` or ``{"text": ...}``. The
  same protocol the session provider uses, so a transcript from either looks
  identical downstream.
* **Text mode** (judge / optimizer seats): plain prompt, raw reply. JudgeClient
  does its own JSON parsing and its own refusing.

An unparseable tool-mode reply becomes a final answer rather than a crash --
the graders then judge the run on its merits, which is the correct consequence
for an agent that stopped speaking the protocol.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Optional

from ..types import Message, ModelResponse, ToolCall, ToolSchema, Usage
from .base import ProviderError, register_provider

DEFAULT_MODEL = "sonnet"

ACTION_PROTOCOL = """\
Reply with EXACTLY ONE JSON object and nothing else. Either an action:
  {"tool": "<tool name>", "arguments": { ... }}
or, when you are finished, your final report:
  {"text": "<what you did and the evidence it worked>"}
Do not use any tools of your own. Do not wrap the JSON in markdown fences."""


class ClaudeCLIProvider:
    name = "claude-cli"

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 180,
                 cli_path: Optional[str] = None, **_):
        path = cli_path or shutil.which("claude")
        if not path:
            raise ProviderError(
                "the `claude` CLI is not on PATH -- install Claude Code, or use "
                "--provider claude with an ANTHROPIC_API_KEY instead"
            )
        self._cli = path
        self.model = model
        self.timeout = timeout
        self._counter = 0

    # -- provider interface ------------------------------------------------
    def generate(self, system: str, messages: list[Message],
                 tools: list[ToolSchema]) -> ModelResponse:
        prompt = self._build_prompt(system, messages, tools)
        text, usage = self._call(prompt)

        if not tools:
            return ModelResponse(text=text, usage=usage, stop_reason="end_turn")

        action = _extract_json(text)
        if action and action.get("tool"):
            self._counter += 1
            return ModelResponse(
                text=str(action.get("text") or ""),
                tool_calls=[ToolCall(id=f"cli_{self._counter}",
                                     name=str(action["tool"]),
                                     arguments=dict(action.get("arguments") or {}))],
                usage=usage, stop_reason="tool_use",
            )
        if action and "text" in action:
            return ModelResponse(text=str(action["text"]), usage=usage,
                                 stop_reason="end_turn")
        # Off-protocol reply: surface it as the final answer and let the
        # graders judge the consequences. Silently retrying would hide a
        # real model behaviour the workbench exists to observe.
        return ModelResponse(text=text, usage=usage, stop_reason="end_turn")

    # -- internals -----------------------------------------------------------
    def _build_prompt(self, system: str, messages: list[Message],
                      tools: list[ToolSchema]) -> str:
        parts = [system.strip()]
        if tools:
            parts.append("TOOLS AVAILABLE (call via the JSON protocol below):")
            parts.append(json.dumps(
                [{"name": t.name, "description": t.description,
                  "parameters": t.parameters} for t in tools],
                ensure_ascii=False))
        parts.append("CONVERSATION SO FAR:")
        for m in messages:
            if m.role == "user":
                parts.append(f"[user]\n{m.content}")
            elif m.role == "assistant":
                for c in m.tool_calls:
                    parts.append(f"[you called] {c.name}({json.dumps(c.arguments, ensure_ascii=False)})")
                if m.content:
                    parts.append(f"[you said]\n{m.content}")
            elif m.role == "tool":
                parts.append(f"[observation from {m.name}]\n{m.content}")
        if tools:
            parts.append(ACTION_PROTOCOL)
        return "\n\n".join(parts)

    def _call(self, prompt: str) -> tuple[str, Usage]:
        cmd = [self._cli, "-p", "--model", self.model, "--output-format", "json"]
        if self._cli.lower().endswith((".cmd", ".bat")):
            cmd = ["cmd", "/c", *cmd]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True,
                                  text=True, encoding="utf-8", timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"claude CLI timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise ProviderError(f"could not launch the claude CLI: {exc}") from exc
        if proc.returncode != 0:
            raise ProviderError(
                f"claude CLI exited {proc.returncode}: {(proc.stderr or proc.stdout)[:400]}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"claude CLI returned unparseable output: {proc.stdout[:200]!r}"
            ) from exc
        if payload.get("is_error"):
            raise ProviderError(f"claude CLI reported an error: {str(payload)[:400]}")

        raw_usage = payload.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens") or 0)
            + int(raw_usage.get("cache_read_input_tokens") or 0)
            + int(raw_usage.get("cache_creation_input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
        )
        return str(payload.get("result") or "").strip(), usage


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    from ..judge.client import _extract_json as extract
    return extract(text)


register_provider("claude-cli", ClaudeCLIProvider)
