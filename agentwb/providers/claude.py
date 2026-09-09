"""Anthropic / Claude adapter.

The SDK is an optional dependency. Importing this module without ``anthropic``
installed is fine -- the failure surfaces only when you actually try to build
the provider, with a message that says what to install.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..types import Message, ModelResponse, ToolCall, ToolSchema, Usage
from .base import ProviderError, register_provider

DEFAULT_MODEL = "claude-sonnet-5"


class ClaudeProvider:
    name = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **_,
    ):
        try:
            import anthropic  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError(
                "the anthropic SDK is not installed -- run `pip install anthropic` "
                "or use `--provider mock`"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set -- export it, or use `--provider mock`"
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # -- translation ------------------------------------------------------
    @staticmethod
    def _tools_payload(tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    @staticmethod
    def _messages_payload(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id or "",
                        "content": m.content,
                    }],
                })
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def generate(self, system: str, messages: list[Message], tools: list[ToolSchema]) -> ModelResponse:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=self._messages_payload(messages),
                tools=self._tools_payload(tools),
            )
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"anthropic request failed: {exc}") from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
        )
        return ModelResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            usage=usage,
            stop_reason=getattr(resp, "stop_reason", None),
        )


register_provider("claude", ClaudeProvider)
register_provider("anthropic", ClaudeProvider)
