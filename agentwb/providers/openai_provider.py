"""OpenAI / ChatGPT adapter.

Exists for the same reason the Claude adapter does: the architecture is
provider-agnostic, and the only way to keep that honest is to have more than
one provider actually implemented. The spec's multi-agent phase pairs a Claude
implementation agent with an OpenAI research/critique agent, and neither role
is allowed to be hard-coded.

The SDK is optional. Importing this module without ``openai`` installed is
fine; the failure surfaces only when you try to build the provider, with a
message saying what to install.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..types import Message, ModelResponse, ToolCall, ToolSchema, Usage
from .base import ProviderError, register_provider

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        **_,
    ):
        try:
            import openai  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError(
                "the openai SDK is not installed -- run `pip install openai` "
                "or use `--provider mock`"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError(
                "OPENAI_API_KEY is not set -- export it, or use `--provider mock`"
            )
        self._client = openai.OpenAI(api_key=key, base_url=base_url or None)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # -- translation ------------------------------------------------------
    @staticmethod
    def _tools_payload(tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages_payload(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate to the chat-completions shape.

        Two differences from Anthropic worth noting: the system prompt is a
        message rather than a top-level field, and tool results are their own
        role keyed by ``tool_call_id`` instead of a content block.
        """
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.content,
                })
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name,
                                         "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ]
                out.append(entry)
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def generate(self, system: str, messages: list[Message], tools: list[ToolSchema]) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._messages_payload(system, messages),
        }
        if tools:
            kwargs["tools"] = self._tools_payload(tools)

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderError(f"openai request failed: {exc}") from exc

        choice = resp.choices[0]
        msg = choice.message

        calls: list[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                # arguments arrive as a JSON *string*; a model that emits
                # malformed JSON becomes a BAD_ARGUMENTS tool error rather
                # than crashing the run.
                arguments=_safe_json(tc.function.arguments),
            ))

        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
        )
        return ModelResponse(
            text=(msg.content or "").strip(),
            tool_calls=calls,
            usage=usage,
            stop_reason=choice.finish_reason,
        )


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"__malformed_arguments__": raw}
    return parsed if isinstance(parsed, dict) else {"__non_object_arguments__": parsed}


register_provider("openai", OpenAIProvider)
register_provider("chatgpt", OpenAIProvider)
