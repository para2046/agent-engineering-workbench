"""Provider interface.

A provider turns (system prompt, message history, tool schemas) into a
ModelResponse. That is the entire contract -- everything else in the workbench
is provider-agnostic, so adding a provider means writing one adapter and
registering it here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Message, ModelResponse, ToolSchema


@runtime_checkable
class ModelProvider(Protocol):
    """Structural interface every model adapter satisfies."""

    name: str
    model: str

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> ModelResponse:
        ...


def reset_provider(provider: object) -> None:
    """Clear any per-run state before a new trajectory begins.

    Real adapters are stateless across runs and do not implement ``reset``.
    Test fixtures that walk a fixed script do, and must be rewound -- otherwise
    one CLI invocation over several tasks leaks the first task's progress into
    the second.
    """
    hook = getattr(provider, "reset", None)
    if callable(hook):
        hook()


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response.

    The runtime converts this into ENVIRONMENT_ERROR rather than letting it
    surface as a task failure -- a provider outage is not the agent's fault.
    """


_REGISTRY: dict[str, type] = {}


def register_provider(key: str, cls: type) -> None:
    _REGISTRY[key] = cls


def build_provider(key: str, **kwargs) -> ModelProvider:
    """Construct a provider by key. Adapters are imported lazily so that a
    missing optional SDK never breaks the rest of the CLI."""
    if key not in _REGISTRY:
        # lazy import of built-ins
        if key == "mock":
            from . import mock  # noqa: F401
        elif key in ("claude", "anthropic"):
            from . import claude  # noqa: F401
        elif key in ("openai", "chatgpt"):
            from . import openai_provider  # noqa: F401
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none loaded"
        raise ProviderError(f"unknown provider {key!r} (registered: {known})")
    return _REGISTRY[key](**kwargs)


def available_providers() -> list[str]:
    for mod in ("mock", "claude", "openai_provider"):
        try:
            __import__(f"{__package__}.{mod}", fromlist=["*"])
        except Exception:
            pass
    return sorted(_REGISTRY)
