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


# Which module registers which key. Kept as data so that adding an adapter is
# one line here, and so a key can never point at a module that does not exist
# without that being obvious.
_ADAPTER_MODULES: dict[str, str] = {
    "mock": "mock",
    "scripted": "mock",
    "claude": "claude",
    "anthropic": "claude",
    "openai": "openai_provider",
    "chatgpt": "openai_provider",
}


def build_provider(key: str, **kwargs) -> ModelProvider:
    """Construct a provider by key.

    Adapters are imported lazily so a missing optional SDK never breaks the
    rest of the CLI. Any import failure becomes a ProviderError -- a raw
    ImportError leaking out of here once made `--provider openai` look like a
    crash rather than a missing adapter.
    """
    if key not in _REGISTRY:
        module = _ADAPTER_MODULES.get(key)
        if module is not None:
            try:
                __import__(f"{__package__}.{module}", fromlist=["*"])
            except ProviderError:
                raise
            except ImportError as exc:
                raise ProviderError(
                    f"provider {key!r} could not be loaded: {exc}"
                ) from exc

    if key not in _REGISTRY:
        known = ", ".join(sorted(set(_ADAPTER_MODULES))) or "none"
        raise ProviderError(f"unknown provider {key!r} (available: {known})")
    return _REGISTRY[key](**kwargs)


def available_providers() -> list[str]:
    """Every provider key that can be requested, whether or not its SDK is
    installed. A key whose SDK is missing still resolves -- it raises a
    ProviderError telling you what to install, which is more useful than
    silently vanishing from the list."""
    for mod in set(_ADAPTER_MODULES.values()):
        try:
            __import__(f"{__package__}.{mod}", fromlist=["*"])
        except Exception:
            pass
    return sorted(set(_ADAPTER_MODULES) | set(_REGISTRY))
