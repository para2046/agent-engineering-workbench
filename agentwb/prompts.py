"""Immutable prompt versioning (spec section 12).

Every prompt that shapes model behaviour carries a version id like
``judge_groundedness:v1``. Two rules make the registry useful rather than
decorative:

1. **Registration is immutable.** Re-registering an id with different text is an
   error, not an overwrite. Edit a prompt, bump the version.
2. **The id lands in the trajectory.** A stored run says exactly which prompt
   text produced it, so a score from last week is still interpretable today.

Optimized prompts (V3) will be registered here as new versions with a recorded
parent, never as edits to the parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class VersionedPrompt:
    name: str
    version: str
    text: str
    parent: Optional[str] = None      # e.g. "judge_groundedness:v1"
    optimizer: Optional[str] = None   # e.g. "GEPA" -- set when machine-generated
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}:{self.version}"


class PromptConflict(ValueError):
    """Raised when an existing prompt id would be redefined with new text."""


_REGISTRY: dict[str, VersionedPrompt] = {}


def register(prompt: VersionedPrompt) -> VersionedPrompt:
    existing = _REGISTRY.get(prompt.id)
    if existing is not None and existing.text != prompt.text:
        raise PromptConflict(
            f"{prompt.id} is already registered with different text -- "
            "prompts are immutable; bump the version instead of editing it"
        )
    _REGISTRY[prompt.id] = prompt
    return prompt


def get(prompt_id: str) -> VersionedPrompt:
    if prompt_id not in _REGISTRY:
        raise KeyError(f"no prompt {prompt_id!r} (known: {', '.join(sorted(_REGISTRY))})")
    return _REGISTRY[prompt_id]


def registered() -> list[str]:
    return sorted(_REGISTRY)


def lineage(prompt_id: str) -> list[str]:
    """Walk a prompt back to its root ancestor."""
    chain, seen = [], set()
    current: Optional[str] = prompt_id
    while current and current not in seen:
        seen.add(current)
        chain.append(current)
        current = _REGISTRY[current].parent if current in _REGISTRY else None
    return chain
