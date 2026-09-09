"""Single source of truth for failure category names.

Kept apart from types.py so the analyzer can list the taxonomy for a model
without importing the runtime, and so a category added to the enum is
automatically offered to the analyzer.
"""

from __future__ import annotations

from .types import FailureCategory


def taxonomy_names() -> list[str]:
    return [c.value for c in FailureCategory]


def is_valid(name: str) -> bool:
    return name in {c.value for c in FailureCategory}
