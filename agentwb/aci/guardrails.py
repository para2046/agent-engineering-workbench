"""Guardrails for the agent-computer interface.

These are hard permissions: the optimizer is explicitly forbidden from touching
them (spec section 11), so they live apart from anything a prompt can change.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


class GuardrailViolation(PermissionError):
    """Raised when a tool call would leave the sandbox or run a denied command."""


# Commands that are never run regardless of task configuration. This is a
# backstop for an agent that wanders, not a security boundary against a
# hostile agent -- for that, run the workbench inside a real sandbox.
DENIED_COMMAND_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\{.*\};:",             # fork bomb
    r"\bcurl\b[^|]*\|\s*(ba)?sh", # curl | sh
    r"\bwget\b[^|]*\|\s*(ba)?sh",
    r"\bgit\s+push\b",
    r"\bsudo\b",
]

_DENIED = [re.compile(p, re.I) for p in DENIED_COMMAND_PATTERNS]


def check_command(command: str) -> None:
    for pat in _DENIED:
        if pat.search(command):
            raise GuardrailViolation(
                f"command blocked by guardrail ({pat.pattern}): refusing to run {command!r}"
            )


def resolve_in_workspace(workspace: Path, relative: str) -> Path:
    """Resolve ``relative`` inside ``workspace``, refusing anything that escapes.

    Catches ``..`` traversal, absolute paths and symlinks pointing outward.
    """
    workspace = Path(workspace).resolve()
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise GuardrailViolation(
            f"path {relative!r} resolves outside the workspace"
        ) from exc
    return candidate


def safe_env() -> dict[str, str]:
    """Environment for subprocesses: inherit PATH-ish essentials, drop secrets.

    Provider keys must not leak into an agent-run shell -- the agent has no
    reason to see them and a leaked key in a transcript is permanent.
    """
    keep = {"PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "HOME", "LANG", "PYTHONIOENCODING"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
