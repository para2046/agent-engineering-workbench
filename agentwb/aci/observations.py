"""Bounded observations.

Rule from the spec (section 4): never dump thousands of lines into context when
a structured summary will do -- but always keep a way back to the raw data.

Every truncation here writes the full payload under the run's ``raw/``
directory and returns a reference the agent can follow with ``read_raw``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

MAX_CHARS = 4000
MAX_LINES = 120
HEAD_LINES = 80
TAIL_LINES = 30


class RawStore:
    """Spillover storage for observations too large to show in full."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, text: str, hint: str = "obs") -> str:
        digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:10]
        ref = f"{hint}_{digest}.txt"
        (self.root / ref).write_text(text, encoding="utf-8")
        return ref

    def get(self, ref: str) -> Optional[str]:
        # never let a ref escape the raw directory
        path = (self.root / Path(ref).name)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")


def bound(
    text: str,
    raw: Optional[RawStore] = None,
    hint: str = "obs",
    max_chars: int = MAX_CHARS,
    max_lines: int = MAX_LINES,
) -> tuple[str, bool, Optional[str]]:
    """Return ``(shown, truncated, raw_ref)``.

    Long output keeps its head and tail -- for test output and stack traces the
    interesting parts live at both ends, and a middle-elision preserves them.
    """
    if len(text) <= max_chars and text.count("\n") + 1 <= max_lines:
        return text, False, None

    ref = raw.put(text, hint) if raw is not None else None
    lines = text.splitlines()
    if len(lines) > max_lines:
        head = lines[:HEAD_LINES]
        tail = lines[-TAIL_LINES:]
        hidden = len(lines) - len(head) - len(tail)
        body = "\n".join(head + [f"... [{hidden} lines elided] ..."] + tail)
    else:
        body = text[:max_chars]
        body += f"\n... [{len(text) - max_chars} characters elided] ..."

    if ref:
        body += f"\n[full output: read_raw(ref=\"{ref}\")]"
    return body, True, ref
