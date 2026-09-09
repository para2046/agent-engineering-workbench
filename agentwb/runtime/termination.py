"""Explicit termination conditions (spec section 17).

No loop in this system runs without one. The monitor also counts repeated
actions, which feeds both the NO_NEW_EVIDENCE stop and the REPEATED_ACTION
failure label.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ..types import TerminationReason


class TerminationMonitor:
    def __init__(self, max_steps: int = 12, max_repeats: int = 3, max_consecutive_failures: int = 4):
        self.max_steps = max_steps
        self.max_repeats = max_repeats
        self.max_consecutive_failures = max_consecutive_failures
        self._seen: dict[str, int] = {}
        self.repeat_count = 0
        self._consecutive_failures = 0

    def observe_action(self, tool: str, arguments: dict[str, Any], ok: bool) -> None:
        key = f"{tool}:{json.dumps(arguments, sort_keys=True, default=str)}"
        self._seen[key] = self._seen.get(key, 0) + 1
        if self._seen[key] > 1:
            self.repeat_count += 1
        self._consecutive_failures = 0 if ok else self._consecutive_failures + 1

    def check(self, step_no: int) -> Optional[TerminationReason]:
        """Return a reason to stop, or None to continue."""
        if step_no >= self.max_steps:
            return TerminationReason.MAX_ITERATIONS
        if any(c > self.max_repeats for c in self._seen.values()):
            # the agent is re-running an identical action and learning nothing
            return TerminationReason.NO_NEW_EVIDENCE
        if self._consecutive_failures >= self.max_consecutive_failures:
            return TerminationReason.BLOCKED
        return None
