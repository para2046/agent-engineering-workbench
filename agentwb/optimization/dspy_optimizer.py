"""DSPy adapter (spec section 11).

DSPy compiles declarative LM calls into optimized pipelines, tuning instructions
and few-shot demonstrations against a metric. It optimizes a *program*, not a
free-text prompt, so this adapter is thinner than it looks: it hands DSPy the
train split and a metric, and converts whatever instruction it settles on back
into a PolicyCandidate.

Optional dependency; the same rules apply as everywhere else. Candidates are
screened by `guard()` before scoring and must clear the promotion gate. DSPy
optimizing against a metric is exactly the situation the invariant guard exists
for -- deleting "verify before claiming success" reliably raises a naive
success metric.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..prompts import VersionedPrompt
from ..providers.base import ProviderError
from .optimizer import PolicyCandidate, guard


class DSPyOptimizer:
    name = "dspy"

    def __init__(self, metric: Optional[Callable[..., float]] = None, **kwargs):
        try:
            import dspy  # noqa: F401,WPS433
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError(
                "DSPy is not installed -- run `pip install dspy-ai`, or use "
                "`--optimizer reflective`"
            ) from exc
        if metric is None:
            raise ValueError(
                "DSPy optimizes against an explicit metric; refusing to invent one. "
                "Pass the metric you actually want maximised."
            )
        self.metric = metric
        self._kwargs = kwargs

    def propose(self, baseline: VersionedPrompt, feedback: str, n: int = 2) -> list[PolicyCandidate]:  # pragma: no cover - requires dspy
        import dspy

        compiled = dspy.teleprompt.BootstrapFewShot(metric=self.metric, **self._kwargs)
        instruction = _extract_instruction(compiled, baseline.text)
        return [guard(PolicyCandidate(
            name=baseline.name, version="dspy1", text=instruction,
            parent=baseline.id, optimizer=self.name,
            rationale="compiled by DSPy against the supplied metric",
        ))]


def _extract_instruction(compiled: Any, fallback: str) -> str:  # pragma: no cover
    for attr in ("instructions", "signature", "prompt"):
        value = getattr(compiled, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return fallback
