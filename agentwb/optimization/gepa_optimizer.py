"""GEPA adapter (spec section 11).

GEPA is reflective prompt evolution: mutate an instruction using feedback about
why it failed, keep what scores better. `ReflectiveOptimizer` already implements
that shape with no dependency; this plugs the real library in when it is
installed, behind the same interface.

Optional. Importing this module without `gepa` is fine -- the failure surfaces
when you try to construct the optimizer, saying what to install.

Whatever the library returns is still screened by `guard()` before scoring, and
still has to clear the promotion gate. A candidate from a published optimizer
gets no more trust than one written by hand.
"""

from __future__ import annotations

from typing import Optional

from ..judge.client import JudgeClient
from ..prompts import VersionedPrompt
from ..providers.base import ProviderError
from .optimizer import PolicyCandidate, ReflectiveOptimizer, guard


class GEPAOptimizer:
    name = "gepa"

    def __init__(self, judge: Optional[JudgeClient] = None, **kwargs):
        try:
            import gepa  # noqa: F401,WPS433
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError(
                "GEPA is not installed -- run `pip install gepa`, or use "
                "`--optimizer reflective`, which implements the same "
                "reflective-mutation approach with no dependency"
            ) from exc
        self._judge = judge
        self._kwargs = kwargs

    def propose(self, baseline: VersionedPrompt, feedback: str, n: int = 2) -> list[PolicyCandidate]:  # pragma: no cover - requires gepa
        import gepa

        raw = gepa.optimize(seed_prompt=baseline.text, feedback=feedback,
                            num_candidates=n, **self._kwargs)
        out = []
        for i, text in enumerate(_as_texts(raw)[:n]):
            out.append(guard(PolicyCandidate(
                name=baseline.name, version=f"gepa{i + 1}", text=text,
                parent=baseline.id, optimizer=self.name,
                rationale="proposed by GEPA reflective evolution",
            )))
        return out


def _as_texts(raw) -> list[str]:  # pragma: no cover - shape depends on gepa
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        return [str(v) for v in raw.values() if isinstance(v, str)]
    return [str(c) for c in (raw or [])]
