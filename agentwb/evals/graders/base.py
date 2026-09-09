"""Grader interface and registry.

Priority order (spec section 7), highest first:

  1. deterministic outcome verification   <- V0 implements this
  2. deterministic trajectory checks      <- V0 implements this
  3. model-based grading                  <- V1+
  4. human review                         <- V1+

If it can be checked in code, do not ask a model. A grader returns structured
evidence and may answer UNKNOWN when the evidence does not support a verdict --
an honest UNKNOWN is worth more than a confident guess, because it tells you the
eval itself needs work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ...types import GraderResult, GraderSpec, GraderVerdict, Task, Trajectory


@dataclass
class GradingContext:
    """Everything a grader is allowed to look at."""

    task: Task
    trajectory: Trajectory
    workspace: Path


GraderFn = Callable[[GradingContext, dict[str, Any]], GraderResult]

_GRADERS: dict[str, GraderFn] = {}


def grader(name: str) -> Callable[[GraderFn], GraderFn]:
    def deco(fn: GraderFn) -> GraderFn:
        _GRADERS[name] = fn
        return fn
    return deco


def get_grader(name: str) -> Optional[GraderFn]:
    return _GRADERS.get(name)


def registered_graders() -> list[str]:
    return sorted(_GRADERS)


def run_grader(spec: GraderSpec, ctx: GradingContext) -> GraderResult:
    """Run one grader, converting any crash into a GRADER_FAILURE result.

    A broken grader must never look like a failing agent -- that is how eval
    suites quietly start lying to you.
    """
    fn = get_grader(spec.type)
    if fn is None:
        return GraderResult(
            grader=spec.label,
            verdict=GraderVerdict.UNKNOWN,
            required=spec.required,
            weight=spec.weight,
            error=f"unknown grader type {spec.type!r}",
            evidence=[{"available": registered_graders()}],
        )
    try:
        result = fn(ctx, spec.params)
    except Exception as exc:  # noqa: BLE001
        return GraderResult(
            grader=spec.label,
            verdict=GraderVerdict.UNKNOWN,
            required=spec.required,
            weight=spec.weight,
            error=f"{type(exc).__name__}: {exc}",
        )
    result.grader = spec.label
    result.required = spec.required
    result.weight = spec.weight
    return result


def ok(score: float = 1.0, evidence: Optional[list] = None, uncertainty: float = 0.0) -> GraderResult:
    return GraderResult(grader="", verdict=GraderVerdict.PASS, score=score,
                        evidence=evidence or [], uncertainty=uncertainty)


def bad(score: float = 0.0, evidence: Optional[list] = None, uncertainty: float = 0.0) -> GraderResult:
    return GraderResult(grader="", verdict=GraderVerdict.FAIL, score=score,
                        evidence=evidence or [], uncertainty=uncertainty)


def unknown(evidence: Optional[list] = None, error: Optional[str] = None) -> GraderResult:
    return GraderResult(grader="", verdict=GraderVerdict.UNKNOWN, score=None,
                        evidence=evidence or [], uncertainty=1.0, error=error)
