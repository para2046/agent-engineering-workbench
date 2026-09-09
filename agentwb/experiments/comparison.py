"""Standardised run comparison (spec section 21).

The job here is less "compute a diff" than "stop people claiming an improvement
they did not measure". When more than one variable differs between baseline and
candidate, the comparison is flagged CONFOUNDED_EXPERIMENT and the metric delta
is reported as not attributable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..types import Trajectory

# Configuration axes that, if they differ, make a metric delta unattributable.
CONTROLLED_AXES = ("task_id", "provider", "model", "prompt_version")

METRICS = ("steps", "tool_calls", "tool_failures", "latency_ms", "input_tokens", "output_tokens")

# For these, lower is better.
LOWER_IS_BETTER = {"steps", "tool_calls", "tool_failures", "latency_ms",
                   "input_tokens", "output_tokens"}


@dataclass
class Comparison:
    baseline_id: str
    candidate_id: str
    config_differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    metric_differences: dict[str, dict[str, Any]] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confounded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_id,
            "candidate": self.candidate_id,
            "confounded": self.confounded,
            "config_differences": {k: {"baseline": a, "candidate": b}
                                   for k, (a, b) in self.config_differences.items()},
            "outcome": self.outcome,
            "metric_differences": self.metric_differences,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "warnings": self.warnings,
        }


def compare(baseline: Trajectory, candidate: Trajectory) -> Comparison:
    cmp = Comparison(baseline_id=baseline.trajectory_id, candidate_id=candidate.trajectory_id)

    for axis in CONTROLLED_AXES:
        a, b = getattr(baseline, axis, None), getattr(candidate, axis, None)
        if a != b:
            cmp.config_differences[axis] = (a, b)

    changed = [k for k in cmp.config_differences if k != "task_id"]
    if "task_id" in cmp.config_differences:
        cmp.warnings.append(
            "baseline and candidate ran different tasks -- these results are not comparable"
        )
        cmp.confounded = True
    if len(changed) > 1:
        cmp.confounded = True
        cmp.warnings.append(
            f"CONFOUNDED_EXPERIMENT: {len(changed)} variables changed at once "
            f"({', '.join(changed)}); a metric delta cannot be attributed to any one of them"
        )

    b_ev, c_ev = baseline.evaluation, candidate.evaluation
    cmp.outcome = {
        "baseline_passed": bool(b_ev and b_ev.passed),
        "candidate_passed": bool(c_ev and c_ev.passed),
        "baseline_score": b_ev.score if b_ev else None,
        "candidate_score": c_ev.score if c_ev else None,
    }
    if cmp.outcome["candidate_passed"] and not cmp.outcome["baseline_passed"]:
        cmp.improvements.append("outcome: FAIL -> PASS")
    if cmp.outcome["baseline_passed"] and not cmp.outcome["candidate_passed"]:
        cmp.regressions.append("outcome: PASS -> FAIL")

    for key in METRICS:
        a = (baseline.metrics or {}).get(key)
        b = (candidate.metrics or {}).get(key)
        if a is None or b is None:
            continue
        delta = b - a
        pct = (delta / a * 100.0) if a else None
        better = (delta < 0) if key in LOWER_IS_BETTER else (delta > 0)
        cmp.metric_differences[key] = {
            "baseline": a, "candidate": b, "delta": delta,
            "pct": round(pct, 1) if pct is not None else None,
            "direction": "better" if delta and better else ("worse" if delta else "same"),
        }
        if delta and not cmp.confounded:
            (cmp.improvements if better else cmp.regressions).append(
                f"{key}: {a} -> {b} ({delta:+})"
            )

    if _single_trial(baseline) and _single_trial(candidate):
        cmp.warnings.append(
            "single trial per side: model execution is stochastic, so a small delta "
            "here is noise -- raise `trials` before drawing a conclusion"
        )
    return cmp


def _single_trial(t: Trajectory) -> bool:
    return True  # one trajectory is by definition one trial; kept explicit for readability


def summarize_task_results(results: list[Any]) -> dict[str, Any]:
    """Aggregate a list of TaskResult into suite-level numbers."""
    total = sum(len(r.trials) for r in results)
    passed = sum(r.pass_count for r in results)
    return {
        "tasks": len(results),
        "trials": total,
        "passed": passed,
        "success_rate": round(passed / total, 4) if total else 0.0,
        "per_task": {
            r.task_id: {"passed": r.pass_count, "trials": len(r.trials),
                        "success_rate": round(r.success_rate, 4)}
            for r in results
        },
    }
