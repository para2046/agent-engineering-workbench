"""The promotion gate (spec section 28).

    candidate -> dev improvement -> held-out test -> regression suite
              -> cost/latency check -> promotion gate

An optimized policy does not replace the current one because it looked better.
It replaces it because it cleared every one of these, in order, on evidence.

Built before any optimizer exists, deliberately. A gate added afterwards is a
gate someone has already worked around; a gate that is the only path to
promotion cannot be skipped by the thing it is meant to check.

Four properties worth stating outright:

**Zero regressions is not negotiable.** A candidate that wins on average while
breaking a previously-passing task has traded a known-good behaviour for an
average, and averages are where regressions go to hide.

**An unmeasured cost blocks a cost ceiling.** If a ceiling is configured and
cost is UNKNOWN, the gate refuses. Treating "not measured" as zero would let an
unpriced model past a spend limit for being unpriced.

**Small samples get an explicit warning, not a silent pass.** Model execution is
stochastic; a two-point improvement over eleven tasks is noise wearing a result.

**Every decision carries its reasons.** A gate that says only yes or no cannot
be argued with, and a promotion nobody can audit is a promotion nobody can undo
with confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ..costs import PriceBook


@dataclass
class SplitResult:
    """Aggregate outcome of running one policy over one split."""

    split: str
    tasks: int = 0
    trials: int = 0
    passed: int = 0
    total_cost: Optional[float] = None      # None = unmeasured, never assume 0
    total_latency_ms: int = 0
    passed_task_ids: set[str] = field(default_factory=set)
    failed_task_ids: set[str] = field(default_factory=set)

    @property
    def success_rate(self) -> float:
        return self.passed / self.trials if self.trials else 0.0

    @property
    def cost_per_trial(self) -> Optional[float]:
        if self.total_cost is None or not self.trials:
            return None
        return self.total_cost / self.trials

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "tasks": self.tasks,
            "trials": self.trials,
            "passed": self.passed,
            "success_rate": round(self.success_rate, 4),
            "total_cost": self.total_cost,
            "cost_per_trial": self.cost_per_trial,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed,
                "blocking": self.blocking, "detail": self.detail}


@dataclass
class PromotionDecision:
    promote: bool = False
    checks: list[GateCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    baseline_policy: str = ""
    candidate_policy: str = ""

    @property
    def blockers(self) -> list[GateCheck]:
        return [c for c in self.checks if c.blocking and not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "baseline_policy": self.baseline_policy,
            "candidate_policy": self.candidate_policy,
            "checks": [c.to_dict() for c in self.checks],
            "blockers": [c.name for c in self.blockers],
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        if self.promote:
            return f"PROMOTE {self.candidate_policy or 'candidate'}"
        names = ", ".join(c.name for c in self.blockers) or "no evidence of improvement"
        return f"REJECT {self.candidate_policy or 'candidate'} -- blocked by: {names}"


@dataclass
class GateConfig:
    min_dev_improvement: float = 0.0      # candidate must beat baseline on dev by at least this
    require_test_not_worse: bool = True   # held-out must not regress
    max_regressions: int = 0              # previously-passing tasks now failing
    max_cost_per_trial: Optional[float] = None
    max_latency_ms_per_trial: Optional[int] = None
    min_trials_for_confidence: int = 20   # below this, results are flagged as noisy


class PromotionGate:
    def __init__(self, config: Optional[GateConfig] = None,
                 prices: Optional[PriceBook] = None):
        self.config = config or GateConfig()
        self.prices = prices

    def evaluate(
        self,
        baseline_dev: SplitResult,
        candidate_dev: SplitResult,
        baseline_test: Optional[SplitResult] = None,
        candidate_test: Optional[SplitResult] = None,
        baseline_regression: Optional[SplitResult] = None,
        candidate_regression: Optional[SplitResult] = None,
        baseline_policy: str = "",
        candidate_policy: str = "",
    ) -> PromotionDecision:
        cfg = self.config
        d = PromotionDecision(baseline_policy=baseline_policy,
                              candidate_policy=candidate_policy)

        # 1. dev improvement -------------------------------------------------
        delta = candidate_dev.success_rate - baseline_dev.success_rate
        d.checks.append(GateCheck(
            name="dev_improvement",
            passed=delta > cfg.min_dev_improvement,
            detail=f"dev success {baseline_dev.success_rate:.3f} -> "
                   f"{candidate_dev.success_rate:.3f} (delta {delta:+.3f}, "
                   f"required > {cfg.min_dev_improvement:+.3f})",
        ))

        # 2. held-out test ---------------------------------------------------
        if cfg.require_test_not_worse:
            if candidate_test is None or baseline_test is None:
                d.checks.append(GateCheck(
                    name="held_out_test",
                    passed=False,
                    detail="no held-out test results supplied -- a candidate tuned on dev "
                           "has not been shown to generalise, so it cannot be promoted",
                ))
            else:
                test_delta = candidate_test.success_rate - baseline_test.success_rate
                d.checks.append(GateCheck(
                    name="held_out_test",
                    passed=test_delta >= 0,
                    detail=f"test success {baseline_test.success_rate:.3f} -> "
                           f"{candidate_test.success_rate:.3f} (delta {test_delta:+.3f})",
                ))

        # 3. regressions -----------------------------------------------------
        regressed = self._regressions(baseline_dev, candidate_dev)
        if baseline_test and candidate_test:
            regressed |= self._regressions(baseline_test, candidate_test)
        if baseline_regression and candidate_regression:
            regressed |= self._regressions(baseline_regression, candidate_regression)
            d.checks.append(GateCheck(
                name="regression_suite",
                passed=candidate_regression.success_rate >= baseline_regression.success_rate,
                detail=f"regression suite {baseline_regression.success_rate:.3f} -> "
                       f"{candidate_regression.success_rate:.3f}",
            ))
        elif baseline_regression or candidate_regression:
            d.warnings.append("regression suite results supplied for only one side; "
                              "the suite comparison was skipped")

        d.checks.append(GateCheck(
            name="no_regressions",
            passed=len(regressed) <= cfg.max_regressions,
            detail=(f"{len(regressed)} task(s) passed at baseline and fail as candidate"
                    + (f": {', '.join(sorted(regressed)[:8])}" if regressed else "")),
        ))

        # 4. cost ------------------------------------------------------------
        if cfg.max_cost_per_trial is not None:
            cost = candidate_dev.cost_per_trial
            if cost is None:
                d.checks.append(GateCheck(
                    name="cost_ceiling",
                    passed=False,
                    detail="a cost ceiling is configured but candidate cost is UNKNOWN "
                           "(no model price set) -- refusing to treat unmeasured as free. "
                           "Configure model_prices in agentwb.json.",
                ))
            else:
                d.checks.append(GateCheck(
                    name="cost_ceiling",
                    passed=cost <= cfg.max_cost_per_trial,
                    detail=f"cost/trial {cost:.6f} vs ceiling {cfg.max_cost_per_trial:.6f}",
                ))

        # 5. latency ---------------------------------------------------------
        if cfg.max_latency_ms_per_trial is not None and candidate_dev.trials:
            per_trial = candidate_dev.total_latency_ms / candidate_dev.trials
            d.checks.append(GateCheck(
                name="latency_ceiling",
                passed=per_trial <= cfg.max_latency_ms_per_trial,
                detail=f"latency/trial {per_trial:.0f}ms vs ceiling "
                       f"{cfg.max_latency_ms_per_trial}ms",
            ))

        # -- warnings (non-blocking) -----------------------------------------
        if candidate_dev.trials < cfg.min_trials_for_confidence:
            d.warnings.append(
                f"only {candidate_dev.trials} dev trial(s); model execution is stochastic, "
                f"so a delta this size is not distinguishable from noise below "
                f"~{cfg.min_trials_for_confidence} trials. Raise --trials before trusting it."
            )
        if baseline_dev.trials and candidate_dev.trials and \
                baseline_dev.trials != candidate_dev.trials:
            d.warnings.append(
                f"unequal trial counts (baseline {baseline_dev.trials}, "
                f"candidate {candidate_dev.trials}) -- success rates are comparable but "
                f"totals are not"
            )
        if 0 < abs(delta) < _noise_floor(candidate_dev.trials):
            d.warnings.append(
                f"dev delta {delta:+.3f} is within the noise floor for "
                f"{candidate_dev.trials} trials"
            )

        d.promote = not d.blockers
        return d

    @staticmethod
    def _regressions(baseline: SplitResult, candidate: SplitResult) -> set[str]:
        """Tasks that passed at baseline and fail as candidate."""
        return baseline.passed_task_ids & candidate.failed_task_ids


def _noise_floor(trials: int) -> float:
    """Rough one-sigma binomial spread at p=0.5 -- a sanity bar, not statistics.

    Real confidence intervals arrive with a proper statistical layer; until then
    this exists to stop a two-trial swing being read as an improvement.
    """
    if trials <= 0:
        return 1.0
    return 0.5 / math.sqrt(trials)


def result_from_task_results(split: str, task_results, prices: Optional[PriceBook] = None) -> SplitResult:
    """Build a SplitResult from harness TaskResult objects."""
    out = SplitResult(split=split)
    for tr in task_results:
        out.tasks += 1
        for trial in tr.trials:
            traj = trial.trajectory
            out.trials += 1
            metrics = traj.metrics or {}
            out.total_latency_ms += int(metrics.get("latency_ms") or 0)

            cost = metrics.get("estimated_cost_usd")
            if cost is None and prices is not None:
                cost = prices.estimate(traj.model,
                                       metrics.get("input_tokens") or 0,
                                       metrics.get("output_tokens") or 0)
            if cost is not None:
                out.total_cost = (out.total_cost or 0.0) + cost

            if traj.evaluation and traj.evaluation.passed:
                out.passed += 1
                out.passed_task_ids.add(traj.task_id)
            else:
                out.failed_task_ids.add(traj.task_id)

    # A task that passed one trial and failed another is not a clean pass.
    out.passed_task_ids -= out.failed_task_ids
    return out
