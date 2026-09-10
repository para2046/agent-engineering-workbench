"""Wiring between :mod:`optimizer` and the eval harness.

``optimize()`` takes an ``evaluate(candidate_or_None, tasks) -> SplitResult``
callable and stays deliberately ignorant of how a policy is actually run. This
module supplies that callable.

The one property it has to guarantee: **only the policy changes between
baseline and candidate.** Same tasks, same tools, same graders, same workspace
construction, same trial count. A score difference that could have come from
two variables is not evidence about either, and `compare` already refuses to
report deltas from confounded runs — an optimizer that produced them would be
manufacturing exactly the input the rest of the system rejects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from ..prompts import VersionedPrompt
from ..types import Task
from .optimizer import PolicyCandidate
from .promotion import SplitResult, result_from_task_results


class PolicyEvaluator:
    """Runs a policy over a set of tasks and aggregates the outcome.

    Callable as ``evaluator(candidate_or_None, tasks)`` so it can be handed
    straight to :func:`optimizer.optimize`. ``None`` means the baseline policy.
    """

    def __init__(self, harness, provider, baseline: VersionedPrompt,
                 trials: int = 1, prices=None, split_name: str = "dev"):
        self.harness = harness
        self.provider = provider
        self.baseline = baseline
        self.trials = trials
        self.prices = prices
        self.split_name = split_name

    def __call__(self, candidate: Optional[PolicyCandidate],
                 tasks: Optional[list[Task]]) -> SplitResult:
        if not tasks:
            return SplitResult(split=self.split_name)

        policy = candidate if candidate is not None else self.baseline
        results = [
            self.harness.run_task(t, self.provider, trials=self.trials, policy=policy)
            for t in tasks
        ]
        return result_from_task_results(self.split_name, results, self.prices)


def load_analyses(store) -> list[Any]:
    """Every stored failure analysis, oldest first.

    Returns FailureAnalysis objects so ``feedback_from_analyses`` can read
    ``root_causes`` and ``proposed_fix`` off them directly.
    """
    from ..analysis import failure_analyzer

    out = []
    root = Path(store.runs_root)
    if not root.is_dir():
        return out
    for run_dir in sorted(root.iterdir()):
        analysis = failure_analyzer.load(run_dir)
        if analysis is not None:
            out.append(analysis)
    return out


def save_optimization_result(result, path: Path) -> Path:
    """Append-only optimization history.

    Rejected candidates are written too. A direction that was tried and failed
    is evidence the next run would otherwise pay full price to rediscover.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_dict(), ensure_ascii=False, default=str) + "\n")
    return path


def load_optimization_history(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
