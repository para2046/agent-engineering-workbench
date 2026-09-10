"""Policy optimization (spec section 11).

    P0 -> eval dataset -> execute -> trajectory + outcome + feedback
       -> optimizer -> candidate P1 -> dev eval -> better?
       -> held-out test -> promotion gate

The optimizer proposes. The dataset and the gate decide. Nothing here can
promote anything -- `optimize()` returns a decision object and leaves the
writing to the caller, so there is exactly one path to a live policy and it
runs through `PromotionGate`.

**The invariant guard is the important part of this module.**

Spec section 11 lists what an optimizer may change (instructions, tool
descriptions, routing, decision rules, demonstrations) and what it must never
silently change (security policy, hard permissions, deterministic phase gates,
human approval requirements). A reflective optimizer rewriting a system prompt
to maximise a success metric has an obvious cheap move available: delete the
sentence telling the agent not to claim unverified success. Scores go up. The
system starts lying. That is not a hypothetical failure mode, it is the most
predictable one, so candidates that drop a protected invariant are rejected
before they are ever scored -- unscored, so a high score can never argue for
them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

from ..judge.client import JudgeClient, JudgeError
from ..prompts import VersionedPrompt, register
from ..types import Task

# Semantic commitments that must survive any rewrite. Matching is on meaning
# carried by keywords rather than exact text, so an optimizer may rephrase but
# cannot drop the commitment.
PROTECTED_INVARIANTS: dict[str, tuple[str, ...]] = {
    "no_unverified_success": ("verif", "observ"),
    "grading_is_external": ("grader", "not decide", "do not decide", "you do not decide"),
}

INVARIANT_DESCRIPTIONS = {
    "no_unverified_success": "the agent must verify with the environment and must not "
                             "report success it has not observed",
    "grading_is_external": "the agent does not decide whether the task passed; graders do",
}

OPTIMIZER_PROMPT = register(VersionedPrompt(
    name="prompt_optimizer",
    version="v1",
    text="""\
You are improving an AI agent's system prompt, given evidence of how it failed.

You may change: instructions, phrasing, ordering, decision rules expressed in
natural language, and worked guidance.

You may NOT weaken or remove:
- the requirement that the agent verify claims against the environment,
- the statement that the agent does not decide whether it passed,
- any security, permission, or human-approval instruction.

A candidate that drops these is rejected before it is scored, so removing them
cannot help you. Do not try.

Work from the failure evidence, not from general prompt-writing advice. If the
evidence shows the agent edited files without running tests, say something that
addresses that specifically.

Reply with a single JSON object and nothing else:

{
  "candidates": [
    {"rationale": "<what failure this addresses and how>", "prompt": "<the full revised prompt>"}
  ]
}""",
))


class InvariantViolation(ValueError):
    """A candidate policy dropped a protected commitment."""


@dataclass
class PolicyCandidate:
    """One proposed replacement for the current policy."""

    name: str
    version: str
    text: str
    parent: str = ""
    optimizer: str = ""
    rationale: str = ""
    rejected_reason: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def viable(self) -> bool:
        return self.rejected_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "parent": self.parent, "optimizer": self.optimizer,
            "rationale": self.rationale, "rejected_reason": self.rejected_reason,
            "viable": self.viable,
        }

    def register(self) -> VersionedPrompt:
        """Record in the immutable prompt registry, keeping the lineage."""
        return register(VersionedPrompt(
            name=self.name, version=self.version, text=self.text,
            parent=self.parent or None, optimizer=self.optimizer or None,
            metadata={"rationale": self.rationale},
        ))


# --------------------------------------------------------------------------
# invariant guard
# --------------------------------------------------------------------------

def check_invariants(text: str,
                     invariants: Optional[dict[str, tuple[str, ...]]] = None) -> list[str]:
    """Return the names of protected invariants missing from ``text``."""
    invariants = PROTECTED_INVARIANTS if invariants is None else invariants
    lowered = (text or "").lower()
    missing = []
    for name, keywords in invariants.items():
        if not any(k in lowered for k in keywords):
            missing.append(name)
    return missing


def guard(candidate: PolicyCandidate,
          invariants: Optional[dict[str, tuple[str, ...]]] = None) -> PolicyCandidate:
    """Mark a candidate rejected if it dropped a protected commitment.

    Runs *before* scoring, deliberately. A candidate rejected here is never
    given a number, so no score can later be pointed at to argue for it.
    """
    missing = check_invariants(candidate.text, invariants)
    if missing:
        detail = "; ".join(INVARIANT_DESCRIPTIONS.get(m, m) for m in missing)
        candidate.rejected_reason = (
            f"dropped protected invariant(s) {', '.join(missing)}: {detail}"
        )
    return candidate


# --------------------------------------------------------------------------
# optimizers
# --------------------------------------------------------------------------

class Optimizer(Protocol):
    name: str

    def propose(self, baseline: VersionedPrompt, feedback: str, n: int) -> list[PolicyCandidate]:
        ...


class ReflectiveOptimizer:
    """Proposes prompt revisions from failure evidence.

    This is the GEPA-shaped approach -- reflect on what went wrong, rewrite the
    instruction -- without the dependency. `gepa_optimizer` and `dspy_optimizer`
    plug the real libraries into the same interface when they are installed.
    """

    name = "reflective"

    def __init__(self, judge: JudgeClient, version_prefix: str = "opt"):
        self.judge = judge
        self.version_prefix = version_prefix

    def propose(self, baseline: VersionedPrompt, feedback: str, n: int = 2) -> list[PolicyCandidate]:
        user = (
            f"CURRENT PROMPT ({baseline.id}):\n{baseline.text}\n\n"
            f"FAILURE EVIDENCE FROM RECENT RUNS:\n{feedback}\n\n"
            f"Propose {n} revised prompt(s)."
        )
        try:
            reply = self.judge.ask_json(OPTIMIZER_PROMPT.text, user,
                                        prompt_id=OPTIMIZER_PROMPT.id)
        except JudgeError:
            # An optimizer that cannot be parsed proposes nothing. It never
            # falls back to editing the prompt itself.
            return []

        out: list[PolicyCandidate] = []
        for i, raw in enumerate(reply.data.get("candidates") or []):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("prompt") or "").strip()
            if not text:
                continue
            out.append(guard(PolicyCandidate(
                name=baseline.name,
                version=f"{self.version_prefix}{_next_suffix(baseline.version)}-{i + 1}",
                text=text,
                parent=baseline.id,
                optimizer=self.name,
                rationale=str(raw.get("rationale") or "")[:600],
            )))
        return out


def _next_suffix(version: str) -> str:
    m = re.search(r"(\d+)", version or "")
    return str(int(m.group(1)) + 1) if m else "1"


# --------------------------------------------------------------------------
# feedback assembly
# --------------------------------------------------------------------------

def feedback_from_analyses(analyses: Iterable[Any], limit: int = 8) -> str:
    """Turn stored failure analyses into optimizer input.

    Deterministic signals lead. A model handed raw transcripts invents a tidy
    story; a model handed "edited files, never ran tests, twice" is reading.
    """
    lines: list[str] = []
    for a in list(analyses)[:limit]:
        causes = getattr(a, "root_causes", None) or []
        task_id = getattr(a, "task_id", "?")
        lines.append(f"- task {task_id}: " + "; ".join(str(c) for c in causes[:3]))
        fix = getattr(a, "proposed_fix", "")
        if fix:
            lines.append(f"  suggested: {fix}")
    return "\n".join(lines) or "(no failure analyses available)"


# --------------------------------------------------------------------------
# the optimization run
# --------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    baseline_id: str = ""
    candidates: list[PolicyCandidate] = field(default_factory=list)
    rejected: list[PolicyCandidate] = field(default_factory=list)
    evaluated: dict[str, Any] = field(default_factory=dict)   # candidate id -> decision
    promoted: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_id,
            "promoted": self.promoted,
            "candidates": [c.to_dict() for c in self.candidates],
            "rejected": [c.to_dict() for c in self.rejected],
            "evaluated": {k: v.to_dict() if hasattr(v, "to_dict") else v
                          for k, v in self.evaluated.items()},
            "notes": self.notes,
        }

    def summary(self) -> str:
        if self.promoted:
            return f"promoted {self.promoted}"
        if self.rejected and not self.candidates:
            return f"no viable candidates ({len(self.rejected)} rejected by the invariant guard)"
        return "no candidate cleared the promotion gate; baseline stands"


def optimize(
    baseline: VersionedPrompt,
    optimizer: Optimizer,
    feedback: str,
    evaluate: Callable[[Optional[PolicyCandidate], list[Task]], Any],
    gate,
    dev_tasks: list[Task],
    test_tasks: Optional[list[Task]] = None,
    regression_tasks: Optional[list[Task]] = None,
    n_candidates: int = 2,
) -> OptimizationResult:
    """Propose, screen, evaluate, and gate. Promotes nothing itself.

    ``evaluate(candidate_or_None, tasks) -> SplitResult`` runs a policy over a
    split; passing None means the baseline policy.
    """
    result = OptimizationResult(baseline_id=baseline.id)

    proposed = optimizer.propose(baseline, feedback, n_candidates)
    if not proposed:
        result.notes.append("optimizer proposed no candidates")
        return result

    for c in proposed:
        (result.candidates if c.viable else result.rejected).append(c)
    for c in result.rejected:
        result.notes.append(f"{c.id} rejected before scoring: {c.rejected_reason}")

    if not result.candidates:
        return result

    baseline_dev = evaluate(None, dev_tasks)
    baseline_test = evaluate(None, test_tasks) if test_tasks else None
    baseline_reg = evaluate(None, regression_tasks) if regression_tasks else None

    for candidate in result.candidates:
        decision = gate.evaluate(
            baseline_dev=baseline_dev,
            candidate_dev=evaluate(candidate, dev_tasks),
            baseline_test=baseline_test,
            candidate_test=evaluate(candidate, test_tasks) if test_tasks else None,
            baseline_regression=baseline_reg,
            candidate_regression=evaluate(candidate, regression_tasks) if regression_tasks else None,
            baseline_policy=baseline.id,
            candidate_policy=candidate.id,
        )
        result.evaluated[candidate.id] = decision
        if decision.promote and result.promoted is None:
            candidate.register()          # lineage recorded; parent never overwritten
            result.promoted = candidate.id

    return result
