"""Disagreement resolution (spec sections 15 and 16).

    two hypotheses -> can existing evidence resolve it?
        yes -> check the evidence
        no  -> construct a discriminative experiment -> run it in the real
               environment -> decide on the result

**Agents do not vote on truth.** Not by majority, not by confidence, not by
which one argued better. Nothing in this module counts agents, and the judge is
never asked which agent is right — it is asked *what observation would tell the
two hypotheses apart*. That is a question a model can answer usefully, because
the answer is checkable. "Who is correct?" is not, and asking it produces a
confident tiebreak with nothing underneath.

Decision priority (spec section 16), highest first:

    primary evidence > environment outcome > tests > predefined metric
                     > judge > human

The judge appears second-to-last on purpose, and only ever as the designer of
an experiment or the reader of an ambiguous result — never as the arbiter.
When nothing above it can settle the question, the honest outcome is
HUMAN_REQUIRED rather than a coin flip dressed as a verdict.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..judge.client import JudgeClient, JudgeError
from ..prompts import VersionedPrompt, register
from .schemas import AgentMessage, Evidence, Verification


class Resolution(str, enum.Enum):
    NO_DISAGREEMENT = "NO_DISAGREEMENT"
    RESOLVED_BY_EVIDENCE = "RESOLVED_BY_EVIDENCE"
    RESOLVED_BY_EXPERIMENT = "RESOLVED_BY_EXPERIMENT"
    UNRESOLVED = "UNRESOLVED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"


EXPERIMENT_DESIGNER = register(VersionedPrompt(
    name="disagreement_experiment_designer",
    version="v1",
    text="""\
Two agents disagree. Your job is NOT to decide which is right. Do not say which
one you find more convincing, and do not weigh how confidently each was stated.

Your job is to design the cheapest observation that would come out differently
depending on which hypothesis holds.

A good experiment names something checkable in the environment -- a command to
run, a file to inspect, a metric to read -- and states, before it is run, what
each hypothesis predicts it will show. If both hypotheses predict the same
result, the experiment is worthless no matter how sophisticated it looks.

If no available observation would distinguish them, say so plainly. That is a
useful answer; a fabricated experiment is not.

Reply with a single JSON object and nothing else:

{
  "discriminative": <true|false>,
  "experiment": "<concrete action to take, or empty if none exists>",
  "metric": "<what to read from it>",
  "expected_if_a": "<what hypothesis A predicts>",
  "expected_if_b": "<what hypothesis B predicts>",
  "reasoning": "<one or two sentences>"
}""",
))

RESULT_READER = register(VersionedPrompt(
    name="disagreement_result_reader",
    version="v1",
    text="""\
An experiment was run to distinguish two hypotheses. You are reading its output
against the predictions that were written down BEFORE it ran.

Match the observation to a prediction. Do not reason about which hypothesis is
more plausible in general -- only about which prediction the observation
actually matches.

If the observation matches neither prediction, or matches both, say UNCLEAR.
An experiment that failed to discriminate is a fact worth reporting, and
guessing here would destroy the only reason the experiment was run.

Reply with a single JSON object and nothing else:

{
  "supports": "A" | "B" | "UNCLEAR",
  "reasoning": "<one or two sentences citing the observation>"
}""",
))


@dataclass
class Disagreement:
    task_id: str
    a: AgentMessage
    b: AgentMessage

    @property
    def agents(self) -> tuple[str, str]:
        return self.a.sender, self.b.sender

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "a": self.a.to_dict(), "b": self.b.to_dict()}


@dataclass
class DisagreementOutcome:
    resolution: Resolution
    winner: Optional[str] = None          # message_id of the surviving hypothesis
    winning_agent: Optional[str] = None
    basis: str = ""                       # which rung of the priority ladder decided
    experiment: Optional[dict[str, Any]] = None
    evidence: list[Evidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.resolution in (Resolution.RESOLVED_BY_EVIDENCE,
                                   Resolution.RESOLVED_BY_EXPERIMENT,
                                   Resolution.NO_DISAGREEMENT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution.value,
            "winner": self.winner,
            "winning_agent": self.winning_agent,
            "basis": self.basis,
            "experiment": self.experiment,
            "evidence": [e.to_dict() for e in self.evidence],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def detect(messages: list[AgentMessage], task_id: str = "") -> Optional[Disagreement]:
    """Find two hypotheses from different agents that assert different things.

    Crude on purpose: it compares claims, not meanings. A false positive costs
    one cheap experiment; a false negative lets two agents proceed on
    incompatible beliefs, which is how a multi-agent run wastes an hour and
    produces a confident wrong answer.
    """
    from .schemas import MessageType

    hypotheses = [m for m in messages if m.type is MessageType.HYPOTHESIS]
    for i, a in enumerate(hypotheses):
        for b in hypotheses[i + 1:]:
            if a.sender == b.sender:
                continue
            if _claims_differ(a.claim, b.claim):
                return Disagreement(task_id=task_id or a.task_id, a=a, b=b)
    return None


def _claims_differ(a: str, b: str) -> bool:
    from ..experience.retrieval import _similarity, _terms

    a_terms, b_terms = _terms(a), _terms(b)
    if not a_terms or not b_terms:
        return False
    return _similarity(a_terms, b_terms) < 0.6


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def resolve(
    disagreement: Disagreement,
    judge: Optional[JudgeClient] = None,
    run_experiment: Optional[Callable[[str], tuple[bool, str]]] = None,
) -> DisagreementOutcome:
    """Work down the priority ladder until something decides, or admit it cannot.

    ``run_experiment(action) -> (ok, observation)`` executes in the real
    environment. Without it, no experiment can be run, and the outcome is
    HUMAN_REQUIRED rather than a judged guess.
    """
    outcome = DisagreementOutcome(resolution=Resolution.UNRESOLVED)

    # Rung 1: does evidence already on the table settle it?
    settled = _resolve_by_existing_evidence(disagreement)
    if settled is not None:
        return settled

    if judge is None:
        outcome.resolution = Resolution.HUMAN_REQUIRED
        outcome.basis = "no judge available to design a discriminating experiment"
        return outcome

    # Rung 2: design an experiment that would come out differently either way.
    design = _design_experiment(disagreement, judge)
    if design is None:
        outcome.resolution = Resolution.HUMAN_REQUIRED
        outcome.basis = "the experiment designer returned nothing usable"
        return outcome

    outcome.experiment = design
    if not design.get("discriminative"):
        outcome.resolution = Resolution.HUMAN_REQUIRED
        outcome.basis = "no available observation distinguishes the two hypotheses"
        outcome.notes.append(str(design.get("reasoning", ""))[:400])
        return outcome

    if run_experiment is None:
        outcome.resolution = Resolution.HUMAN_REQUIRED
        outcome.basis = ("an experiment was designed but no environment was supplied to "
                         "run it in -- refusing to decide on an unrun experiment")
        return outcome

    # Rung 3: run it for real. The environment decides, not the designer.
    ok, observation = run_experiment(str(design.get("experiment", "")))
    outcome.evidence.append(Evidence(source="experiment", observation=observation[:2000]))
    if not ok and not observation.strip():
        outcome.resolution = Resolution.HUMAN_REQUIRED
        outcome.basis = "the experiment could not be executed"
        return outcome

    verdict = _read_result(disagreement, design, observation, judge)
    if verdict not in ("A", "B"):
        outcome.resolution = Resolution.UNRESOLVED
        outcome.basis = "the experiment ran but did not discriminate"
        return outcome

    winner = disagreement.a if verdict == "A" else disagreement.b
    outcome.resolution = Resolution.RESOLVED_BY_EXPERIMENT
    outcome.winner = winner.message_id
    outcome.winning_agent = winner.sender
    outcome.basis = "environment outcome"
    return outcome


def _resolve_by_existing_evidence(d: Disagreement) -> Optional[DisagreementOutcome]:
    """Prefer the hypothesis that is actually grounded.

    Deliberately narrow: this fires only when exactly one side cites evidence at
    all. Weighing evidence quality is a judgement call, and a judgement call
    made here would quietly become the arbiter this module exists to avoid.
    """
    a_has, b_has = bool(d.a.evidence), bool(d.b.evidence)
    if a_has == b_has:
        return None
    winner = d.a if a_has else d.b
    return DisagreementOutcome(
        resolution=Resolution.RESOLVED_BY_EVIDENCE,
        winner=winner.message_id,
        winning_agent=winner.sender,
        basis="primary evidence",
        evidence=list(winner.evidence),
        notes=["one hypothesis cited observations and the other cited none"],
    )


def _design_experiment(d: Disagreement, judge: JudgeClient) -> Optional[dict[str, Any]]:
    user = (
        f"HYPOTHESIS A (from {d.a.sender}): {d.a.claim}\n"
        f"  its own stated check: {_verification_line(d.a.verification)}\n"
        f"  unknowns: {', '.join(d.a.unknowns) or 'none stated'}\n\n"
        f"HYPOTHESIS B (from {d.b.sender}): {d.b.claim}\n"
        f"  its own stated check: {_verification_line(d.b.verification)}\n"
        f"  unknowns: {', '.join(d.b.unknowns) or 'none stated'}\n\n"
        f"Design the cheapest observation that would distinguish them."
    )
    try:
        reply = judge.ask_json(EXPERIMENT_DESIGNER.text, user, prompt_id=EXPERIMENT_DESIGNER.id)
    except JudgeError:
        return None

    data = reply.data
    experiment = str(data.get("experiment") or "").strip()
    discriminative = bool(data.get("discriminative")) and bool(experiment)

    # A design whose two predictions match discriminates nothing, whatever the
    # model asserted about itself.
    if discriminative:
        v = Verification(
            metric=str(data.get("metric") or ""),
            expected_if_correct=str(data.get("expected_if_a") or ""),
            expected_if_wrong=str(data.get("expected_if_b") or ""),
        )
        if not v.is_discriminative():
            discriminative = False

    return {
        "discriminative": discriminative,
        "experiment": experiment,
        "metric": str(data.get("metric") or ""),
        "expected_if_a": str(data.get("expected_if_a") or ""),
        "expected_if_b": str(data.get("expected_if_b") or ""),
        "reasoning": str(data.get("reasoning") or "")[:600],
        "designer_prompt": EXPERIMENT_DESIGNER.id,
    }


def _read_result(d: Disagreement, design: dict[str, Any], observation: str,
                 judge: JudgeClient) -> str:
    user = (
        f"HYPOTHESIS A: {d.a.claim}\n"
        f"  predicted: {design.get('expected_if_a')}\n\n"
        f"HYPOTHESIS B: {d.b.claim}\n"
        f"  predicted: {design.get('expected_if_b')}\n\n"
        f"EXPERIMENT RUN: {design.get('experiment')}\n"
        f"OBSERVED:\n{judge.truncate_evidence(observation)}"
    )
    try:
        reply = judge.ask_json(RESULT_READER.text, user, prompt_id=RESULT_READER.id)
    except JudgeError:
        return "UNCLEAR"
    return str(reply.data.get("supports") or "UNCLEAR").upper()


def _verification_line(v: Optional[Verification]) -> str:
    if v is None:
        return "none stated"
    return (f"{v.metric}: correct -> {v.expected_if_correct}; "
            f"wrong -> {v.expected_if_wrong}")
